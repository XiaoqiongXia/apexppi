#!/usr/bin/env python3
"""Training loop for the final ApexPPI model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .apexppi import ApexPPI
from .data import (
    build_positives_by_host,
    build_positives_by_pathogen,
    build_supervision_tensors,
    load_graph,
)
from .losses import infonce_loss, sample_negative_hosts, sample_negative_pathogens
from .metrics import compute_binary_metrics


def evaluate_model(
    model: ApexPPI,
    data,
    host_idx: torch.Tensor,
    pathogen_idx: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        protein_points = model.encode(data)
        loader = DataLoader(
            TensorDataset(host_idx, pathogen_idx, labels),
            batch_size=batch_size,
            shuffle=False,
        )
        for batch_host, batch_pathogen, batch_labels in loader:
            logits_all.append(
                model.decode(
                    protein_points,
                    batch_host.to(device),
                    batch_pathogen.to(device),
                ).cpu()
            )
            labels_all.append(batch_labels.cpu())
    logits = torch.cat(logits_all)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits encountered during evaluation")
    return compute_binary_metrics(torch.cat(labels_all), logits)


def train_apexppi(
    data_dir: Path,
    heterodata_path: Path,
    output_dir: Path,
    device_name: str = "cuda",
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    num_layers: int = 2,
    num_negatives: int = 64,
    temperature: float = 0.1,
    p2h_infonce_weight: float = 0.25,
    h2p_infonce_weight: float = 0.25,
    max_tangent_norm: float = 5.0,
    initial_curvature: float = 1.0,
    min_curvature: float = 1e-4,
    grad_clip_norm: float = 1.0,
    patience: int = 15,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(device_name)

    data, node_maps = load_graph(heterodata_path)
    data = data.to(device)
    num_host_candidates = len(node_maps["host_protein"])
    pathogen_candidates = torch.tensor(
        list(node_maps["pathogen_protein"].values()), dtype=torch.long, device=device
    )

    all_positive_edges = pd.read_csv(data_dir / "positive_edges.tsv", sep="\t").assign(label=1)
    positives_by_pathogen = build_positives_by_pathogen(all_positive_edges, node_maps)
    positives_by_host = build_positives_by_host(all_positive_edges, node_maps)

    split_tensors = {}
    for split in ["train", "valid", "test"]:
        edges = pd.read_csv(data_dir / f"{split}_edges.tsv", sep="\t")
        split_tensors[split] = tuple(
            tensor.to(device) for tensor in build_supervision_tensors(edges, node_maps)
        )
    train_host, train_pathogen, train_labels = split_tensors["train"]

    model = ApexPPI(
        edge_types=data.edge_types,
        input_dim=data["protein"].x.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        max_tangent_norm=max_tangent_norm,
        initial_curvature=initial_curvature,
        min_curvature=min_curvature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(train_host, train_pathogen, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_valid_ap = -1.0
    best_epoch = 0
    best_path = output_dir / "apexppi_best.pt"
    history = []
    stopped_reason = None
    completed_epochs = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        bce_losses = []
        p2h_losses = []
        h2p_losses = []
        for batch_host, batch_pathogen, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            protein_points = model.encode(data)
            logits = model.decode(protein_points, batch_host, batch_pathogen)
            if not torch.isfinite(logits).all():
                stopped_reason = f"non_finite_train_logits_epoch_{epoch}"
                break
            bce_loss = bce_loss_fn(logits, batch_labels)

            positive_mask = batch_labels > 0.0
            if bool(positive_mask.any()):
                positive_host = batch_host[positive_mask]
                positive_pathogen = batch_pathogen[positive_mask]
                positive_logits = model.decode(protein_points, positive_host, positive_pathogen)

                negative_hosts = sample_negative_hosts(
                    pathogen_idx=positive_pathogen,
                    positives_by_pathogen=positives_by_pathogen,
                    num_hosts=num_host_candidates,
                    num_negatives=num_negatives,
                    generator=generator,
                )
                flat_negative_hosts = negative_hosts.reshape(-1)
                repeated_pathogen = positive_pathogen.repeat_interleave(num_negatives)
                p2h_negative_logits = model.decode(
                    protein_points, flat_negative_hosts, repeated_pathogen
                ).view(positive_host.shape[0], num_negatives)
                p2h_loss = infonce_loss(
                    positive_logits=positive_logits,
                    negative_logits=p2h_negative_logits,
                    temperature=temperature,
                )

                negative_pathogens = sample_negative_pathogens(
                    host_idx=positive_host,
                    positives_by_host=positives_by_host,
                    pathogen_candidates=pathogen_candidates,
                    num_negatives=num_negatives,
                    generator=generator,
                )
                repeated_host = positive_host.repeat_interleave(num_negatives)
                flat_negative_pathogens = negative_pathogens.reshape(-1)
                h2p_negative_logits = model.decode(
                    protein_points, repeated_host, flat_negative_pathogens
                ).view(positive_host.shape[0], num_negatives)
                h2p_loss = infonce_loss(
                    positive_logits=positive_logits,
                    negative_logits=h2p_negative_logits,
                    temperature=temperature,
                )
            else:
                p2h_loss = torch.zeros((), dtype=bce_loss.dtype, device=device)
                h2p_loss = torch.zeros((), dtype=bce_loss.dtype, device=device)

            loss = (
                bce_loss
                + p2h_infonce_weight * p2h_loss
                + h2p_infonce_weight * h2p_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce_loss.detach().cpu()))
            p2h_losses.append(float(p2h_loss.detach().cpu()))
            h2p_losses.append(float(h2p_loss.detach().cpu()))

        if stopped_reason is not None:
            print(f"[ApexPPI] stopped: {stopped_reason}", flush=True)
            break

        valid_host, valid_pathogen, valid_labels = split_tensors["valid"]
        try:
            valid_metrics = evaluate_model(
                model,
                data,
                valid_host,
                valid_pathogen,
                valid_labels,
                device,
                batch_size=max(batch_size, 4096),
            )
        except FloatingPointError:
            stopped_reason = f"non_finite_validation_logits_epoch_{epoch}"
            print(f"[ApexPPI] stopped: {stopped_reason}", flush=True)
            break

        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "train_bce_loss": sum(bce_losses) / len(bce_losses),
            "train_p2h_infonce_loss": sum(p2h_losses) / len(p2h_losses),
            "train_h2p_infonce_loss": sum(h2p_losses) / len(h2p_losses),
            "curvature": float(model.curvature().detach().cpu()),
            **valid_metrics,
        }
        history.append(row)
        print(
            f"[ApexPPI] epoch={epoch} "
            f"loss={row['train_loss']:.4f} bce={row['train_bce_loss']:.4f} "
            f"p2h={row['train_p2h_infonce_loss']:.4f} "
            f"h2p={row['train_h2p_infonce_loss']:.4f} "
            f"curvature={row['curvature']:.4f} "
            f"valid_ap={valid_metrics['average_precision']:.4f} "
            f"valid_auc={valid_metrics['roc_auc']:.4f}",
            flush=True,
        )
        if valid_metrics["average_precision"] > best_valid_ap:
            best_valid_ap = valid_metrics["average_precision"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": "apexppi",
                    "input_dim": int(data["protein"].x.shape[1]),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "num_layers": num_layers,
                    "max_tangent_norm": max_tangent_norm,
                    "initial_curvature": initial_curvature,
                    "min_curvature": min_curvature,
                    "decoder_type": "gated_bilinear",
                    "learned_curvature": float(model.curvature().detach().cpu()),
                    "heterodata_path": str(heterodata_path),
                    "node_types": list(data.node_types),
                    "edge_types": [tuple(edge_type) for edge_type in data.edge_types],
                    "training_objective": "bce_plus_bidirectional_infonce",
                    "num_negatives": num_negatives,
                    "temperature": temperature,
                    "p2h_infonce_weight": p2h_infonce_weight,
                    "h2p_infonce_weight": h2p_infonce_weight,
                    "num_host_candidates": num_host_candidates,
                    "num_pathogen_candidates": int(pathogen_candidates.numel()),
                    "relation_aggregation": "attention",
                    "relation_attention": model.relation_attention(),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        completed_epochs = epoch
        if patience > 0 and epochs_without_improvement >= patience:
            stopped_reason = f"early_stopping_no_valid_ap_improvement_{patience}_epochs"
            print(f"[ApexPPI] stopped: {stopped_reason}", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError("No finite validation checkpoint was saved")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_batch_size = max(batch_size, 4096)
    train_metrics = evaluate_model(
        model, data, *split_tensors["train"], device, eval_batch_size
    )
    valid_metrics = evaluate_model(
        model, data, *split_tensors["valid"], device, eval_batch_size
    )
    test_metrics = evaluate_model(
        model, data, *split_tensors["test"], device, eval_batch_size
    )
    metrics = {
        "model": "ApexPPI",
        "manifold": "lorentz_learnable_global_curvature",
        "relation_aggregation": "attention",
        "decoder_type": "gated_bilinear",
        "relation_attention": model.relation_attention(),
        "initial_curvature": initial_curvature,
        "min_curvature": min_curvature,
        "learned_curvature": float(model.curvature().detach().cpu()),
        "training_objective": "bce_plus_bidirectional_infonce",
        "device": str(device),
        "best_epoch": best_epoch,
        "input_dim": int(data["protein"].x.shape[1]),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "num_layers": num_layers,
        "max_tangent_norm": max_tangent_norm,
        "grad_clip_norm": grad_clip_norm,
        "early_stopping_metric": "valid_average_precision",
        "patience": patience,
        "num_negatives": num_negatives,
        "temperature": temperature,
        "p2h_infonce_weight": p2h_infonce_weight,
        "h2p_infonce_weight": h2p_infonce_weight,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "stopped_reason": stopped_reason,
        "node_types": list(data.node_types),
        "edge_types": [str(edge_type) for edge_type in data.edge_types],
        "protein_nodes": int(data["protein"].num_nodes),
        "host_candidates": num_host_candidates,
        "pathogen_candidates": int(pathogen_candidates.numel()),
        "train_edges": int(len(split_tensors["train"][2])),
        "valid_edges": int(len(split_tensors["valid"][2])),
        "test_edges": int(len(split_tensors["test"][2])),
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "best_model_path": str(best_path),
    }
    pd.DataFrame(history).to_csv(
        output_dir / "apexppi_history.tsv",
        sep="\t",
        index=False,
    )
    (output_dir / "apexppi_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    return metrics
