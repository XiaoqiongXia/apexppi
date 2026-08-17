import hashlib
import tarfile

from scripts.package_release_assets import package_release_assets


def test_package_release_assets_builds_verified_archive(tiny_bundle, tmp_path):
    output_dir = tmp_path / "artifacts"
    archive_path, checksum_path = package_release_assets(
        checkpoint=tiny_bundle / "models" / "apexppi" / "apexppi_best.pt",
        graph=(
            tiny_bundle
            / "data"
            / "processed"
            / "hpidb_human_ppi_unified_protein_graph"
            / "heterodata_unified_protein.pt"
        ),
        data_dir=tiny_bundle / "data" / "processed" / "hpidb_human_ppi",
        output_dir=output_dir,
    )
    assert archive_path.is_file()
    assert checksum_path.is_file()

    expected_digest = checksum_path.read_text().split()[0]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == expected_digest

    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    root = "apexppi-bundle-v0.1.0"
    assert f"{root}/checksums.sha256" in names
    assert f"{root}/models/apexppi/apexppi_best.pt" in names
