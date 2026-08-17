#!/usr/bin/env bash
set -euo pipefail

apexppi-predict \
  --bundle-dir "${APEXPPI_BUNDLE_DIR:-apexppi-bundle-v0.1.0}" \
  --pathogen-uniprot Q69027 \
  --top-k 50 \
  --output-tsv results/q69027_host_ranking.tsv
