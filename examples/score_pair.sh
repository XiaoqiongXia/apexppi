#!/usr/bin/env bash
set -euo pipefail

apexppi-predict \
  --bundle-dir "${APEXPPI_BUNDLE_DIR:-apexppi-bundle-v0.1.0}" \
  --host-uniprot O00170 \
  --pathogen-uniprot Q69027 \
  --device cpu
