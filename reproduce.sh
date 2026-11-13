#!/usr/bin/env bash
# Fresh-environment reproduction: installs pinned deps, runs tests, downloads ^NSEI, regenerates all results.
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.run_all --data nifty --out results
