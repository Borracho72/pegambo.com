#!/usr/bin/env bash
# Reproduce all results (CPU, ~10 min on 2 cores, ~2 GB RAM, ~200 MB disk for raw/).
# Setup:  pip install -r requirements.txt   (torch CPU wheel: --index-url https://download.pytorch.org/whl/cpu)
# Usage:  ./run_all.sh        all steps
#         ./run_all.sh 1      only step 1 (format facts, no model download; smoke test)
set -e
cd "$(dirname "$0")"
mkdir -p results raw
python3 s1_format_facts.py        # 1. fp16 format facts + softmax underflow scan
[ "${1:-}" = "1" ] && exit 0
python3 s2_run_model.py           # 2. all precision modes; raw attentions -> raw/*.npz; non-finite log
python3 s2b_diagnose_nan.py       # 2b. inputs/outputs of attention.dense: is the NaN already upstream?
python3 s2c_attn_internals.py     # 2c. q, k, v, QK^T, softmax, A@V: where Inf/NaN first appears
python3 s2d_overflow_location.py  # 2d. fp32 logit magnitudes vs fp16 max (65504), visible vs masked
python3 s3_analyze.py             # 3. per-head / per-layer metrics, M grid
python3 s4_summary.py             # 4. summary tables (markdown/csv)
python3 s5_bins.py                # 5. fp64 top-8 edge weights by fp16/fp32 threshold bins
python3 s6_pooled_knife.py        # 6. pooled fraction of top-8 edges below 2^-14 (threshold on A vs on W)
