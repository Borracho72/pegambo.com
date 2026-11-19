Independent check of fp16 numerical behaviour in Pythia-160M attention graphs
=============================================================================

WHAT IS MEASURED
EleutherAI/pythia-160m (revision 50f5173d932e8e61f858120bcb800b97af589f46; 12 layers x 12 heads)
is run on three probe texts (texts/: prose 172 tokens, code 233, arithmetic 181) in several
precision modes. For every layer and head we record:
 - where Inf/NaN first appears (module hooks plus instrumentation of q, k, QK^T, softmax, A@V);
 - the fraction of exact zeros and the smallest non-zero attention weights A;
 - the graph: W = (A + A^T)/2 without the diagonal, top-8 neighbours per node; the fraction of edges
   below 2^-14 with the threshold applied to A (before symmetrisation) and to W; edge agreement
   with fp32 (Jaccard); number of connected components;
 - M = log10(w_min/eps) for 5 explicit definitions of w_min and eps in {2^-14, 2^-24, 2^-126, 2^-149}.

MODES (s2_run_model.py)
 fp32              reference
 fp64_true         model.double(), softmax in fp64 (control)
 fp64              model.double() with the stock HF eager attention (its softmax runs in fp32)
 fp16              model.half(), stock HF eager attention
 fp16_puresoftmax  as fp16, softmax in fp16
 fp16_prescale     as fp16, q multiplied by the scale before the matmul
 fp16_upcastQK     fp16 model, QK^T and softmax in fp32, A stored in fp16 (isolates fp16 storage of A)
 fp16_sdpa         stock SDPA (A is not returned; only finiteness is checked)

HOW TO RUN
 pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
 pip install -r requirements.txt
 ./run_all.sh       full run, ~10 min on 2 CPU cores, ~2 GB RAM; raw A tensors go to raw/ (~160 MB)
 ./run_all.sh 1     step 1 only (format facts, no model download)
Environment of the published results: environment.txt (Python 3.14.8, torch 2.15.0+cpu,
transformers 5.23.0, Intel Xeon with AVX-512, 2 threads). The computation is deterministic
(eval mode, no dropout).

KEY RESULTS (results/)
1. fp16 format (s1_format_facts.json): smallest normal 2^-14 = 6.10e-5, smallest subnormal
   2^-24 = 5.96e-8, epsilon 2^-10, maximum 65504. fp16 softmax on CPU keeps subnormal values;
   exact zero only for values <= 2^-25 ~ 2.98e-8.
2. Inf/NaN (summary_nonfinite.csv, s2c_attn_internals.json, s2d_overflow_location.json): in fp16
   the unscaled q·k^T exceeds 65504 (in fp32 up to 79305 in layer 4 on prose), producing +Inf;
   softmax returns NaN, which propagates. Fully NaN: layers 5–11 (prose, code), 6–11 (arithmetic);
   layers counted from zero. fp32, fp64, fp16_upcastQK and fp16_sdpa are finite throughout;
   fp16_prescale postpones the failure to layer 10.
3. Fraction of top-8 edges below 2^-14 in layers 8–11 (s6_pooled_knife.csv; pooled over heads and
   nodes; fp64_true and fp32 agree to <= 0.5 percentage points):
      threshold on A: prose 43.2–71.7 %, code 33.7–65.8 %, arithmetic 27.4–70.8 %
      threshold on W: prose 46.3–72.6 %, code 37.2–67.2 %, arithmetic 30.2–72.2 %
   Actually zeroed when A is stored in fp16 (fp16_upcastQK): 7.9–61.8 %; the rest are kept as
   subnormals.
4. Unthresholded top-8 graphs are connected (1 component) in every layer of every finite mode
   (per_layer.csv, n_comp_top8). In NaN layers the graph is undefined; if NaN is replaced by zero,
   all edges are zero.
5. M (M_grid.csv, M_ranges.csv) varies by tens of orders of magnitude with w_min and eps.
   In fp16 with w_min = smallest non-zero weight and eps = 2^-24, M = 0.00 in every layer.

FILES
 s1…s6, run_all.sh   scripts (order and purpose listed in run_all.sh)
 texts/              probe texts
 results/per_head.csv, per_layer.csv   all metrics per (text, mode, layer, head) / per layer
 results/summary_*.csv, summary_tables.md   summary tables

LIMITATIONS
 - CPU only; GPU behaviour (including flush-to-zero modes of specific libraries) was not tested.
 - One model and one revision; three short texts (172–233 tokens), batch size 1.
 - fp16_upcastQK and fp64_true are control constructions, not stock configurations.
 - M is a diagnostic quantity; no definition of w_min is singled out as canonical.
