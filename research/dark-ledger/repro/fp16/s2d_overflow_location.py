"""fp32 run: where are the large QK^T logits — in causal (visible) or masked (future) positions? Compare with fp16 limit 65504 (unscaled and scaled)."""
import torch, json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.gpt_neox.modeling_gpt_neox as gm
torch.set_num_threads(2)
MODEL="EleutherAI/pythia-160m"; tok=AutoTokenizer.from_pretrained(MODEL)
orig=gm.eager_attention_forward; REC=[]
def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    s=torch.matmul(query, key.transpose(2,3))  # unscaled, fp32
    T=s.shape[-1]; tril=torch.tril(torch.ones(T,T,dtype=torch.bool))
    vis=s[..., tril]; msk=s[..., ~tril]
    REC.append({"layer":module.layer_idx,"scaling":scaling,
      "unscaled_max_abs_visible":float(vis.abs().max()),"unscaled_max_abs_masked":float(msk.abs().max()),
      "n_visible_unscaled_over_65504":int((vis.abs()>65504).sum()),"n_masked_unscaled_over_65504":int((msk.abs()>65504).sum()),
      "n_visible_scaled_over_65504":int((vis.abs()*scaling>65504).sum()),"n_masked_scaled_over_65504":int((msk.abs()*scaling>65504).sum()),
      "rows_with_visible_overflow_unscaled":int(((s.abs()>65504)&tril).any(-1).sum()),
      "max_visible_logit_scaled":float(vis.max()*scaling), "min_visible_logit_scaled":float(vis.min()*scaling)})
    return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
gm.eager_attention_forward=eager
m=AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32).eval()
out={}
for t in ["prose","code","arith"]:
    REC.clear(); ids=tok(open(f"texts/{t}.txt").read(), return_tensors="pt")["input_ids"]
    with torch.no_grad(): m(ids)
    out[t]=list(REC)
    for r in REC: print(t,r["layer"],round(r["unscaled_max_abs_visible"]),round(r["unscaled_max_abs_masked"]),r["n_visible_unscaled_over_65504"],r["n_masked_unscaled_over_65504"],r["n_visible_scaled_over_65504"],r["n_masked_scaled_over_65504"],round(r["min_visible_logit_scaled"]),round(r["max_visible_logit_scaled"]))
json.dump(out,open("results/s2d_overflow_location.json","w"),indent=1)
