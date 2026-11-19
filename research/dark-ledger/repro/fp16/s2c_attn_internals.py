"""Instrument eager attention internals (q,k,v, QK^T scores, softmax, A@V) to find where the first non-finite value is born in fp16."""
import torch, json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.gpt_neox.modeling_gpt_neox as gm
torch.set_num_threads(2); torch.manual_seed(0)
MODEL="EleutherAI/pythia-160m"; tok=AutoTokenizer.from_pretrained(MODEL)
REC=[]
def stat(t):
    tf=t.float(); fin=torch.isfinite(tf)
    return {"maxabs_finite":float(tf[fin].abs().max()) if fin.any() else None,"n_nan":int(torch.isnan(tf).sum()),"n_inf":int(torch.isinf(tf).sum())}
def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    r={"layer":module.layer_idx,"dtype":str(query.dtype),"q":stat(query),"k":stat(key),"v":stat(value)}
    s=torch.matmul(query, key.transpose(2,3))*scaling
    r["scores"]=stat(s)
    s_ref=torch.matmul(query.float(), key.float().transpose(2,3))*scaling
    r["scores_fp32_from_same_qk"]=stat(s_ref)
    if attention_mask is not None:
        r["mask_min"]=float(attention_mask.float().min()); s=s+attention_mask
    r["scores_masked"]=stat(s)
    w=F.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
    r["A"]=stat(w)
    o=torch.matmul(w, value); r["AV"]=stat(o)
    o_ref=torch.matmul(w.float(), value.float()); r["AV_fp32accum"]=stat(o_ref)
    REC.append(r)
    return o.transpose(1,2).contiguous(), w
gm.eager_attention_forward=eager
out={}
for dt in [torch.float32, torch.float16]:
    m=AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32).to(dt).eval()
    for name in ["prose","code","arith"]:
        REC.clear()
        ids=tok(open(f"texts/{name}.txt").read(), return_tensors="pt")["input_ids"]
        with torch.no_grad(): m(ids)
        out[f"{name}_{str(dt).split('.')[-1]}"]=list(REC)
for k,v in out.items():
    print(k)
    for r in v[:7]:
        print(" L",r["layer"],"q",r["q"],"k",r["k"],"v",r["v"]["maxabs_finite"],"scores",r["scores"],"masked",r["scores_masked"],"A",r["A"]["n_nan"],"AV",r["AV"],"AVref",r["AV_fp32accum"]["n_nan"])
json.dump(out,open("results/s2c_attn_internals.json","w"),indent=1)
