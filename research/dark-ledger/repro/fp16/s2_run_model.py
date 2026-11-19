"""Run Pythia-160M on 3 probe texts in fp64/fp32/fp16 (model.half() on CPU, eager attention).
Saves raw attention tensors A[layer,head,T,T] and non-finite hook log."""
import torch, json, numpy as np, sys, platform, transformers
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.gpt_neox.modeling_gpt_neox as gm
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)
MODEL="EleutherAI/pythia-160m"; REV="main"
tok=AutoTokenizer.from_pretrained(MODEL)
orig_eager=gm.eager_attention_forward
def eager_pure_fp16(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    w=torch.matmul(query, key.transpose(2,3))*scaling
    if attention_mask is not None: w=w+attention_mask
    w=F.softmax(w, dim=-1)            # softmax in the tensor dtype (fp16) instead of fp32-upcast
    out=torch.matmul(w, value).transpose(1,2).contiguous()
    return out, w
def eager_prescale(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    # mathematically identical to HF eager, but scales q BEFORE the fp16 matmul (avoids overflow of unscaled QK^T)
    w=torch.matmul(query*scaling, key.transpose(2,3))
    if attention_mask is not None: w=w+attention_mask
    w=F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    out=torch.matmul(w, value).transpose(1,2).contiguous()
    return out, w
def eager_upcast_qk(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    # fp16 model, but QK^T, mask and softmax in fp32; A is then STORED in fp16 (isolates fp16 representation of A from overflow)
    w=torch.matmul(query.float(), key.float().transpose(2,3))*scaling
    if attention_mask is not None: w=w+attention_mask.float()
    w=F.softmax(w, dim=-1).to(query.dtype)
    out=torch.matmul(w, value).transpose(1,2).contiguous()
    return out, w
def eager_native_dtype(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    # softmax in the model dtype (used for a TRUE fp64 control: HF eager forces softmax in fp32 even for fp64 models)
    w=torch.matmul(query, key.transpose(2,3))*scaling
    if attention_mask is not None: w=w+attention_mask
    w=F.softmax(w, dim=-1, dtype=query.dtype)
    out=torch.matmul(w, value).transpose(1,2).contiguous()
    return out, w
PATCH={"native":eager_native_dtype,"upcastqk":eager_upcast_qk, False:orig_eager, True:eager_pure_fp16, "prescale":eager_prescale}
MODES={"fp64":(torch.float64,False),"fp32":(torch.float32,False),"fp16":(torch.float16,False),"fp16_puresoftmax":(torch.float16,True),
       "fp16_prescale":(torch.float16,"prescale"),"fp16_sdpa":(torch.float16,"sdpa"),"fp16_upcastQK":(torch.float16,"upcastqk"),"fp64_true":(torch.float64,"native")}
ONLY=sys.argv[1:]
log={"env":{"torch":torch.__version__,"transformers":transformers.__version__,"python":sys.version,"platform":platform.platform(),
     "cpu":open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),"threads":torch.get_num_threads(),
     "model":MODEL,"attn_implementation":"eager"},"runs":{}}
import os
if ONLY and os.path.exists("results/s2_run_log.json"): log["runs"]=json.load(open("results/s2_run_log.json"))["runs"]
for mode,(dt,pure) in MODES.items():
    if ONLY and mode not in ONLY: continue
    gm.eager_attention_forward = PATCH.get(pure, orig_eager)
    impl = "sdpa" if pure=="sdpa" else "eager"
    model=AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation=impl, dtype=torch.float32).to(dt).eval()
    for name in ["prose","code","arith"]:
        ids=tok(open(f"texts/{name}.txt").read(), return_tensors="pt")["input_ids"]
        events=[]; maxabs={}
        hooks=[]
        for mn,m in model.named_modules():
            if mn=="": continue
            def h(m,i,o,mn=mn):
                t=o[0] if isinstance(o,(tuple,list)) else o
                if torch.is_tensor(t) and t.is_floating_point():
                    fin=torch.isfinite(t)
                    if not bool(fin.all()):
                        events.append({"module":mn,"n_nonfinite":int((~fin).sum()),"n_nan":int(torch.isnan(t).sum()),"n_inf":int(torch.isinf(t).sum())})
                    maxabs[mn]=float(t.abs().max())
            hooks.append(m.register_forward_hook(h))
        with torch.no_grad():
            out=model(ids, output_attentions=(impl=="eager"), output_hidden_states=True)
        for hk in hooks: hk.remove()
        HS=[bool(torch.isfinite(h).all()) for h in out.hidden_states]
        if impl=="sdpa":
            A=torch.zeros(1)
        else:
            A=torch.stack([a[0] for a in out.attentions])  # L,H,T,T
        logits=out.logits[0].float()
        np.savez_compressed(f"raw/attn_{name}_{mode}.npz", A=A.numpy(), logits_last=logits[-1].numpy())
        log["runs"][f"{name}_{mode}"]={"hidden_states_finite_by_layer":HS,"T":int(ids.shape[1]),"dtype_A":str(A.dtype),"first_nonfinite":events[0] if events else None,
            "n_nonfinite_events":len(events),"nonfinite_events":events[:20],"logits_finite":bool(torch.isfinite(logits).all()),
            "attn_finite":bool(torch.isfinite(A.float()).all()),
            "max_abs_activation_overall":max([v for v in maxabs.values() if v==v] or [float("nan")]),
            "max_abs_by_layer":{f"layer{l}":max([v for k,v in maxabs.items() if k.startswith(f"gpt_neox.layers.{l}.") and v==v] or [float("nan")]) for l in range(model.config.num_hidden_layers)}}
        print(mode,name,ids.shape[1],"nonfinite events:",len(events),"first:",events[0]["module"] if events else None,"logits finite:",bool(torch.isfinite(logits).all()),"maxabs:",round(max([v for v in maxabs.values() if v==v] or [float("nan")]),1),flush=True)
    del model
gm.eager_attention_forward=orig_eager
json.dump(log,open("results/s2_run_log.json","w"),indent=1)
