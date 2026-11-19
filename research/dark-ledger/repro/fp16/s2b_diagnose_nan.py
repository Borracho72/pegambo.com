"""Diagnose the origin of the first NaN in fp16 (CPU, model.half()): inspect inputs/weights of the first offending module."""
import torch, json, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0); torch.set_num_threads(2)
MODEL="EleutherAI/pythia-160m"; tok=AutoTokenizer.from_pretrained(MODEL)
m32=AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32).eval()
m16=AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32).half().eval()
res={}
for name in ["prose","code","arith"]:
    ids=tok(open(f"texts/{name}.txt").read(), return_tensors="pt")["input_ids"]
    cap={}
    def mk(tag):
        def h(m,i,o): cap[tag]=(i[0].detach().clone(), (o[0] if isinstance(o,tuple) else o).detach().clone())
        return h
    hs=[]
    for l in range(12):
        hs.append(m16.gpt_neox.layers[l].attention.dense.register_forward_hook(mk(f"16_dense{l}")))
        hs.append(m32.gpt_neox.layers[l].attention.dense.register_forward_hook(mk(f"32_dense{l}")))
        hs.append(m16.gpt_neox.layers[l].attention.query_key_value.register_forward_hook(mk(f"16_qkv{l}")))
        hs.append(m32.gpt_neox.layers[l].attention.query_key_value.register_forward_hook(mk(f"32_qkv{l}")))
    with torch.no_grad(): m32(ids); m16(ids)
    for h in hs: h.remove()
    r={}
    for l in range(12):
        x16,y16=cap[f"16_dense{l}"]; x32,y32=cap[f"32_dense{l}"]
        q16,_=cap[f"16_qkv{l}"][1],0
        W=m16.gpt_neox.layers[l].attention.dense.weight
        # exact fp32 product of the fp16 inputs (what an fp32 accumulator would give)
        y_ref=(x16.float()@W.float().T)+m16.gpt_neox.layers[l].attention.dense.bias.float()
        r[l]={"dense_in_finite":bool(torch.isfinite(x16).all()),"dense_in_maxabs_fp16":float(x16.float().abs().max()),
              "dense_in_maxabs_fp32run":float(x32.abs().max()),
              "dense_out_nan_fp16":int(torch.isnan(y16).sum()),"dense_out_inf_fp16":int(torch.isinf(y16).sum()),
              "dense_out_maxabs_fp32run":float(y32.abs().max()),
              "dense_out_maxabs_ref_fp32accum_of_fp16_inputs":float(y_ref.abs().nan_to_num(0).max()),
              "qkv_out_maxabs_fp16":float(cap[f"16_qkv{l}"][1].float().abs().nan_to_num(0).max()),
              "qkv_out_maxabs_fp32":float(cap[f"32_qkv{l}"][1].abs().max()),
              "nan_token_positions":sorted(set(torch.nonzero(torch.isnan(y16[0]).any(-1)).flatten().tolist()))[:30]}
        if l in (4,5) and r[l]["dense_out_nan_fp16"]>0 and r[l]["dense_in_finite"]:
            # where is the NaN born? test partial sums for one offending output element
            t,c=[int(v) for v in torch.nonzero(torch.isnan(y16[0]))[0]]
            prods=(x16[0,t]*W[c])        # fp16 elementwise products
            r[l]["probe"]={"token":t,"channel":c,"prod_has_inf":bool(torch.isinf(prods).any()),
               "max_abs_prod_fp32":float((x16[0,t].float()*W[c].float()).abs().max()),
               "max_abs_x":float(x16[0,t].float().abs().max()),"max_abs_w_row":float(W[c].float().abs().max()),
               "ref_value_fp32":float(y_ref[0,t,c]),
               "matmul_fp16_single_row":float((x16[0,t:t+1]@W.T)[0,c]),
               "matmul_fp16_via_linear_single":float(torch.nn.functional.linear(x16[0,t:t+1],W)[0,c])}
    res[name]=r
    print(name, {l:(v["dense_out_nan_fp16"],round(v["dense_in_maxabs_fp16"],1),round(v["dense_out_maxabs_ref_fp32accum_of_fp16_inputs"],1)) for l,v in r.items()})
    for l in (4,5):
        if "probe" in r[l]: print(" probe L",l,r[l]["probe"])
json.dump(res,open("results/s2b_nan_diagnosis.json","w"),indent=1)
