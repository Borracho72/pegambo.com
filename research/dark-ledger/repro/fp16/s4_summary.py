"""Compact summary tables (markdown) from results/*.csv + extra fp32-vs-fp64_true underflow check."""
import pandas as pd, numpy as np, json
d=pd.read_csv("results/per_layer.csv"); h=pd.read_csv("results/per_head.csv")
out=[]
def md(df): return df.to_markdown(index=False)
# 0. M ranges per (mode, w_min definition, eps, text): all layers and deep layers (8-11)
g=pd.read_csv("results/M_grid.csv"); res=[]
for (m,wd,e,t),y in g.groupby(["mode","w_def","eps","text"]):
    a=y.M.values; dp=y[y.layer>=8].M.values
    f=lambda v,fn: float(fn(v)) if np.isfinite(v).any() else np.nan
    res.append(dict(mode=m,w_def=wd,eps=e,text=t,M_min=f(a,np.nanmin),M_max=f(a,np.nanmax),deep_min=f(dp,np.nanmin),deep_max=f(dp,np.nanmax),n_nan=int(np.isnan(a).sum())))
pd.DataFrame(res).to_csv("results/M_ranges.csv",index=False)
# 1. per-layer table, all texts
cols={"fp32":["top8_knife_W_frac_med","top8_knife_W_frac_worst","top8_knife_A_frac_med","top8_zero_frac_med","A_zero_frac_med"],
      "fp64_true":["top8_knife_W_frac_med","top8_zero_frac_med","A_zero_frac_med"],
      "fp16_upcastQK":["fp32top8_became_zero_frac_med","fp32top8_became_zero_frac_worst","A_zero_frac_med","jaccard_vs_fp32_med"],
      "fp16":["A_nan_frac_med","jaccard_vs_fp32_med","naive_nan2zero_top8_zero_frac_med"]}
tabs={}
for t in ["prose","code","arith"]:
    base=d[(d.text==t)][["layer"]].drop_duplicates().sort_values("layer").reset_index(drop=True)
    for m,cs in cols.items():
        x=d[(d.text==t)&(d["mode"]==m)].sort_values("layer").reset_index(drop=True)
        for c in cs: base[f"{m}:{c}"]=x[c].round(3)
    tabs[t]=base; base.to_csv(f"results/summary_layers_{t}.csv",index=False)
# 2. deep-layer ranges (median over heads) of knife fractions
rng=[]
for m in ["fp64_true","fp32","fp16_upcastQK"]:
    for c in ["top8_knife_W_frac_med","top8_knife_A_frac_med","top8_knife_W_frac_worst","top8_knife_A_frac_worst","top8_zero_frac_med","fp32top8_became_zero_frac_med"]:
        for lo in [6,8]:
            x=d[(d["mode"]==m)&(d.layer>=lo)][c]
            if x.notna().any(): rng.append({"mode":m,"metric":c,"layers":f"{lo}-11","min":round(x.min(),3),"max":round(x.max(),3)})
rng=pd.DataFrame(rng); rng.to_csv("results/summary_deep_ranges.csv",index=False)
# 3. first NaN layer per text in eager fp16
log=json.load(open("results/s2_run_log.json"))["runs"]
nan=[{"run":k,"first_nonfinite_module":(v["first_nonfinite"] or {}).get("module"),"logits_finite":v["logits_finite"],
      "hidden_states_finite(emb,L0..L11)":"".join("1" if b else "0" for b in v.get("hidden_states_finite_by_layer",[]))} for k,v in log.items()]
pd.DataFrame(nan).to_csv("results/summary_nonfinite.csv",index=False)
with open("results/summary_tables.md","w") as f:
    for t,b in tabs.items(): f.write(f"\n### {t}\n\n"+md(b)+"\n")
    f.write("\n### deep ranges\n\n"+md(rng)+"\n\n### nonfinite\n\n"+md(pd.DataFrame(nan))+"\n")
print(open("results/summary_tables.md").read())
