"""Analyse saved attention tensors: zeros/underflow, top-8 knife fractions (knife on A vs on W), M under explicit definitions,
fp16-vs-fp32 graph comparison. Writes results/per_head.csv, results/per_layer.csv, results/M_grid.csv."""
import numpy as np, csv, json, itertools, os
TINY16=2.0**-14; SUB16=2.0**-24
EPS={"2^-14":2.0**-14,"2^-24":2.0**-24,"2^-126":2.0**-126,"2^-149":2.0**-149}
K=8
TEXTS=["prose","code","arith"]; MODES=["fp64","fp32","fp16","fp16_puresoftmax","fp16_prescale","fp16_upcastQK","fp64_true"]
def load(t,m):
    f=f"raw/attn_{t}_{m}.npz"
    return np.load(f)["A"].astype(np.float64) if os.path.exists(f) else None
def components(T, edges):
    p=list(range(T))
    def f(x):
        while p[x]!=x: p[x]=p[p[x]]; x=p[x]
        return x
    for a,b in edges:
        ra,rb=f(a),f(b)
        if ra!=rb: p[ra]=rb
    return len({f(i) for i in range(T)})
def topk_graph(A):
    """A: TxT (float64, may contain NaN). Returns W, idx[T,K] of top-K neighbours by W (diag removed), their W values."""
    W=(A+A.T)/2.0; np.fill_diagonal(W,-np.inf)
    Wsel=np.where(np.isnan(W),-np.inf,W)   # NaN never selected (naive torch.topk would pick NaN first)
    idx=np.argsort(-Wsel,axis=1,kind="stable")[:,:K]
    vals=np.take_along_axis(W,idx,1)
    Aval=np.maximum(np.take_along_axis(A,idx,1), np.take_along_axis(A.T,idx,1))  # the single nonzero causal entry behind the edge
    return W,idx,vals,Aval
rows=[]
ref={}
for t in TEXTS:
    for m in MODES:
        X=load(t,m)
        if X is None: continue
        L,H,T,_=X.shape
        tril=np.tril(np.ones((T,T),bool))
        for l in range(L):
            for h in range(H):
                A=X[l,h]; v=A[tril]
                nan=np.isnan(v); vf=v[~nan]
                r={"text":t,"mode":m,"layer":l,"head":h,"T":T,
                   "A_nan_frac":nan.mean(),"A_inf_any":bool(np.isinf(v).any()),
                   "A_zero_frac":float((vf==0).mean()) if vf.size else np.nan,
                   "A_min_nonzero":float(vf[vf>0].min()) if (vf>0).any() else np.nan,
                   "A_frac_below_2^-14":float((vf<TINY16).mean()) if vf.size else np.nan,
                   "A_frac_subnormal16":float(((vf>0)&(vf<TINY16)).mean()) if vf.size else np.nan,
                   "A_frac_below_2^-25":float((vf<2**-25).mean()) if vf.size else np.nan}
                W,idx,vals,Aval=topk_graph(A)
                fin=np.isfinite(vals)
                r["top8_nonfinite_frac"]=1-fin.mean()
                vv=vals[fin]; aa=Aval[fin]
                r["top8_knife_W_frac"]=float((vv<TINY16).mean()) if vv.size else np.nan   # threshold applied to W=(A+A^T)/2
                r["top8_knife_A_frac"]=float((aa<TINY16).mean()) if aa.size else np.nan   # threshold applied to A (before symmetrisation)
                r["top8_zero_frac"]=float((vv==0).mean()) if vv.size else np.nan
                # nan->0 naive pipeline
                A0=np.nan_to_num(A,nan=0.0); _,_,v0,_=topk_graph(A0)
                r["naive_nan2zero_top8_zero_frac"]=float((v0==0).mean())
                r["naive_nan2zero_top8_knifeW_frac"]=float((v0<TINY16).mean())
                # w_min candidates (on W, over finite top-8 edges)
                r["wmin_top8_global"]=float(vv.min()) if vv.size else np.nan
                pn=np.where(fin,vals,np.inf).min(1); pn=pn[np.isfinite(pn)]
                r["wmin_top8_median_node"]=float(np.median(pn)) if pn.size else np.nan
                r["w_top8_median"]=float(np.median(vv)) if vv.size else np.nan
                r["wmin_A_nonzero"]=r["A_min_nonzero"]
                # graph structure
                edges=[(i,int(j)) for i in range(T) for k,j in enumerate(idx[i]) if np.isfinite(vals[i,k])]
                r["n_comp_top8"]=components(T,edges)
                ek=[(i,int(j)) for i in range(T) for k,j in enumerate(idx[i]) if np.isfinite(vals[i,k]) and vals[i,k]>=TINY16]
                r["n_comp_top8_knifeW"]=components(T,ek)
                r["isolated_after_knifeW"]=int(T-len({x for e in ek for x in e}))
                S=set(map(tuple,np.sort(np.array(edges),1).tolist())) if edges else set()
                if m=="fp32": ref[(t,l,h)]=(S,idx,A)
                rows.append((r,S,idx))
# compare to fp32
out=[]
for r,S,idx in rows:
    key=(r["text"],r["layer"],r["head"])
    if key in ref:
        S32,idx32,A32=ref[key]
        r["jaccard_vs_fp32"]=len(S&S32)/len(S|S32) if (S|S32) else np.nan
        r["top8_set_overlap_vs_fp32"]=float(np.mean([len(set(a)&set(b))/K for a,b in zip(idx,idx32)]))
        # what fp16 actually did to the fp32 top-8 edges: fraction that became exact 0 in this mode
        if r["mode"]!="fp32":
            X=load(r["text"],r["mode"])[r["layer"],r["head"]]
            T=X.shape[0]; ii=np.repeat(np.arange(T),K); jj=idx32.ravel()
            a=np.maximum(X[ii,jj],X[jj,ii]); a32=np.maximum(A32[ii,jj],A32[jj,ii])
            r["fp32top8_became_zero_frac"]=float(np.mean(a==0)); r["fp32top8_became_nan_frac"]=float(np.mean(np.isnan(a)))
            fz=(a32<TINY16)
            r["fp32top8_below_2^-14_in_fp32"]=float(fz.mean())
            with np.errstate(divide="ignore",invalid="ignore"):
                rel=np.abs(a-a32)/a32
            r["fp32top8_relerr_median"]=float(np.nanmedian(rel))
    out.append(r)
keys=list(dict.fromkeys(k for r in out for k in r))
with open("results/per_head.csv","w",newline="") as f:
    w=csv.DictWriter(f,keys); w.writeheader(); [w.writerow(r) for r in out]
# per-layer summary: median over heads + worst head
num=[k for k in keys if k not in("text","mode","layer","head","T","A_inf_any")]
summ=[]
for (t,m,l),grp in itertools.groupby(sorted(out,key=lambda r:(r["text"],r["mode"],r["layer"])),key=lambda r:(r["text"],r["mode"],r["layer"])):
    g=list(grp); s={"text":t,"mode":m,"layer":l}
    for k in num:
        a=np.array([x.get(k,np.nan) for x in g],float)
        s[k+"_med"]=float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
        worst=np.nanmin if k.startswith(("wmin","w_top8","jaccard","top8_set")) else np.nanmax
        s[k+"_worst"]=float(worst(a)) if np.isfinite(a).any() else np.nan
    # layer-level wmin (over all heads) for M
    s["wmin_top8_global_layer"]=float(np.nanmin([x["wmin_top8_global"] for x in g])) if any(np.isfinite(x["wmin_top8_global"]) for x in g) else np.nan
    s["wmin_A_nonzero_layer"]=float(np.nanmin([x["wmin_A_nonzero"] for x in g])) if any(np.isfinite(x["wmin_A_nonzero"]) for x in g) else np.nan
    summ.append(s)
k2=list(summ[0].keys())
with open("results/per_layer.csv","w",newline="") as f:
    w=csv.DictWriter(f,k2); w.writeheader(); [w.writerow(r) for r in summ]
# M grid
WDEF={"top8_global_min(layer)":"wmin_top8_global_layer","top8_min_median_node(med heads)":"wmin_top8_median_node_med",
      "top8_min_median_node(worst head)":"wmin_top8_median_node_worst","top8_median(med heads)":"w_top8_median_med",
      "A_min_nonzero(layer)":"wmin_A_nonzero_layer"}
grid=[]
for s in summ:
    for wd,col in WDEF.items():
        for en,e in EPS.items():
            wv=s[col]
            grid.append({"text":s["text"],"mode":s["mode"],"layer":s["layer"],"w_def":wd,"eps":en,"w":wv,
                         "M":float(np.log10(wv/e)) if (wv==wv and wv>0) else np.nan})
with open("results/M_grid.csv","w",newline="") as f:
    w=csv.DictWriter(f,list(grid[0].keys())); w.writeheader(); [w.writerow(r) for r in grid]
print("rows",len(out),"layers",len(summ))
