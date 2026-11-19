"""Pooled (over heads and nodes) fraction of top-8 edges below 2^-14: knife on A vs knife on W, for fp64_true / fp32 / fp16_upcastQK; plus fraction actually ==0."""
import numpy as np, pandas as pd
K=8; T14=2.0**-14; rows=[]
for t in ["prose","code","arith"]:
    for m in ["fp64_true","fp32","fp16_upcastQK"]:
        X=np.load(f"raw/attn_{t}_{m}.npz")["A"].astype(np.float64)
        for l in range(X.shape[0]):
            wv=[];av=[]
            for h in range(X.shape[1]):
                A=X[l,h]; W=(A+A.T)/2; np.fill_diagonal(W,-np.inf)
                idx=np.argsort(-W,1,kind="stable")[:,:K]
                wv.append(np.take_along_axis(W,idx,1).ravel()); av.append(np.maximum(np.take_along_axis(A,idx,1),np.take_along_axis(A.T,idx,1)).ravel())
            wv=np.concatenate(wv); av=np.concatenate(av)
            rows.append({"text":t,"mode":m,"layer":l,"pooled_knife_W":round(float((wv<T14).mean()),4),"pooled_knife_A":round(float((av<T14).mean()),4),"pooled_exact_zero":round(float((wv==0).mean()),4)})
df=pd.DataFrame(rows); df.to_csv("results/s6_pooled_knife.csv",index=False)
p=df.pivot_table(index=["text","layer"],columns="mode",values=["pooled_knife_A","pooled_knife_W","pooled_exact_zero"]).round(3)
print(p[p.index.get_level_values(1)>=6].to_markdown())
