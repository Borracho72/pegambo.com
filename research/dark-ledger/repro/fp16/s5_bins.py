"""Where do the 'true' (fp64, softmax in fp64) top-8 edge weights lie relative to fp16/fp32 thresholds? Per text x layer, pooled over heads/nodes.
Bin edges refer to A (the single causal entry behind each W-edge; knife on A)."""
import numpy as np, pandas as pd
K=8; rows=[]
B=[("A>=2^-14 (normal fp16)",2.0**-14,np.inf),("2^-24<=A<2^-14 (fp16 subnormal, kept)",2.0**-24,2.0**-14),
   ("2^-25<A<2^-24 (rounds up to 2^-24)",2.0**-25,2.0**-24),("2^-149<=A<=2^-25 (fp16 -> 0)",2.0**-149,2.0**-25),("A<2^-149 (0 even in fp32)",0,2.0**-149)]
for t in ["prose","code","arith"]:
    X=np.load(f"raw/attn_{t}_fp64_true.npz")["A"]
    for l in range(X.shape[0]):
        vals=[]
        for h in range(X.shape[1]):
            A=X[l,h]; W=(A+A.T)/2; np.fill_diagonal(W,-np.inf)
            idx=np.argsort(-W,1,kind="stable")[:,:K]
            vals.append(np.maximum(np.take_along_axis(A,idx,1),np.take_along_axis(A.T,idx,1)).ravel())
        v=np.concatenate(vals); r={"text":t,"layer":l}
        for name,lo,hi in B: r[name]=round(float(((v>=lo)&(v<hi)).mean()) if name!="2^-149<=A<=2^-25 (fp16 -> 0)" else float(((v>=lo)&(v<=hi)).mean()),4)
        r["min_top8_A"]=float(v.min()); rows.append(r)
df=pd.DataFrame(rows); df.to_csv("results/s5_top8_bins_fp64true.csv",index=False); print(df.to_markdown(index=False))
