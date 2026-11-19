import torch, json, platform
torch.manual_seed(0)
fi=torch.finfo(torch.float16); f32=torch.finfo(torch.float32)
out={"torch":torch.__version__,"platform":platform.platform(),"processor":platform.processor(),
 "fp16":{"tiny_normal":fi.tiny,"smallest_subnormal":2**-24,"eps":fi.eps,"max":fi.max},
 "fp32":{"tiny_normal":f32.tiny,"smallest_subnormal":2**-149,"eps":f32.eps},
 "denormals_flush_default_cpu": None}
# subnormal representability
x=torch.tensor([3e-5,1e-6,6e-8,5.96e-8,2.9e-8,2.98e-8,3.0e-8],dtype=torch.float32)
out["cast_fp32_to_fp16"]={str(float(a)):float(b) for a,b in zip(x,x.half())}
# arithmetic on subnormals
a=torch.tensor([2**-14],dtype=torch.float16)
out["2^-14/16 in fp16"]=float(a/16); out["2^-14/2^10"]=float(a/1024); out["2^-14/2^11"]=float(a/2048)
# softmax in fp16 on CPU: logits gap d -> p_min
res=[]
for d in [5,8,9,9.7,10,11,12,14,16,16.5,17,17.3,17.5,18,20]:
    for n in [2,128]:
        l=torch.zeros(n); l[1:]=-d
        p16=torch.softmax(l.half(),-1); p32=torch.softmax(l,-1)
        res.append({"gap":d,"n":n,"p_min_fp32":float(p32[1]),"p_min_fp16":float(p16[1]),
                    "fp16_zero":bool(p16[1]==0),"fp16_subnormal":bool(0<p16[1]<2**-14)})
out["softmax_gap_scan"]=res
# random softmax
l=torch.randn(64,256,256)*6
p16=torch.softmax(l.half(),-1); p32=torch.softmax(l,-1)
nz=p16[p16>0]
out["random_softmax"]={"min_nonzero_fp16":float(nz.min()),"frac_zero_fp16":float((p16==0).float().mean()),
 "frac_zero_fp32":float((p32==0).float().mean()),"frac_fp32_below_2^-14":float((p32<2**-14).float().mean()),
 "frac_fp16_subnormal":float(((p16>0)&(p16<2**-14)).float().mean()),
 "frac_fp32_below_2^-25":float((p32<2**-25).float().mean())}
# set_flush_denormal
out["set_flush_denormal_supported"]=torch.set_flush_denormal(True)
out["after_flush: cast 1e-6 fp16"]=float(torch.tensor([1e-6]).half()); out["after_flush: 2^-14/16"]=float(a/16)
out["after_flush softmax gap12 fp16"]=float(torch.softmax(torch.tensor([0.,-12.]).half(),-1)[1])
torch.set_flush_denormal(False)
json.dump(out,open("results/s1_format_facts.json","w"),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!="softmax_gap_scan"},indent=1))
for r in res: print(r)
