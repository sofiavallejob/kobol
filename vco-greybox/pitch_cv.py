import numpy as np, pandas as pd, torch, h5py, json
from scipy.optimize import least_squares
torch.set_num_threads(1)
exec(open('exp.py').read().split("\nif __name__=='__main__'")[0])
NI=Nn-21
def model_logf(p,i,det=True,off=True,dl_override=None):
    a,b,c,d,lTr=p[:5]; dl=p[5:] if dl_override is None else dl_override
    x=a+b*(CVF[i]-6.27)+(c*(CVW[i]-4.5)+d*(CVW[i]-4.5)*(CVF[i]-6.27) if det else 0)+(dl[NI[i]] if off else 0)
    fe=2**x; return np.log2(1/(1/fe+np.exp(lTr)))
def fit(tr,det=True,off=True,reset=True):
    def res(p):
        if not reset: p=p.copy(); p[4]=np.log(1e-12)
        r=1200*(model_logf(p,tr,det,off)-np.log2(F0[tr])); dl=p[5:][2:21]; kk=np.arange(19)-9
        return np.r_[r,300*dl.sum(),300*(dl*kk).sum()]
    p=least_squares(res,np.r_[np.log2(400),1.5,0,0,np.log(1e-6),np.zeros(23)]).x
    if not reset: p[4]=np.log(1e-12)
    # CV rows never seen in training: interpolate their offsets from neighbouring rows
    seen=np.unique(NI[tr]); dl=p[5:].copy(); dl=np.interp(np.arange(23),seen,dl[seen]); p[5:]=dl
    return p
# thesis periodic unit (trained on all data in the thesis: not a held-out estimate)
f=h5py.File('KobolRSF/models/PU.h5'); g=f['model_weights']
L=[(g[f'dense_{i}/dense_{i}/kernel:0'][:],g[f'dense_{i}/dense_{i}/bias:0'][:]) for i in (4,5,6,7)]
def thesis_f(i):
    h=((CVF[i]-2.64)/7.26)[:,None]
    for j,(Wm,b) in enumerate(L):
        h=h@Wm+b
        if j<3: h=np.maximum(h,0)
    w=h[:,0]*(55812.369258-7.051323)+7.051323; return w/(2*np.pi)
class PU(nn.Module):
    def __init__(s): super().__init__(); s.lin=nn.Linear(2,1); s.res=mlp(2,64,1,3)
    def forward(s,x): return s.lin(x)+s.res(x)
def nn_pitch(tr):
    torch.manual_seed(0); P=PU(); X=Xall; y=torch.tensor(np.log2(F0),dtype=torch.float32)
    with torch.no_grad():
        A=np.c_[X[tr].numpy(),np.ones(len(tr))]; sol=np.linalg.lstsq(A,np.log2(F0[tr]),rcond=None)[0]
        P.lin.weight[:]=torch.tensor(sol[:2][None],dtype=torch.float32); P.lin.bias[:]=float(sol[2])
    opt=torch.optim.Adam(P.parameters(),2e-3)
    for it in range(4000):
        opt.zero_grad(); l=((P(X[tr])[:,0]-y[tr])**2).mean(); l.backward(); opt.step()
        if it in (2000,3000): opt.param_groups[0]['lr']*=0.3
    with torch.no_grad(): return P(X)[:,0].numpy()
def grid_res(lf,tr,j):   # residual correction from training neighbours (bilinear on grid ~ average of available neighbours)
    trs=set(tr); nb=[key[(W[j]+dw,Nn[j]+dn)] for dw,dn in [(-1,0),(1,0),(0,-1),(0,1)] if (W[j]+dw,Nn[j]+dn) in key and key[(W[j]+dw,Nn[j]+dn)] in trs]
    return np.mean([np.log2(F0[i])-lf[i] for i in nb]) if nb else 0
rows=[]
for scheme in ['rand5','col','row']:
    for va in folds(scheme):
        tr=np.setdiff1d(np.arange(len(W)),va); allI=np.arange(len(W))
        preds={}
        p=fit(tr); preds['physical']=model_logf(p,allI)
        preds['physical, no waveform detune']=model_logf(fit(tr,det=False),allI,det=False)
        preds['physical, no CV offsets']=model_logf(fit(tr,off=False),allI,off=False)
        preds['physical, no reset term']=model_logf(fit(tr,reset=False),allI)
        preds['neural pitch unit (v2)']=nn_pitch(tr)
        preds['thesis periodic unit*']=np.log2(thesis_f(allI))
        for mname,lf in list(preds.items()):
            for j in va:
                rows.append(dict(scheme=scheme,method=mname,j=j,cents=1200*(lf[j]-np.log2(F0[j]))))
                if mname in ('physical','neural pitch unit (v2)'):
                    rows.append(dict(scheme=scheme,method=mname+' + residual',j=j,cents=1200*(lf[j]+grid_res(lf,tr,j)-np.log2(F0[j]))))
    print(scheme,'done',flush=True)
d=pd.DataFrame(rows); d.to_csv('pitch_cv.csv',index=False)
# exclude the clamped extremes (21,22,42,43) in a second view
d['interior']=~np.isin(Nn[d.j],[21,22,42,43])
t=d.groupby(['method','scheme']).cents.apply(lambda c:np.sqrt(np.mean(c**2))).unstack().round(2)
t2=d[d.interior].groupby(['method','scheme']).cents.apply(lambda c:np.sqrt(np.mean(c**2))).unstack().round(2)
print('RMS cents, all held-out');print(t.to_string());print('RMS cents, interior range');print(t2.to_string())
