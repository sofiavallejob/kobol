# Evaluation framework: render methods offline, compare with raw WAVs on held-out settings
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.signal import stft
exec(open('f0.py').read())
z=np.load('harm.npz',allow_pickle=True); C,Mk,meta=z['C'],z['M'],z['meta'].astype(float)
W,Nn,F0=meta[:,0].astype(int),meta[:,1].astype(int),meta[:,2]; K=256
CVF=2.64+(Nn-21)*0.33; CVW=W*0.5
np.random.seed(0); perm=np.random.permutation(len(W)); VA,TR=perm[:66],perm[66:]
key={(W[i],Nn[i]):i for i in range(len(W))}
def raw(i,start=0,n=SR): return trim(D[(W[i],Nn[i])])[start:start+n]
def render(f,c,n=SR):
    t=np.arange(n)/SR; y=np.zeros(n); k=np.arange(1,K+1); v=k*f<18000
    for kk,cc in zip(k[v],c[v]): y+=np.abs(cc)*np.cos(2*np.pi*kk*f*t+np.angle(cc))
    return y
def spec_db(y,nper):
    w=np.hanning(nper); f,_,X=stft(y,SR,window=w,nperseg=nper,noverlap=nper*3//4,boundary=None,padded=False)
    X=np.abs(X)*2/w.sum()*nper/nper; m=f<18000
    return X[m]
def mrstft(a,b):
    # mean over 3 resolutions of: log-magnitude distance in dB (floor -100 dBFS) 
    out=[]
    for n in (512,2048,8192):
        A,B=spec_db(a,n),spec_db(b,n); t=min(A.shape[1],B.shape[1]); A,B=A[:,:t],B[:,:t]
        out.append(np.mean(np.abs(20*np.log10(np.maximum(A,1e-5))-20*np.log10(np.maximum(B,1e-5)))))
    return np.mean(out)
def specconv(a,b):
    out=[]
    for n in (512,2048,8192):
        A,B=spec_db(a,n),spec_db(b,n); t=min(A.shape[1],B.shape[1]); A,B=A[:,:t],B[:,:t]
        out.append(20*np.log10(np.linalg.norm(A-B)/np.linalg.norm(B)))
    return np.mean(out)
def hsnr(c,ref,m): return 10*np.log10((np.abs(ref*m)**2).sum()/(np.abs((c-ref)*m)**2).sum())
def lsd(c,ref,m): return np.sqrt(np.mean((20*np.log10((np.abs(c[m])+1e-4)/(np.abs(ref[m])+1e-4)))**2))
def register(n): return 'low' if n<=27 else ('mid' if n<=35 else 'high')
def family(w): return 'tri' if w<=4 else 'saw' if w<=10 else 'square' if w<=13 else 'pulse'
def evaluate(methods, idx=VA, name='eval'):
    rows=[]
    for j in idx:
        ref=raw(j); m=Mk[j]
        for mn,fn in methods.items():
            f,c=fn(j); y=render(f,c)
            rows.append(dict(method=mn,w=W[j],n=Nn[j],reg=register(Nn[j]),fam=family(W[j]),
                cents=1200*np.log2(f/F0[j]),snr=hsnr(c,C[j],m),lsd=lsd(c,C[j],m),mrstft=mrstft(y,ref),sc=specconv(y,ref)))
        # references: raw vs raw (second second of the same file is not available; use 0.5 s halves), raw vs neighbour raw
        a=trim(D[(W[j],Nn[j])]); h=len(a)//2; rows.append(dict(method='ref: same file halves',w=W[j],n=Nn[j],reg=register(Nn[j]),fam=family(W[j]),mrstft=mrstft(a[:h],a[h:2*h]),sc=specconv(a[:h],a[h:2*h])))
        for dw in (-1,1):
            i=key.get((W[j]+dw,Nn[j]))
            if i is not None:
                rows.append(dict(method='ref: neighbour recording',w=W[j],n=Nn[j],reg=register(Nn[j]),fam=family(W[j]),mrstft=mrstft(raw(i),ref),sc=specconv(raw(i),ref)));break
    d=pd.DataFrame(rows); d.to_csv(f'{name}.csv',index=False); return d
def summary(d):
    g=d.groupby('method')[['cents','snr','lsd','mrstft','sc']].median().round(2)
    print(g.to_string()); print(d.pivot_table(index='method',columns='reg',values='sc',aggfunc='median').round(2).to_string())
    print(d.pivot_table(index='method',columns='fam',values='mrstft',aggfunc='median').round(2).to_string())
# ---- methods from the existing v2 (val-trained nets) ----
def mlp(i,h,o,d):
    L=[nn.Linear(i,h),nn.SiLU()]
    for _ in range(d-1): L+=[nn.Linear(h,h),nn.SiLU()]
    return nn.Sequential(*L,nn.Linear(h,o))
class PU(nn.Module):
    def __init__(s): super().__init__(); s.lin=nn.Linear(2,1); s.res=mlp(2,64,1,3)
    def forward(s,x): return s.lin(x)+s.res(x)
def Xn(i): return torch.tensor([[(CVF[i]-6.27)/3.63,(CVW[i]-4.5)/4.5]],dtype=torch.float32)
def load_v2():
    sd=torch.load('model_val.pt'); P=PU(); P.load_state_dict(sd['P']); S=mlp(2,128,512,3); S.load_state_dict({k[4:]:v for k,v in sd['S'].items()})
    with torch.no_grad():
        X=torch.tensor(np.stack([(CVF-6.27)/3.63,(CVW-4.5)/4.5],1),dtype=torch.float32)
        lf=P(X)[:,0].numpy(); o=S(X).numpy()
    return 2**lf, (o[:,:K]+1j*o[:,K:])
def neighbours(j):
    return [key[(W[j]+dw,Nn[j]+dn)] for dw,dn in [(-1,0),(1,0),(0,-1),(0,1)] if (W[j]+dw,Nn[j]+dn) in key and key[(W[j]+dw,Nn[j]+dn)] in set(TR)]
TRS=set(TR)
def make_v2_methods():
    fs,cs=load_v2(); Rc={i:(C[i]*Mk[i]-cs[i]*Mk[i]) for i in TR}; Rp={i:np.log2(F0[i])-np.log2(fs[i]) for i in TR}
    def nb_avg(j):
        nb=[i for i in neighbours(j) if Nn[i]==Nn[j]] or neighbours(j)
        return F0[j]*0+2**np.mean([np.log2(F0[i])+ (Nn[j]-Nn[i])*0.4958 for i in nb]), np.mean([C[i]*Mk[i] for i in nb],0)
    def smooth(j): return fs[j],cs[j]
    def smooth_res(j):
        nb=neighbours(j); return 2**(np.log2(fs[j])+np.mean([Rp[i] for i in nb])), cs[j]+np.mean([Rc[i] for i in nb],0)
    return {'neighbour average':nb_avg,'v2 smooth net':smooth,'v2 net + residual':smooth_res}
if __name__=='__main__':
    d=evaluate(make_v2_methods(),name='eval_v2'); summary(d)
