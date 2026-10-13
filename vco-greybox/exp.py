# Experiment harness: cross-validation schemes x model variants. Saves predictions for all 437 settings per run.
import numpy as np, torch, torch.nn as nn, os, sys, time, json
torch.set_num_threads(1)
exec(open('evalfw.py').read().split("if __name__")[0])
k=torch.arange(1,K+1,dtype=torch.float32)
Nt=8192; ph=np.arange(Nt)/Nt; TRI0=torch.tensor((2*np.fft.fft(1-2*np.abs(2*ph-1))/Nt)[1:K+1].astype(np.complex64))
k64=k.double()
E=lambda phi: torch.exp(-2j*np.pi*k*phi)
E64=lambda phi: torch.exp(-2j*np.pi*k64*phi)
def pwl(bps,dsl):   # periodic piecewise-linear: breakpoints [B,J], slope changes [B,J] -> c_k [B,K]
    return (2*(dsl[:,:,None]*E64(bps[:,:,None])).sum(1)/((2j*np.pi*k64)**2)).to(torch.complex64)
def folds(scheme):
    if scheme=='rand5':
        p=np.random.RandomState(0).permutation(len(W)); return [np.sort(p[i::5]) for i in range(5)]
    if scheme=='col':   # interior waveform columns, interleaved
        return [np.where((W>=1)&(W<=17)&(W%4==r))[0] for r in range(4)]
    if scheme=='row':   # interior CV rows, interleaved
        return [np.where((Nn>=22)&(Nn<=42)&(Nn%4==r))[0] for r in range(4)]
class GB(nn.Module):
    def __init__(s,fl):
        super().__init__(); s.fl=fl; s.net=mlp(2,64,8,3)
        s.lfh=nn.Parameter(torch.tensor(np.log(5.0))); s.lTr=nn.Parameter(torch.tensor(np.log(2e-6)))
        s.lsl=nn.Parameter(torch.tensor([np.log(2e-6),np.log(2e-6)],dtype=torch.float32)); s.lfc2=nn.Parameter(torch.tensor(np.log(20000.)))
        with torch.no_grad(): s.net[-1].bias[:]=torch.tensor([0.1,0.1,0.1,0.0,0.1,1.0,0.0,np.log(15000.)]); s.net[-1].weight.mul_(0.1)
    def forward(s,x,f):
        o=s.net(x).float(); fl=s.fl; f=f.float()[:,None]; B=len(f)
        At,As,A1,t1,A2,t2,pt,lfc=[o[:,i:i+1] for i in range(8)]; th1,th2=torch.tanh(t1),torch.tanh(t2)
        if fl.get('reset'):
            r=torch.clamp(f.double()*torch.exp(s.lTr.double()),1e-4,0.3); s1=2/(1-r); s2=-2/r; z=torch.zeros_like(r)
            SAW=pwl(torch.cat([z,1-r],1),torch.cat([s1-s2,s2-s1],1))
            pa=(1-r)/2; pb=1-r/2
            TRI=pwl(torch.cat([z,pa,1-r,pb],1),torch.cat([2*s1-2*s2,-4*s1,-2*s2+2*s1,4*s2],1))
            rf=r.float(); cross=lambda th:((1-rf)*(1+th)/2, (1-rf)+rf*(1-th)/2)
        else:
            SAW=(2j/(np.pi*k)).expand(B,K); TRI=TRI0.expand(B,K)
            cross=lambda th:((1+th)/2, torch.ones_like(th))
        def comp(th):
            p1,p2=cross(th); a,b=E(p1),E(p2)
            if fl.get('slew'):
                e=f*torch.exp(s.lsl); a=a*torch.sinc(k*e[:,0:1]); b=b*torch.sinc(k*e[:,1:2])
            return 2*(a-b)/(1j*np.pi*k)
        c=As*SAW+A1*comp(th1)
        if fl.get('tri',True): c=c+At*TRI*E(0.1*torch.tanh(pt))
        if fl.get('compB',True): c=c+A2*comp(th2)
        fk=k*f
        if fl.get('lp',True): c=c/(1+1j*fk/torch.exp(lfc))
        if fl.get('lp2'): c=c/(1+1j*fk/torch.exp(s.lfc2))
        if fl.get('hp',True): fh=torch.exp(s.lfh); c=c*(1j*fk/fh)/(1+1j*fk/fh)
        rot=torch.angle(c[:,0])+np.pi/2; return c*torch.exp(-1j*k*rot[:,None])
Xall=torch.tensor(np.stack([(CVF-6.27)/3.63,(CVW-4.5)/4.5],1),dtype=torch.float32)
Ct=torch.tensor(C.astype(np.complex64)); Mt=torch.tensor(Mk.astype(np.float32)); Ft=torch.tensor(F0.astype(np.float32))
def lossf(c,i):
    m=Mt[i]; mse=((torch.abs(c-Ct[i])**2)*m).sum(1).mean()
    lm=((torch.log(torch.abs(c)+1e-4)-torch.log(torch.abs(Ct[i])+1e-4)).abs()*m).sum(1).mean()/K
    return mse+0.002*lm
class BB(nn.Module):  # black-box baseline (v2 shape unit)
    def __init__(s): super().__init__(); s.net=mlp(2,128,2*K,3)
    def forward(s,x,f): o=s.net(x).float(); return torch.complex(o[:,:K],o[:,K:])
def run(variant,tr,steps,seed=0):
    torch.manual_seed(seed)
    if variant=='bb': m=BB(); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=0.2)
    else: m=GB(VARIANTS[variant]); opt=torch.optim.Adam(m.parameters(),3e-3)
    tr=torch.tensor(tr)
    for it in range(steps):
        opt.zero_grad(); L=lossf(m(Xall[tr],Ft[tr]),tr)
        if variant!='bb' and m.fl.get('l1'): o=m.net(Xall[tr]); L=L+m.fl['l1']*(o[:,[0,1,2,4]].abs().mean())
        L.backward(); opt.step()
        if it in (steps//2,3*steps//4): opt.param_groups[0]['lr']*=0.3
    with torch.no_grad(): c=m(Xall,Ft).numpy()
    glob={}
    if variant!='bb':
        glob=dict(fh=float(torch.exp(m.lfh)),Tr_us=float(torch.exp(m.lTr))*1e6,slew_us=(torch.exp(m.lsl)*1e6).tolist(),fc2=float(torch.exp(m.lfc2)))
    return c,glob,m
VARIANTS={'full':{},'-tri':{'tri':False},'-compB':{'compB':False},'-lowpass':{'lp':False},'-chainHP':{'hp':False},
          '+slew':{'slew':True},'+reset':{'reset':True},'+lp2':{'lp2':True},'+all':{'slew':True,'reset':True,'lp2':True},'l1':{'l1':2e-3},'final':{'l1':2e-3,'lp2':True,'slew':True}}
if __name__=='__main__':
    plan=json.loads(sys.argv[1]); os.makedirs('runs',exist_ok=True); os.makedirs('seeds',exist_ok=True)
    for item in plan:
        scheme,variant,steps=item[:3]; seed=item[3] if len(item)>3 else 0
        for fi,va in enumerate(folds(scheme)):
            out=f'runs/{scheme}_{variant}_{fi}.npz' if seed==0 else f'seeds/{scheme}_{variant}_s{seed}_{fi}.npz'
            if os.path.exists(out): continue
            tr=np.setdiff1d(np.arange(len(W)),va); t0=time.time()
            c,glob,_=run(variant,tr,steps,seed)
            np.savez(out,c=c,va=va,glob=json.dumps(glob))
            print(scheme,variant,fi,'%.0fs'%(time.time()-t0),'heldout SNR med %.2f'%np.median([hsnr(c[j],C[j],Mk[j]) for j in va]),glob,flush=True)
