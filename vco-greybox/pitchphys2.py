import numpy as np, pandas as pd
from scipy.optimize import least_squares
exec(open('evalfw.py').read().split("if __name__")[0])
NI=Nn-21
def phys(p,cvf,cvw,ni):
    a,b,c,d,lTr=p[:5]; dl=p[5:]
    fe=2**(a+b*(cvf-6.27)+c*(cvw-4.5)+d*(cvw-4.5)*(cvf-6.27)+dl[ni])
    return 1/(1/fe+np.exp(lTr))
def res(p,idx):
    r=1200*np.log2(phys(p,CVF[idx],CVW[idx],NI[idx])/F0[idx]); dl=p[5:][2:21]; k=np.arange(19)-9
    return np.r_[r, 300*dl.sum(), 300*(dl*k).sum()]   # identifiability: offsets have no mean or linear trend
p0=np.r_[np.log2(400),1.5,0,0,np.log(1e-6),np.zeros(23)]
def fitp(idx): return least_squares(res,p0,args=(idx,)).x
p=fitp(TR)
cents=lambda p,idx:1200*np.log2(phys(p,CVF[idx],CVW[idx],NI[idx])/F0[idx])
ev=cents(p,VA); et=cents(p,TR)
print('oct/V %.4f  detune c=%.5f d=%.5f  Tr=%.2f us'%(p[1],p[2],p[3],np.exp(p[4])*1e6))
print('physical + CV-step offsets: train rms %.2f, held-out rms %.2f, median abs %.2f'%(np.sqrt(np.mean(et**2)),np.sqrt(np.mean(ev**2)),np.median(np.abs(ev))))
# without reset term
p_nr=least_squares(lambda q,idx:res(np.r_[q[:4],np.log(1e-12),q[4:]],idx),np.r_[p0[:4],p0[5:]],args=(TR,)).x
e_nr=cents(np.r_[p_nr[:4],np.log(1e-12),p_nr[4:]],VA); print('same without reset term: held-out rms %.2f'%np.sqrt(np.mean(e_nr**2)))
# without waveform terms
p_nw=least_squares(lambda q,idx:res(np.r_[q[:2],0,0,q[2:]],idx),np.r_[p0[:2],p0[4:]],args=(TR,)).x
e_nw=cents(np.r_[p_nw[:2],0,0,p_nw[2:]],VA); print('same without waveform detune terms: held-out rms %.2f'%np.sqrt(np.mean(e_nw**2)))
pf=least_squares(res,p,args=(np.arange(len(W)),)).x; np.save('pitch_phys_full.npy',pf); np.save('pitch_phys_val.npy',p)
print('CV step offsets (cents):',np.round(1200*pf[5:],1))
r=cents(pf,np.arange(len(W))); d=pd.DataFrame(dict(n=Nn,w=W,r=r))
print(d.pivot_table(index='n',columns='w',values='r').iloc[:, [0,5,9,13,18]].round(1).T.to_string())
