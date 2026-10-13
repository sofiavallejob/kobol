import numpy as np, pandas as pd
from scipy.signal import welch
exec(open('f0.py').read())
e=pd.read_csv('f0b.csv')
def analyze(w,n,f0,full=False):
    y=trim(D[(w,n)]); y=y-y.mean(); N=len(y); t=np.arange(N)/SR
    P=SR/f0; L=max(2,int(round(2*P)))
    z=y*np.exp(-2j*np.pi*f0*t)
    k=np.ones(L)/L; zf=np.convolve(z,k,'valid')  # demod fundamental
    ph=np.unwrap(np.angle(zf)); 
    dec=max(1,int(P/4)); ph=ph[::dec]; tt=np.arange(len(ph))*dec/SR
    A=np.c_[tt,np.ones_like(tt)]; ph=ph-A@np.linalg.lstsq(A,ph,rcond=None)[0]   # remove f0 error
    # instantaneous freq deviation in cents
    df=np.gradient(ph,tt)/(2*np.pi); cents=1200*np.log2(1+df/f0)
    amp=np.abs(zf)[::dec]; ar=amp/amp.mean()-1
    # tracked phase -> periodic reconstruction -> residual
    phi=2*np.pi*f0*t
    # interpolate phase deviation back to full rate
    pd_full=np.interp(np.arange(N)/SR, tt+ (L/2)/SR, ph)
    phi=phi+pd_full
    Kmax=int(min(18000/f0, 1500)); y_hat=np.zeros(N)
    win=np.hanning(N)
    for k0 in range(1,Kmax+1,64):
        kk=np.arange(k0,min(k0+64,Kmax+1))[:,None]
        E=np.exp(-1j*kk*phi); c=2*(E*(y*win)).sum(1)/win.sum()
        y_hat+=np.real(c[:,None]*np.conj(E)).sum(0)
    r=(y-y_hat)[2000:-2000]
    f,Pr=welch(r,SR,nperseg=4096); f,Py=welch(y,SR,nperseg=4096)
    return dict(w=w,n=n,f0=f0,jit=np.std(cents),amp=np.std(ar),snr=10*np.log10(np.mean(y**2)/np.mean(r**2)),
                cents=cents,tt=tt,f=f,Pr=Pr,Py=Py)
if __name__=='__main__':
    for w,n in [(0,24),(0,30),(9,30),(17,30),(0,36),(9,40)]:
        f0=e[(e.w==w)&(e.n==n)].f0.item(); a=analyze(w,n,f0)
        print(w,n,round(f0,1),'jitter cents',round(a['jit'],2),'amp %',round(100*a['amp'],2),'periodic/residual dB',round(a['snr'],1))
