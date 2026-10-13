import numpy as np
exec(open('load.py').read())
SR=44100
def trim(x):
    x=x[:,0]; m=np.abs(x).max(); i=np.argmax(np.abs(x)>0.05*m)
    return x[i+1500:]
def f0_est(x):
    x=x-x.mean(); n=len(x)
    X=np.fft.rfft(x,2*n); ac=np.fft.irfft(np.abs(X)**2)[:n]; ac/=ac[0]
    # normalize by overlap
    ac=ac*n/(n-np.arange(n))
    z=np.argmax(ac<0) if (ac<0).any() else 2
    lo=max(z,2); hi=min(int(0.2*SR),n//2)
    seg=ac[lo:hi]; L=lo+np.argmax(seg)
    # pick smallest lag peak with >0.9 of max (avoid octave errors)
    mx=ac[L]
    peaks=[i for i in range(lo+1,hi-1) if ac[i]>ac[i-1] and ac[i]>=ac[i+1] and ac[i]>0.85*mx]
    if peaks: L=peaks[0]
    # refine with multiple periods
    k=max(1,int((n//2)/L)-1); Lk=int(round(L*k))
    w=max(2,int(0.02*L)+2)
    j=Lk-w+np.argmax(ac[Lk-w:Lk+w+1]); 
    a,b,c=ac[j-1],ac[j],ac[j+1]; d=0.5*(a-c)/(a-2*b+c) if (a-2*b+c)!=0 else 0
    return SR*k/(j+d), mx
def fft_refine(y, coarse, tol=0.06):
    y=y-y.mean(); N=len(y); Y=np.abs(np.fft.rfft(y*np.hanning(N),N*8)); fr=np.fft.rfftfreq(N*8,1/SR)
    m=(fr>coarse*(1-tol))&(fr<coarse*(1+tol)); i=np.where(m)[0][np.argmax(Y[m])]
    a,b,c=np.log(Y[i-1:i+2]+1e-12); d=0.5*(a-c)/(a-2*b+c)
    return (i+d)*(fr[1]-fr[0])
