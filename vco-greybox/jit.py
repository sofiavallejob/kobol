import numpy as np, pandas as pd
exec(open('f0.py').read())
e=pd.read_csv('f0b.csv'); out=[]
for _,r in e.iterrows():
    w,n,f0=int(r.w),int(r.n),r.f0
    y=trim(D[(w,n)]); y=y-y.mean(); N=len(y); t=np.arange(N)/SR
    z=y*np.exp(-2j*np.pi*f0*t); Z=np.fft.fft(z); fr=np.fft.fftfreq(N,1/SR)
    bw=min(f0/2,100.0); Z[np.abs(fr)>bw]=0; zf=np.fft.ifft(Z)[2000:-2000]
    dec=max(1,int(SR/(4*bw))); ph=np.unwrap(np.angle(zf))[::dec]; tt=np.arange(len(ph))*dec/SR
    A=np.c_[tt,np.ones_like(tt)]; ph=ph-A@np.linalg.lstsq(A,ph,rcond=None)[0]
    c=1200*np.log2(1+np.gradient(ph,tt)/(2*np.pi)/f0)
    amp=np.abs(zf)[::dec]; out.append((w,n,f0,np.std(c),np.std(amp/amp.mean()), bw))
d=pd.DataFrame(out,columns=['w','n','f0','jit','amp','bw']); d.to_csv('jit.csv',index=False)
print(d.pivot(index='n',columns='w',values='jit').round(1).iloc[:, ::2].to_string())
print(d.pivot(index='n',columns='w',values='amp').mul(100).round(2).iloc[:, ::3].to_string())
