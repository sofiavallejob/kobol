import numpy as np, pandas as pd
exec(open('f0.py').read())
K=256
e=pd.read_csv('f0b.csv')
rows=[];C=[];M=[]
for _,r in e.iterrows():
    w,n,f0=int(r.w),int(r.n),r.f0
    y=trim(D[(w,n)]); y=y-y.mean(); N=len(y)
    # use integer number of periods
    P=SR/f0; np_=int(N/P); L=int(np_*P); y=y[:L]
    t=np.arange(L)/SR; win=np.hanning(L)
    k=np.arange(1,K+1)
    # chunked projection
    c=np.zeros(K,complex)
    for i in range(0,K,32):
        kk=k[i:i+32,None]; c[i:i+32]=2*(np.exp(-2j*np.pi*kk*f0*t)*(y*win)).sum(1)/win.sum()
    # align: rotate so fundamental is pure sine phase (angle -pi/2)
    rot=np.angle(c[0])+np.pi/2
    c=c*np.exp(-1j*k*rot)
    valid=(k*f0<18000)
    rows.append((w,n,f0,(w)*0.5,2.64+(n-21)*0.33)); C.append(c); M.append(valid)
meta=pd.DataFrame(rows,columns=['w','n','f0','cvw','cvf'])
np.savez('harm.npz',C=np.array(C),M=np.array(M),meta=meta.values)
print(meta.head()); print(np.abs(np.array(C))[:5,:6])
