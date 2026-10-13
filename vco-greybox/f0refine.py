# coarse ACF f0 per file -> median per CV step -> FFT peak refinement per file -> f0b.csv
import numpy as np, pandas as pd
exec(open('f0.py').read())
rows=[]
for (a,b),x in D.items():
    y=trim(x); f,q=f0_est(y); rows.append((a,b,f))
d=pd.DataFrame(rows,columns=['w','n','f0']); med=d.groupby('n').f0.median().to_dict()
for n in [42,43]:
    y=trim(D[(0,n)]); y=y-y.mean(); Y=np.abs(np.fft.rfft(y)); fr=np.fft.rfftfreq(len(y),1/SR); med[n]=fr[np.argmax(Y*(fr>1000))]
out=[]
for (a,b),x in D.items():
    y=trim(x); out.append((a,b,fft_refine(y,med[b]) if med[b]>40 else d[(d.w==a)&(d.n==b)].f0.item()))
pd.DataFrame(out,columns=['w','n','f0']).sort_values(['w','n']).to_csv('f0b.csv',index=False)
