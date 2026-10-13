import numpy as np, pandas as pd
exec(open('pitch_cv.py').read().split("rows=[]")[0])
interior=lambda idx: idx[~np.isin(Nn[idx],[21,22,42,43])]
rows=[]
for scheme in ['rand5','col','row']:
    for va in folds(scheme):
        tr=interior(np.setdiff1d(np.arange(len(W)),va)); allI=np.arange(len(W))
        lf=model_logf(fit(tr,off=False),allI,off=False)
        for j in interior(va): rows.append(dict(scheme=scheme,j=j,cents=1200*(lf[j]-np.log2(F0[j]))))
d=pd.DataFrame(rows); print(d.groupby('scheme').cents.apply(lambda c:np.sqrt(np.mean(c**2))).round(2))
