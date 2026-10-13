import numpy as np, glob
exec(open('analyze.py').read().split("\nif __name__=='__main__'")[0])
def score(pattern):
    s=[];sr=[]
    for fi,va in enumerate(FOLDS['rand5']):
        z=np.load(pattern.format(fi)); c=z['c']; trs=set(np.setdiff1d(np.arange(len(W)),va)); R={i:(C[i]-c[i])*Mk[i] for i in trs}
        for j in va:
            s.append(hsnr(c[j],C[j],Mk[j])); nb=nbrs(j,trs); sr.append(hsnr(c[j]+(np.mean([R[i] for i in nb],0) if nb else 0),C[j],Mk[j]))
    return np.median(s),np.median(sr)
res=[score('runs/rand5_final_{}.npz')]+[score(f'seeds/rand5_final_s{s}_'+'{}.npz') for s in (1,2)]
res=np.array(res); print('per seed (SNR, +residual):',res.round(2)); print('mean +- sd:',res.mean(0).round(2),res.std(0,ddof=1).round(2))
