import numpy as np, pandas as pd, glob, json, sys
exec(open('evalfw.py').read().split("if __name__")[0])
def nbrs(j,trs):
    return [key[(W[j]+dw,Nn[j]+dn)] for dw,dn in [(-1,0),(1,0),(0,-1),(0,1)] if (W[j]+dw,Nn[j]+dn) in key and key[(W[j]+dw,Nn[j]+dn)] in trs]
NAMES={'full':'grey-box','bb':'black-box (v2)'}
def harmonic_table():
    rows=[]; globs=[]
    for fpath in sorted(glob.glob('runs/*.npz')):
        sch,var,fi=fpath[5:-4].rsplit('_',2) if fpath.count('_')>2 else fpath[5:-4].split('_')
        z=np.load(fpath); c,va=z['c'],z['va']; g=json.loads(str(z['glob']))
        if g: globs.append(dict(scheme=sch,variant=var,fold=fi,**{q:(v if not isinstance(v,list) else str(np.round(v,2))) for q,v in g.items()}))
        trs=set(np.setdiff1d(np.arange(len(W)),va)); R={i:(C[i]-c[i])*Mk[i] for i in trs}
        for j in va:
            nb=nbrs(j,trs); cr=c[j]+(np.mean([R[i] for i in nb],0) if nb else 0)
            for tag,cc in [('',c[j]),(' + residual',cr)]:
                rows.append(dict(scheme=sch,method=NAMES.get(var,'grey-box '+var)+tag,j=j,fold=fi,snr=hsnr(cc,C[j],Mk[j]),lsd=lsd(cc,C[j],Mk[j]),
                                 reg=register(Nn[j]),fam=family(W[j])))
    # neighbour-average baseline for each scheme
    for sch in ['rand5','col','row']:
        for va in __import__('exp',fromlist=['x']).folds(sch) if False else FOLDS[sch]:
            trs=set(np.setdiff1d(np.arange(len(W)),va))
            for j in va:
                nb=nbrs(j,trs)
                if not nb: continue
                cc=np.mean([C[i]*Mk[i] for i in nb],0)
                rows.append(dict(scheme=sch,method='linear interpolation of neighbours',j=j,snr=hsnr(cc,C[j],Mk[j]),lsd=lsd(cc,C[j],Mk[j]),reg=register(Nn[j]),fam=family(W[j])))
    return pd.DataFrame(rows), pd.DataFrame(globs)
def folds(scheme):
    if scheme=='rand5':
        p=np.random.RandomState(0).permutation(len(W)); return [np.sort(p[i::5]) for i in range(5)]
    if scheme=='col': return [np.where((W>=1)&(W<=17)&(W%4==r))[0] for r in range(4)]
    if scheme=='row': return [np.where((Nn>=22)&(Nn<=42)&(Nn%4==r))[0] for r in range(4)]
FOLDS={s:folds(s) for s in ['rand5','col','row']}
def boot(x,n=2000):
    x=np.asarray(x); m=[np.median(np.random.choice(x,len(x))) for _ in range(n)]; return np.percentile(m,[2.5,97.5])
if __name__=='__main__':
    d,g=harmonic_table(); d.to_csv('harm_cv.csv',index=False); g.to_csv('globals_cv.csv',index=False)
    np.random.seed(0)
    out=[]
    for (sch,m),x in d.groupby(['scheme','method']):
        lo,hi=boot(x.snr); out.append(dict(scheme=sch,method=m,n=len(x),snr_med=x.snr.median(),snr_ci=f'[{lo:.1f}, {hi:.1f}]',lsd_med=x.lsd.median()))
    t=pd.DataFrame(out).round(2); print(t.to_string()); t.to_csv('summary_cv.csv',index=False)
    print(d[d.scheme=='rand5'].pivot_table(index='method',columns='fam',values='snr',aggfunc='median').round(1).to_string())
    print(d[d.scheme=='rand5'].pivot_table(index='method',columns='reg',values='snr',aggfunc='median').round(1).to_string())
    if len(g): print(g.groupby(['scheme','variant'])[['fh','Tr_us','fc2']].median().round(2).to_string()); print(g[g.variant.isin(['+slew','+all'])][['variant','fold','slew_us']].to_string())
