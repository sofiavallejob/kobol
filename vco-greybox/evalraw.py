# raw-WAV evaluation (spectral convergence, log-mag distance) for the main methods under 5-fold CV
import numpy as np, pandas as pd, sys
exec(open('analyze.py').read().split("\nif __name__=='__main__'")[0])
pc=pd.read_csv('pitch_cv.csv'); pc=pc[pc.scheme==sys.argv[1]] if len(sys.argv)>1 else pc[pc.scheme=='rand5']
sch=sys.argv[1] if len(sys.argv)>1 else 'rand5'; VAR=sys.argv[2] if len(sys.argv)>2 else 'full'
cents=lambda m:dict(zip(pc[pc.method==m].j,pc[pc.method==m].cents))
P={'phys':cents('physical'),'physr':cents('physical + residual'),'nn':cents('neural pitch unit (v2)'),'nnr':cents('neural pitch unit (v2) + residual')}
NS=SR//2
def render2(f,c): return render(f,c,NS)
rows=[]
for fi,va in enumerate(FOLDS[sch]):
    trs=set(np.setdiff1d(np.arange(len(W)),va))
    zg=np.load(f'runs/{sch}_{VAR}_{fi}.npz'); zb=np.load(f'runs/{sch}_bb_{fi}.npz'); cg,cb=zg['c'],zb['c']
    Rg={i:(C[i]-cg[i])*Mk[i] for i in trs}; Rb={i:(C[i]-cb[i])*Mk[i] for i in trs}
    for j in va[:int(__import__('os').environ.get('MAXJ','100000'))]:
        nb=nbrs(j,trs); ref=raw(j,0,NS); f0=F0[j]
        cand={'grey-box + physical pitch':(f0*2**(P['phys'][j]/1200),cg[j]),
              'grey-box + residual':(f0*2**(P['physr'][j]/1200),cg[j]+(np.mean([Rg[i] for i in nb],0) if nb else 0)),
              'black-box (v2)':(f0*2**(P['nn'][j]/1200),cb[j]),
              'black-box (v2) + residual':(f0*2**(P['nnr'][j]/1200),cb[j]+(np.mean([Rb[i] for i in nb],0) if nb else 0))}
        if nb: cand['linear interpolation of neighbours']=(2**np.mean([np.log2(F0[i]) for i in nb]) * 2**(np.mean([ (Nn[j]-Nn[i])*0.4958 for i in nb])), np.mean([C[i]*Mk[i] for i in nb],0))
        for m,(f,c) in cand.items():
            y=render2(f,c); rows.append(dict(method=m,j=j,sc=specconv(y,ref),lmd=mrstft(y,ref),reg=register(Nn[j]),fam=family(W[j])))
        a=trim(D[(W[j],Nn[j])]); rows.append(dict(method='reference: same recording, other segment',j=j,sc=specconv(a[NS:2*NS],ref),lmd=mrstft(a[NS:2*NS],ref),reg=register(Nn[j]),fam=family(W[j])))
    print('fold',fi,flush=True)
d=pd.DataFrame(rows); d.to_csv(f'raw_cv_{sch}_{VAR}.csv',index=False)
print(d.groupby('method')[['sc','lmd']].median().round(2).to_string())
print(d.pivot_table(index='method',columns='reg',values='sc',aggfunc='median').round(1).to_string())
print(d.pivot_table(index='method',columns='fam',values='sc',aggfunc='median').round(1).to_string())
