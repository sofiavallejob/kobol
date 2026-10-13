import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
exec(open('analyze.py').read().split("\nif __name__=='__main__'")[0])
pred={}
for v in ['final','bb']:
    P=np.zeros_like(C)
    for fi,va in enumerate(FOLDS['rand5']): z=np.load(f'runs/rand5_{v}_{fi}.npz'); P[va]=z['c'][va]
    pred[v]=P
def cyc(c,f,N=400):
    ph=np.arange(N)/N; y=np.zeros(N)
    for q in range(K):
        if (q+1)*f<18000: y+=np.abs(c[q])*np.cos(2*np.pi*(q+1)*ph+np.angle(c[q]))
    return y
sel=[(0,30),(3,27),(9,30),(12,34),(15,30),(17,38)]
fig,ax=plt.subplots(2,3,figsize=(7.2,3.6)); ax=ax.ravel()
for a,(w,n) in zip(ax,sel):
    j=key[(w,n)]; f=F0[j]
    a.plot(cyc(C[j],f),'k',lw=1.6,label='recording'); a.plot(cyc(pred['final'][j],f),'C0',lw=1,label='grey-box'); a.plot(cyc(pred['bb'][j],f),'C3--',lw=.8,label='black-box')
    a.set_title(f'{w}_{n}.wav, {f:.0f} Hz, held out',fontsize=7); a.set_xticks([]); a.set_yticks([])
ax[0].legend(fontsize=6,frameon=False)
plt.tight_layout(); plt.savefig('figs/fig_cycles.pdf'); plt.savefig('figs/fig_cycles.png',dpi=150); print('ok')
