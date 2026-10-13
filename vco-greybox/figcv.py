import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
d=pd.read_csv('harm_cv.csv')
order=['linear interpolation of neighbours','black-box (v2)','black-box (v2) + residual','grey-box final','grey-box final + residual']
lab=['linear interp.','black-box','black-box + res.','grey-box','grey-box + res.']
fig,ax=plt.subplots(1,3,figsize=(7.2,2.4),sharey=True)
for a,(sch,tt) in zip(ax,[('rand5','random 5-fold'),('row','CV rows held out'),('col','waveform columns held out')]):
    data=[d[(d.scheme==sch)&(d.method==m)].snr.values for m in order]
    keep=[i for i,x in enumerate(data) if len(x)]
    a.boxplot([data[i] for i in keep],showfliers=False,widths=.6,medianprops=dict(color='k'))
    a.set_xticks(range(1,len(keep)+1)); a.set_xticklabels([lab[i] for i in keep],rotation=35,ha='right'); a.set_title(tt)
ax[0].set_ylabel('held-out harmonic SNR (dB)')
plt.tight_layout(); plt.savefig('figs/fig_cv.pdf'); plt.savefig('figs/fig_cv.png',dpi=150)
# error by held-out column
c=d[(d.scheme=='col')&(d.method.isin(['grey-box final','linear interpolation of neighbours']))]
import numpy as np
exec(open('evalfw.py').read().split("def raw")[0].split("exec(")[0]) if False else None
z=np.load('harm.npz',allow_pickle=True); Wv=z['meta'][:,0].astype(int)
c=c.assign(col=Wv[c.j]); t=c.pivot_table(index='col',columns='method',values='snr',aggfunc='median').round(1)
print(t.to_string()); t.to_csv('col_breakdown.csv')
fig,a=plt.subplots(figsize=(3.5,2.2)); 
for m,st in [('grey-box final','o-'),('linear interpolation of neighbours','s--')]: a.plot(t.index,t[m],st,ms=3,lw=1,label=lab[order.index(m)])
a.set_xlabel('held-out waveform column'); a.set_ylabel('median SNR (dB)'); a.legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig('figs/fig_col.pdf'); plt.savefig('figs/fig_col.png',dpi=150)
