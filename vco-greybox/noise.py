exec(open('stoch.py').read().split("if __name__")[0])
Ps=[]
for w,n in [(0,24),(0,26),(0,28)]:
    f0=e[(e.w==w)&(e.n==n)].f0.item(); a=analyze(w,n,f0); Ps.append(a['Pr']); f=a['f']
P=np.median(Ps,0)
# smooth on log-frequency, remove harmonic leftovers with a running minimum-ish percentile
from scipy.ndimage import percentile_filter
Psm=percentile_filter(P,30,size=9)
print('noise rms dBFS',10*np.log10(np.trapezoid(Psm,f)))
# FIR from magnitude (zero-phase, windowed), unit-variance white input
Nf=1024; fr=np.fft.rfftfreq(Nf,1/SR); mag=np.sqrt(np.interp(fr,f,Psm)*SR/2)
h=np.fft.irfft(mag,Nf); h=np.roll(h,Nf//2)*np.hanning(Nf)
np.save('noise_ir.npy',h.astype(np.float32))
x=np.convolve(np.random.randn(SR*2),h,'same'); print('check rms dBFS',20*np.log10(x.std()))
for fq in [50,100,500,2000,10000]: print(fq, round(10*np.log10(np.interp(fq,f,Psm)),1))
