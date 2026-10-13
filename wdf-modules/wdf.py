"""Minimal wave digital filter library (voltage waves, a = v + R i, b = v - R i) and the
Kobol control-board models: noise colouring network, voltage-processor lag, LFO, CA3080 VCA."""
import numpy as np
class Leaf:
    def __init__(s,R): s.R=R; s.a=0.0; s.b=0.0
    def v(s): return 0.5*(s.a+s.b)
    def i(s): return 0.5*(s.a-s.b)/s.R
class Resistor(Leaf):
    def reflect(s): s.b=0.0; return s.b
    def incident(s,a): s.a=a
class Capacitor(Leaf):
    def __init__(s,C,fs,fwarp=None):   # bilinear; optional pre-warping at fwarp
        k=2*fs if not fwarp else 2*np.pi*fwarp/np.tan(np.pi*fwarp/fs)
        super().__init__(1.0/(k*C)); s.z=0.0
    def reflect(s): s.b=s.z; return s.b
    def incident(s,a): s.a=a; s.z=a
class ResVoltageSource(Leaf):       # adaptable: Thevenin source with series resistance R
    def __init__(s,R,V=0.0): super().__init__(R); s.V=V
    def reflect(s): s.b=s.V; return s.b
    def incident(s,a): s.a=a
class Series(Leaf):
    def __init__(s,p1,p2): s.p1,s.p2=p1,p2; super().__init__(p1.R+p2.R)
    def reflect(s): s.b=-(s.p1.reflect()+s.p2.reflect()); return s.b
    def incident(s,a):
        s.a=a; tot=s.p1.b+s.p2.b+a
        s.p1.incident(s.p1.b-s.p1.R/s.R*tot); s.p2.incident(s.p2.b-s.p2.R/s.R*tot)
class Parallel(Leaf):
    def __init__(s,p1,p2): s.p1,s.p2=p1,p2; s.G1,s.G2=1/p1.R,1/p2.R; super().__init__(1/(s.G1+s.G2))
    def reflect(s): s.b=(s.G1*s.p1.reflect()+s.G2*s.p2.reflect())*s.R; return s.b
    def incident(s,a):
        s.a=a; v2=s.b+a          # 2v at the node
        s.p1.incident(v2-s.p1.b); s.p2.incident(v2-s.p2.b)
def root_res_source(tree,V,Rs):   # non-adapted resistive voltage source at the root
    a=tree.reflect(); g=(Rs-tree.R)/(Rs+tree.R)
    tree.incident(g*a+(1-g)*V)
def root_current_source(tree,I):  # ideal current source at the root (OTA output)
    a=tree.reflect(); tree.incident(a+2*tree.R*I)

# ---------------- component values (ASSUMED until the parts list is available) ----------------
P=dict(
 noise=dict(R48=1e3,C8=1e-6,C9=10e-9,R50=100e3),      # source R, coupling C, shunt C, load R
 lag=dict(Rmin=1e3,Rpot=1e6,C=1e-6),                  # lag pot (log taper) + capacitor
 lfo=dict(C=1e-6,Rmin=4.7e3,Rpot=1e6,Vsat=13.5,ratio=0.5,sym=0.0),   # integrator C, rate R, Schmitt divider
 vca=dict(Rin=100e3,Rdiv=220.0,Rload=33e3,Cload=100e-12,Vt=0.02585,Iabc_max=5e-4,Rc=27e3,feed=0.002,Vfs=5.0)
)
# ---------------- models ----------------
class NoiseColour:
    def __init__(s,fs,p=P['noise']):
        s.Rl=Resistor(p['R50']); fc=1/(2*np.pi*(p['R48']*p['R50']/(p['R48']+p['R50']))*p['C9']); s.fc=fc
        s.Cs=Capacitor(p['C9'],fs,fc); s.Cc=Capacitor(p['C8'],fs)
        s.tree=Series(s.Cc,Parallel(s.Cs,s.Rl)); s.Rs=p['R48']
    def tick(s,vin): root_res_source(s.tree,vin,s.Rs); return s.Rl.v()
class Lag:
    def __init__(s,fs,p=P['lag']): s.p=p; s.C=Capacitor(p['C'],fs); s.set(0.5)
    def set(s,x): s.R=s.p['Rmin']+s.p['Rpot']*(np.expm1(3*x)/np.expm1(3))   # log-taper pot
    def tick(s,vin): root_res_source(s.C,vin,s.R); return s.C.v()
class LFO:
    """Op-amp integrator (WDF capacitor fed through the rate resistor from the comparator output)
    plus a Schmitt comparator; triangle = -Vc, square = comparator state."""
    def __init__(s,fs,p=P['lfo']): s.p=p; s.C=Capacitor(p['C'],fs); s.q=1.0; s.set(0.5)
    def set(s,x): s.R=s.p['Rmin']+s.p['Rpot']*(np.expm1(5*x)/np.expm1(5))
    def tick(s):
        sym=s.p['sym']; R=s.R*(1+sym if s.q>0 else 1-sym)
        # virtual ground: the rate resistor feeds an ideal current q*Vsat/R into the capacitor
        root_current_source(s.C,s.q*s.p['Vsat']/R); tri=-s.C.v()
        th=s.p['ratio']*s.p['Vsat']
        if s.q>0 and tri<-th: s.q=-1.0
        elif s.q<0 and tri>th: s.q=1.0
        return tri, s.q*s.p['Vsat']
class OTAVCA:
    """CA3080 VCA: input attenuator -> differential pair tanh -> output current into R||C (WDF)."""
    def __init__(s,fs,p=P['vca']):
        s.p=p; s.Rl=Resistor(p['Rload']); s.Cl=Capacitor(p['Cload'],fs); s.tree=Parallel(s.Rl,s.Cl)
    def tick(s,vin,vctl):
        p=s.p; vd=vin*p['Rdiv']/(p['Rin']+p['Rdiv'])
        iabc=min(max((vctl-0.6)/p['Rc'],0.0),p['Iabc_max'])
        iout=iabc*np.tanh(vd/(2*p['Vt']))+p['feed']*iabc     # feed: control feedthrough from mismatch
        root_current_source(s.tree,iout); return s.Rl.v()

# ---------------- MM5837 digital noise + pink filter, CA3080 OTA LFO (labelled hypotheses) ----------------
P['mm5837']=dict(fclk=100e3)                 # 17-stage LFSR, taps 14 and 17; clock ASSUMED (typ. ~100 kHz)
P['otalfo']=dict(C=100e-9,Vsat=13.5,ratio=0.5,Rin=100e3,Rdiv=220.0,Vt=0.02585,fmin=0.01,fmax=100.0)
class MM5837:
    """Bit-exact 17-bit LFSR (feedback = XOR of stages 14 and 17), clocked at fclk, integrated over
    each audio sample (the chip's binary output seen through the analog path), zero mean, +-1."""
    def __init__(s,fs,p=P['mm5837']): s.reg=(1<<17)-1; s.acc=0.0; s.step=p['fclk']/fs
    def clock(s):
        b=((s.reg>>13)^(s.reg>>16))&1; s.reg=((s.reg<<1)|b)&0x1FFFF; return 1.0 if (s.reg>>16)&1 else -1.0
    def tick(s):
        s.acc+=s.step; n=int(s.acc); s.acc-=n; tot=0.0
        for _ in range(n): tot+=s.clock()
        return tot/max(n,1)
class Pink:
    """-3 dB/oct filter (P. Kellet's refined method). The RSF pink network itself is not documented."""
    def __init__(s): s.b=[0.0]*7
    def tick(s,w):
        b=s.b; b[0]=0.99886*b[0]+w*0.0555179; b[1]=0.99332*b[1]+w*0.0750759; b[2]=0.96900*b[2]+w*0.1538520
        b[3]=0.86650*b[3]+w*0.3104856; b[4]=0.55000*b[4]+w*0.5329522; b[5]=-0.7616*b[5]-w*0.0168980
        y=b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6]+w*0.5362; b[6]=w*0.115926; return y*0.11
class OTALFO:
    """CA3080 integrator: the comparator square drives the OTA through a 100k/220R attenuator; the OTA
    output current (Iabc x tanh) charges C (WDF capacitor, ideal current source at the root).
    Iabc is exponential in the rate control, calibrated so the knob spans fmin..fmax.
    Symmetry = input offset voltage at the OTA (the trimmer)."""
    def __init__(s,fs,p=P['otalfo']):
        s.p=p; s.C=Capacitor(p['C'],fs); s.q=1.0; s.voff=0.0
        s.vd=p['Vsat']*p['Rdiv']/(p['Rin']+p['Rdiv']); s.g=np.tanh(s.vd/(2*p['Vt'])); s.th=p['ratio']*p['Vsat']; s.set(0.5)
    def iabc_for(s,f): return 4*s.p['C']*s.th*f/s.g
    def set(s,x,cv=0.0):     # x: knob 0..1 (4 decades); cv: volts, 1 V/oct
        f=s.p['fmin']*(s.p['fmax']/s.p['fmin'])**x*2**cv; s.iabc=s.iabc_for(f)
    def tick(s):
        p=s.p; i=s.iabc*np.tanh((s.q*s.vd+s.voff)/(2*p['Vt']))
        root_current_source(s.C,i); tri=s.C.v()
        # sub-sample switching: fold the overshoot past the threshold back (ramp reverses mid-sample)
        if s.q>0 and tri>s.th: s.q=-1.0; o=tri-s.th; s.C.z-=2*o; tri-=2*o
        elif s.q<0 and tri<-s.th: s.q=1.0; o=tri+s.th; s.C.z-=2*o; tri-=2*o
        return tri/s.th, s.q
