// Wave digital filter library + control-board models (port of wdf.py, validated against ngspice)
const WDF=(function(){
  class Leaf{constructor(R){this.R=R;this.a=0;this.b=0;} v(){return 0.5*(this.a+this.b);}}
  class Resistor extends Leaf{reflect(){this.b=0;return 0;} incident(a){this.a=a;}}
  class Capacitor extends Leaf{constructor(C,fs,fw){const k=fw?2*Math.PI*fw/Math.tan(Math.PI*fw/fs):2*fs;super(1/(k*C));this.z=0;}
    reflect(){this.b=this.z;return this.b;} incident(a){this.a=a;this.z=a;}}
  class Series extends Leaf{constructor(p1,p2){super(p1.R+p2.R);this.p1=p1;this.p2=p2;}
    reflect(){this.b=-(this.p1.reflect()+this.p2.reflect());return this.b;}
    incident(a){this.a=a;const t=this.p1.b+this.p2.b+a;this.p1.incident(this.p1.b-this.p1.R/this.R*t);this.p2.incident(this.p2.b-this.p2.R/this.R*t);}}
  class Parallel extends Leaf{constructor(p1,p2){const G1=1/p1.R,G2=1/p2.R;super(1/(G1+G2));this.p1=p1;this.p2=p2;this.G1=G1;this.G2=G2;}
    reflect(){this.b=(this.G1*this.p1.reflect()+this.G2*this.p2.reflect())*this.R;return this.b;}
    incident(a){this.a=a;const v2=this.b+a;this.p1.incident(v2-this.p1.b);this.p2.incident(v2-this.p2.b);}}
  const rootV=(t,V,Rs)=>{const a=t.reflect(),g=(Rs-t.R)/(Rs+t.R);t.incident(g*a+(1-g)*V);};
  const rootI=(t,I)=>{const a=t.reflect();t.incident(a+2*t.R*I);};
  // component values: ASSUMED (era-typical) until the parts list is available
  const P={noise:{R48:1e3,C8:1e-6,C9:10e-9,R50:100e3},lag:{Rmin:1e3,Rpot:1e6,C:1e-6},
    lfo:{C:1e-6,Rmin:4.7e3,Rpot:1e6,Vsat:13.5,ratio:0.5},
    vca:{Rin:100e3,Rdiv:220,Rload:33e3,Cload:100e-12,Vt:0.02585,Imax:5e-4,feed:0.0002}};
  class NoiseColour{constructor(fs,p=P.noise){this.Rl=new Resistor(p.R50);const fc=1/(2*Math.PI*(p.R48*p.R50/(p.R48+p.R50))*p.C9);
      this.tree=new Series(new Capacitor(p.C8,fs),new Parallel(new Capacitor(p.C9,fs,fc),this.Rl));this.Rs=p.R48;}
    tick(v){rootV(this.tree,v,this.Rs);return this.Rl.v();}}
  class Lag{constructor(fs,p=P.lag){this.p=p;this.C=new Capacitor(p.C,fs);this.set(0.3);}
    set(x){this.R=this.p.Rmin+this.p.Rpot*(Math.expm1(3*x)/Math.expm1(3));}
    tick(v){rootV(this.C,v,this.R);return this.C.v();}}
  class LFO{constructor(fs,p=P.lfo){this.p=p;this.C=new Capacitor(p.C,fs);this.q=1;this.sym=0;this.set(0.4);}
    set(x){this.R=this.p.Rmin+this.p.Rpot*(Math.expm1(5*(1-x))/Math.expm1(5));}   // knob up = faster
    freq(){return 1/(4*this.R*this.p.C*this.p.ratio);}
    tick(){const R=this.R*(this.q>0?1+this.sym:1-this.sym);rootI(this.C,this.q*this.p.Vsat/R);const tri=-this.C.v(),th=this.p.ratio*this.p.Vsat;
      if(this.q>0&&tri<-th)this.q=-1;else if(this.q<0&&tri>th)this.q=1;return tri/th;}}   // triangle normalised to +-1
  class OTA{constructor(fs,p=P.vca){this.p=p;this.Rl=new Resistor(p.Rload);this.tree=new Parallel(this.Rl,new Capacitor(p.Cload,fs));
      this.g0=p.Rload*p.Imax*(p.Rdiv/(p.Rin+p.Rdiv))/(2*p.Vt);}   // small-signal gain at full control, for normalisation
    tick(vin,gain){const p=this.p,vd=vin*p.Rdiv/(p.Rin+p.Rdiv),iabc=Math.max(0,Math.min(1,gain))*p.Imax;
      rootI(this.tree,iabc*(Math.tanh(vd/(2*p.Vt))+p.feed));return this.Rl.v()/this.g0;}}
  // MM5837: 17-stage LFSR (XOR of stages 14 and 17), clock assumed 100 kHz, integrated per audio sample
  P.mm5837={fclk:100e3};
  class MM5837{constructor(fs){this.reg=0x1FFFF;this.acc=0;this.step=P.mm5837.fclk/fs;}
    tick(){this.acc+=this.step;const n=this.acc|0;this.acc-=n;let t=0;
      for(let k=0;k<n;k++){const b=((this.reg>>13)^(this.reg>>16))&1;this.reg=((this.reg<<1)|b)&0x1FFFF;t+=(this.reg>>16)&1?1:-1;}
      return t/Math.max(n,1);}}
  class Pink{constructor(){this.b=new Float64Array(7);}   // -3 dB/oct (Kellet); RSF's own pink network is undocumented
    tick(w){const b=this.b;b[0]=0.99886*b[0]+w*0.0555179;b[1]=0.99332*b[1]+w*0.0750759;b[2]=0.969*b[2]+w*0.153852;
      b[3]=0.8665*b[3]+w*0.3104856;b[4]=0.55*b[4]+w*0.5329522;b[5]=-0.7616*b[5]-w*0.016898;
      const y=b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6]+w*0.5362;b[6]=w*0.115926;return y*0.11;}}
  // CA3080 OTA integrator LFO: square -> 100k/220R -> OTA (Iabc x tanh) -> WDF capacitor; Schmitt comparator
  P.otalfo={C:100e-9,Vsat:13.5,ratio:0.5,Rin:100e3,Rdiv:220,Vt:0.02585,fmin:0.01,fmax:100};
  class OTALFO{constructor(fs,p=P.otalfo){this.p=p;this.C=new Capacitor(p.C,fs);this.q=1;this.voff=0;
      this.vd=p.Vsat*p.Rdiv/(p.Rin+p.Rdiv);this.g=Math.tanh(this.vd/(2*p.Vt));this.th=p.ratio*p.Vsat;this.set(0.5,0);}
    static hz(x,cv){const p=P.otalfo;return p.fmin*Math.pow(p.fmax/p.fmin,x)*Math.pow(2,cv||0);}
    set(x,cv){this.iabc=4*this.p.C*this.th*OTALFO.hz(x,cv)/this.g;}
    tick(){const i=this.iabc*Math.tanh((this.q*this.vd+this.voff)/(2*this.p.Vt));rootI(this.C,i);let t=this.C.v();
      if(this.q>0&&t>this.th){this.q=-1;const o=t-this.th;this.C.z-=2*o;t-=2*o;}
      else if(this.q<0&&t<-this.th){this.q=1;const o=t+this.th;this.C.z-=2*o;t-=2*o;}
      return t/this.th;}}
  return {NoiseColour,Lag,LFO,OTA,MM5837,Pink,OTALFO,P};
})();
