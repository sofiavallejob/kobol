// Kobol VCO engine v4: physical pitch + grey-box waveshaper + exact residual correction at the measured grid
function b64bytes(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
function f16(s){const u=b64bytes(s);const h=new Uint16Array(u.buffer,u.byteOffset,u.length/2);const o=new Float32Array(h.length);
  for(let i=0;i<h.length;i++){const x=h[i],sg=x>>15?-1:1,e=(x>>10)&31,m=x&1023;
    o[i]=e===0?sg*m*5.960464477539063e-8:e===31?(m?NaN:sg*Infinity):sg*Math.pow(2,e-15)*(1+m/1024);}return o;}
function f32(s){const u=b64bytes(s);return new Float32Array(u.buffer.slice(u.byteOffset,u.byteOffset+u.byteLength));}
function forward(net,x){let h=Float32Array.from(x);net.forEach((L,li)=>{const y=new Float32Array(L.o);
  for(let r=0;r<L.o;r++){let s=L.b[r];const off=r*L.i;for(let c=0;c<L.i;c++)s+=L.W[off+c]*h[c];
    y[r]=li<net.length-1?s/(1+Math.exp(-s)):s;}h=y;});return h;}
function makeEngine(M){
  const K=M.rec.K,NW=19,NN=23,N0=21,TWO_PI=2*Math.PI;
  const w=f32(M.gb.w);let p=0;const net=M.gb.layers.map(L=>{const W=w.subarray(p,p+=L.o*L.i),b=w.subarray(p,p+=L.o);return{W,b,o:L.o,i:L.i};});
  const TRI=M.gb.tri, FH=M.gb.fh, FC2=M.gb.fc2||0, SL=M.gb.slew||[0,0], P=M.pitch;
  const sinc=x=>x===0?1:Math.sin(Math.PI*x)/(Math.PI*x);
  const C=f16(M.rec.c), idx={};M.rec.w.forEach((ww,i)=>idx[ww*100+M.rec.n[i]]=i);const gi=(ww,n)=>idx[ww*100+n];
  // physical pitch: exponential converter with waveform detune, plus per-step CV offsets (interpolated)
  function physLog2(cvf,cvw){const v=Math.max(0,Math.min(NN-1,(cvf-2.64)/0.33)),i0=Math.min(NN-2,Math.floor(v)),t=v-i0;
    const dl=P[5+i0]*(1-t)+P[6+i0]*t;
    const fe=Math.pow(2,P[0]+P[1]*(cvf-6.27)+P[2]*(cvw-4.5)+P[3]*(cvw-4.5)*(cvf-6.27)+dl);return Math.log2(1/(1/fe+Math.exp(P[4])));}
  function params(cvf,cvw){const o=forward(net,[(cvf-6.27)/3.63,(cvw-4.5)/4.5]);
    return {At:o[0],As:o[1],A1:o[2],th1:Math.tanh(o[3]),A2:o[4],th2:Math.tanh(o[5]),pt:0.1*Math.tanh(o[6]),fc:Math.exp(o[7])};}
  function synth(q,f,chain){const re=new Float32Array(K),im=new Float32Array(K);
    const comp=(th,k)=>{const p1=(1+th)/2,x1=-TWO_PI*k*p1,x2=-TWO_PI*k;const s1=sinc(k*f*SL[0]),s2=sinc(k*f*SL[1]);
      // 2*(e^{ix1}*s1 - e^{ix2}*s2)/(i*pi*k)   (comparator with finite rise and fall slew)
      const dr=Math.cos(x1)*s1-Math.cos(x2)*s2,di=Math.sin(x1)*s1-Math.sin(x2)*s2;return [2*di/(Math.PI*k),-2*dr/(Math.PI*k)];};
    for(let k=1;k<=K;k++){
      const tr=TRI[k-1][0],ti=TRI[k-1][1],ang=-TWO_PI*k*q.pt,cr=Math.cos(ang),ci=Math.sin(ang);
      let r=q.At*(tr*cr-ti*ci),i=q.At*(tr*ci+ti*cr);
      i+=q.As*2/(Math.PI*k);
      const c1=comp(q.th1,k),c2=comp(q.th2,k);r+=q.A1*c1[0]+q.A2*c2[0];i+=q.A1*c1[1]+q.A2*c2[1];
      const fk=k*f;let dr=1,di=fk/q.fc,den=dr*dr+di*di;[r,i]=[(r*dr+i*di)/den,(i*dr-r*di)/den];
      if(FC2){const d2=fk/FC2,den2=1+d2*d2;[r,i]=[(r+i*d2)/den2,(i-r*d2)/den2];}
      if(chain){const x=fk/FH,hr=x*x/(1+x*x),hi=x/(1+x*x);[r,i]=[r*hr-i*hi,r*hi+i*hr];}
      re[k-1]=r;im[k-1]=i;}
    const rot=Math.atan2(im[0],re[0])+Math.PI/2;
    for(let k=1;k<=K;k++){const a=-k*rot,c=Math.cos(a),s=Math.sin(a),r=re[k-1],q2=im[k-1];re[k-1]=r*c-q2*s;im[k-1]=r*s+q2*c;}
    return {re,im};}
  const RS=new Float32Array(NW*NN*2*K),RP=new Float32Array(NW*NN);
  for(let ww=0;ww<NW;ww++)for(let n=N0;n<N0+NN;n++){const i=gi(ww,n),g=ww*NN+(n-N0),cvf=2.64+(n-N0)*0.33,cvw=ww*0.5;
    const lf=physLog2(cvf,cvw),s=synth(params(cvf,cvw),Math.pow(2,lf),true);
    for(let k=0;k<K;k++){RS[g*2*K+k]=C[i*2*K+k]-s.re[k];RS[g*2*K+K+k]=C[i*2*K+K+k]-s.im[k];}
    RP[g]=Math.log2(M.rec.f0[i])-lf;}
  function corners(cvf,cvw){const u=Math.max(0,Math.min(NW-1,cvw/0.5)),v=Math.max(0,Math.min(NN-1,(cvf-2.64)/0.33));
    const u0=Math.min(NW-2,Math.floor(u)),v0=Math.min(NN-2,Math.floor(v)),a=u-u0,b=v-v0;
    return [[u0,v0,(1-a)*(1-b)],[u0+1,v0,a*(1-b)],[u0,v0+1,(1-a)*b],[u0+1,v0+1,a*b]];}
  return {K,FH,
    params,
    synthAt(cvf,cvw,f,chain=true){return synth(params(cvf,cvw),f,chain);},
    pitch(cvf,cvw,corr){let l=physLog2(cvf,cvw);if(corr)for(const [u,v,wt] of corners(cvf,cvw))l+=wt*RP[u*NN+v];return Math.pow(2,l);},
    shape(cvf,cvw,corr,chain=true){const f=Math.pow(2,physLog2(cvf,cvw));const s=synth(params(cvf,cvw),f,chain);
      if(corr)for(const [u,v,wt] of corners(cvf,cvw)){if(wt<1e-6)continue;const off=(u*NN+v)*2*K;for(let k=0;k<K;k++){s.re[k]+=wt*RS[off+k];s.im[k]+=wt*RS[off+K+k];}}
      return s;},
    recording(ww,n){const i=gi(ww,n);return {f0:M.rec.f0[i],re:C.subarray(i*2*K,i*2*K+K),im:C.subarray(i*2*K+K,(i+1)*2*K)};}
  };
}
if(typeof module!=='undefined')module.exports={makeEngine,f16,b64bytes};
