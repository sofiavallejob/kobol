(function(){
'use strict';
const M=JSON.parse(document.getElementById('model').textContent);
const E=makeEngine(M), K=E.K;
const NAMES=["Triangle","Triangle +","Triangle + Saw","Triangle + Saw +","Triangle + Saw ++","Saw","Saw +","Saw ++","Saw + Square","Saw + Square +","Saw + Square ++","Square","Square +","Square ++","Square + Pulse","Square + Pulse +","Square + Pulse ++","Pulse","Pulse +"];
const OCT_PER_V=1.501, CVF_MIN=2.64, CVF_MAX=9.9;
const $=s=>document.querySelector(s), clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
// ---------------- state ----------------
const S={
  lfo:{rate:0.55,sym:0,vmod:0}, vp:{gin:1,lag:0,gout:1}, noise:{lvl:0,pink:0},
  v1:{freq:6.27,wave:4.5,fine:0,vol:0.8,mod:1}, v2:{freq:5.604,wave:2.5,beat:0,vol:0.5,sync:0},
  vca:{src:1,depth:0,att:0.12,dec:0.35,sus:0.7,out:0.7,decayOn:1},
  vcf:{freq:6,res:3,amt:3,kbd:1,att:0.1,dec:0.4,sus:0.3},
  note:0,oct:0,gate:false,hold:false
};
const REDRAW=[];          // redraw functions of every knob and switch (used by presets)
// ---------------- ports and patch cables ----------------
const OUTS=['lfo.tri','lfo.sq','vp.out','vp.rev','noise.out','v1.out','v2.out','vca.env','vca.out','kb.cv','kb.gate','vcf.out','vcf.env'];
const OI={}; OUTS.forEach((p,i)=>OI[p]=i);
const conn={};            // input port -> output port
let PV=new Float64Array(OUTS.length), CV=new Float64Array(OUTS.length);   // previous / current sample values
// ---------------- wavetables ----------------
const N=2048, COS=new Float32Array(N), SIN=new Float32Array(N);
for(let i=0;i<N;i++){COS[i]=Math.cos(2*Math.PI*i/N);SIN[i]=Math.sin(2*Math.PI*i/N);}
const LIMITS=[256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1];
function buildTables(re,im){const acc=new Float32Array(N),snaps={},want=new Set(LIMITS);
  for(let k=1;k<=K;k++){const a=re[k-1],b=im[k-1];if(a||b){for(let i=0,idx=0;i<N;i++,idx=(idx+k)&(N-1))acc[i]+=a*COS[idx]-b*SIN[idx];}
    if(want.has(k))snaps[k]=Float32Array.from(acc);}return snaps;}
function pickTable(snaps,f,sr){const kmax=Math.floor(Math.min(18000,0.45*sr)/Math.max(f,1));for(const L of LIMITS)if(L<=kmax)return snaps[L];return snaps[1];}
const V=[null,{cvf:-1},{cvf:-1}];
function evalVoice(n,fcv,wcv){       // model at the effective control voltages (fcv in 1 V/oct volts)
  const st=n===1?S.v1:S.v2, base=n===1?st.freq+st.fine:st.freq+st.beat;
  const cvf=clamp(base+fcv/OCT_PER_V,CVF_MIN,CVF_MAX), cvw=clamp(st.wave+wcv,0,9);
  const f=E.pitch(cvf,cvw,true), s=E.shape(cvf,cvw,true,false);
  V[n]={f,cvf,cvw,fcv,wcv,snaps:buildTables(s.re,s.im),stamp:(V[n].stamp||0)+1};
}
let dirty={1:true,2:true};
// ---------------- audio ----------------
let ctx=null,node=null;
const JIT=M.noise.jitterCents, NOISEF=Math.pow(10,-68/20);
function gauss(){let u=Math.random()||1e-9,v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
const inp=(port,normal)=>{const o=conn[port];return o!==undefined?PV[OI[o]]:normal;};
const D={ph1:0,ph2:0,j1:0,j2:0,env:0,stage:0,syncPrev:0};
let meter=0,lfoLed=1;
function startAudio(){
  ctx=new (window.AudioContext||window.webkitAudioContext)();
  const sr=ctx.sampleRate, BS=2048;
  const lfoW=new WDF.OTALFO(sr), lag=new WDF.Lag(sr), nc=new WDF.NoiseColour(sr), mm=new WDF.MM5837(sr), pk=new WDF.Pink();
  const ota1=new WDF.OTA(sr), ota2=new WDF.OTA(sr), otaM=new WDF.OTA(sr);
  const vcf=new KVCF.SSM2040(2*sr), ads1=new KVCF.SSM2050(sr), ads2=new KVCF.SSM2050(sr);   // VCF runs at 2x
  let lawP=KVCF.law(6,3), ctl=0;
  const aj=Math.exp(-2*Math.PI*15/sr), sj=JIT*Math.sqrt((1+aj)/(1-aj)), dcA=Math.exp(-2*Math.PI*5/sr);
  let dcx=0,dcy=0;
  node=ctx.createScriptProcessor(BS,0,2);
  node.onaudioprocess=e=>{
    const L=e.outputBuffer.getChannelData(0),R=e.outputBuffer.getChannelData(1),n=L.length;
    // control-rate model update: pitch and waveform CVs averaged over the last block
    for(const k of [1,2]){
      const fp='v'+k+'.fcv', wp='v'+k+'.wcv';
      const fcv=conn[fp]!==undefined?blockMean[k].f:PV[OI['vp.out']], wcv=conn[wp]!==undefined?blockMean[k].w:0;
      if(dirty[k]||Math.abs(fcv-V[k].fcv)>0.004||Math.abs(wcv-V[k].wcv)>0.02){dirty[k]=false;evalVoice(k,fcv,wcv);}
      blockMean[k].f=0;blockMean[k].w=0;
    }
    const v1=V[1],v2=V[2],vc=S.vca,gate=S.gate||S.hold;
    lfoW.voff=S.lfo.sym*0.015; lag.set(S.vp.lag);
    const fv=S.vcf; ads1.decayOn=ads2.decayOn=!!vc.decayOn;
    const kbcv=(S.note+12*S.oct)/12, vol1=S.v1.vol*S.v1.vol, vol2=S.v2.vol*S.v2.vol, nl=S.noise.lvl*S.noise.lvl, outG=vc.out*vc.out;
    for(let i=0;i<n;i++){
      // LFO: CA3080 integrator (WDF) + Schmitt, 0.01-100 Hz, rate CV 1 V/oct
      lfoW.set(S.lfo.rate,clamp(inp('lfo.rate',0),-8,8));
      const tri=lfoW.tick(); CV[OI['lfo.tri']]=5*tri; CV[OI['lfo.sq']]=5*lfoW.q;
      // keyboard
      CV[OI['kb.cv']]=kbcv; CV[OI['kb.gate']]=gate?5:0;
      // voltage processor: (in1 x gain + in2) -> WDF lag -> x out gain; reversed output
      const vp=lag.tick(inp('vp.in1',kbcv)*S.vp.gin+inp('vp.in2',0))*S.vp.gout; CV[OI['vp.out']]=vp; CV[OI['vp.rev']]=-vp;
      // noise generator
      {const wn=mm.tick(), src=S.noise.pink?pk.tick(wn):wn; CV[OI['noise.out']]=nc.tick(src)*3*nl;}   // MM5837 -> white/pink -> WDF colouring network
      // ADS1 (VCA) and ADS2 (VCF): SSM2050 models, gate from keyboard or the gate input
      const g=conn['vca.gate']!==undefined?PV[OI[conn['vca.gate']]]>1:gate;
      D.env=ads1.process(g,vc.att,vc.dec,vc.sus); CV[OI['vca.env']]=5*D.env;
      const e2=ads2.process(g,fv.att,fv.dec,fv.sus); CV[OI['vcf.env']]=5*e2;
      // VCOs: pitch CV in 1 V/oct relative to the block the tables were built for
      const vib=S.lfo.vmod*2*tri/12;      // VCO mod (LFO) in octaves
      const fcv1=inp('v1.fcv',vp), fcv2=inp('v2.fcv',vp); blockMean[1].f+=fcv1/n; blockMean[2].f+=fcv2/n;
      blockMean[1].w+=inp('v1.wcv',0)/n; blockMean[2].w+=inp('v2.wcv',0)/n;
      D.j1=aj*D.j1+(1-aj)*sj*gauss(); D.j2=aj*D.j2+(1-aj)*sj*gauss();
      const f1=v1.f*Math.pow(2,fcv1-v1.fcv+(S.v1.mod?vib:0)+D.j1/1200), f2=v2.f*Math.pow(2,fcv2-v2.fcv+vib+D.j2/1200);
      const i1=f1/sr,i2=f2/sr; let wrapped=false,frac=0;
      D.ph1+=i1; if(D.ph1>=1){D.ph1-=1;wrapped=true;frac=D.ph1/i1;}
      D.ph2+=i2; if(D.ph2>=1)D.ph2-=1;
      if(conn['v2.sync']!==undefined){const s=PV[OI[conn['v2.sync']]];if(D.syncPrev<=0&&s>0)D.ph2=0;D.syncPrev=s;}
      else if(S.v2.sync&&wrapped)D.ph2=frac*i2;          // internal synchro from VCO 1
      const t1=pickTable(v1.snaps,f1,sr),t2=pickTable(v2.snaps,f2,sr);
      const x1=D.ph1*N,a1=x1|0,u1=x1-a1,x2=D.ph2*N,a2=x2|0,u2=x2-a2;
      const s1=t1[a1]+(t1[(a1+1)&(N-1)]-t1[a1])*u1+NOISEF*gauss(), s2=t2[a2]+(t2[(a2+1)&(N-1)]-t2[a2])*u2+NOISEF*gauss();
      // per-VCO CA3080 VCAs: volume knob + volume CV, modulated by LFO / ADS / both
      const mL=1-vc.depth*(0.5-0.5*tri), mE=1-vc.depth+vc.depth*D.env, m=vc.src===0?mL:vc.src===1?mE:mL*mE;
      const g1=clamp(vol1+inp('v1.vcv',0)/5,0,1)*m, g2=clamp(vol2+inp('v2.vcv',0)/5,0,1)*m;
      CV[OI['v1.out']]=ota1.tick(10*s1,g1); CV[OI['v2.out']]=ota2.tick(10*s2,g2);
      // VCF (SSM2040, control law learned from the RSF recordings): input normalled to the VCO mix plus noise
      const mix=inp('vcf.in',CV[OI['v1.out']]+CV[OI['v2.out']]+CV[OI['noise.out']]);
      if(ctl--<=0){ctl=15;   // control rate: cutoff = knob + ADS2 + keyboard tracking + CV input (volts)
        const cvF=fv.freq+fv.amt*e2+fv.kbd*kbcv/KVCF.octPerV+inp('vcf.fcv',0);
        lawP=KVCF.law(clamp(cvF,-2,11),clamp(fv.res+inp('vcf.rcv',0),0,10));lawP.g=Math.pow(10,lawP.gainDb/20);}
      const xf=mix/10, yf=0.5*(vcf.process(xf,lawP.fc,lawP.K,KVCF.drive,lawP.g)+vcf.process(xf,lawP.fc,lawP.K,KVCF.drive,lawP.g));
      CV[OI['vcf.out']]=10*yf;
      // output VCA (CA3080): input normalled to the VCF, CV normalled to ADS1
      let y=otaM.tick(inp('vca.in',CV[OI['vcf.out']]),clamp(inp('vca.cv',5*D.env)/5,0,1))*outG;
      CV[OI['vca.out']]=y; y=y/10*1.6;
      dcy=y-dcx+dcA*dcy; dcx=y; y=dcy;
      L[i]=R[i]=y>1?1:y<-1?-1:y;
      const t=PV;PV=CV;CV=t;            // one-sample delay between modules (allows feedback patches)
    }
    meter=D.env; lfoLed=lfoW.q;
  };
  node.connect(ctx.destination);
}
const blockMean={1:{f:0,w:0},2:{f:0,w:0}};
// ---------------- UI widgets ----------------
function svgEl(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e;}
function knob(host,o){
  const size=o.size,r=size/2,wrap=document.createElement('div');wrap.className='kn'+(o.big?' big':'');
  const lab=document.createElement('div');lab.className='lbl';lab.textContent=o.label;wrap.appendChild(lab);
  const sv=svgEl('svg',{width:size+18,height:size+18,viewBox:`-9 -9 ${size+18} ${size+18}`,class:'knob',tabindex:'0',role:'slider','aria-label':o.label});
  const nT=o.big?21:11;
  for(let i=0;i<nT;i++){const a=(-135+270*i/(nT-1)-90)*Math.PI/180,maj=i%5===0;
    sv.appendChild(svgEl('line',{x1:r+(r+7)*Math.cos(a),y1:r+(r+7)*Math.sin(a),x2:r+(r+(maj?1:3))*Math.cos(a),y2:r+(r+(maj?1:3))*Math.sin(a),class:'tick'+(maj?' maj':'')}));}
  sv.appendChild(svgEl('circle',{cx:r,cy:r+3,r:r-1,class:'shadow'}));sv.appendChild(svgEl('circle',{cx:r,cy:r,r:r-1,class:'skirt'}));
  const g=svgEl('g',{}),cap=r-(o.big?9:6);
  if(o.big)for(let i=0;i<40;i++){const a=2*Math.PI*i/40;g.appendChild(svgEl('line',{x1:r+cap*Math.cos(a),y1:r+cap*Math.sin(a),x2:r+(cap-4)*Math.cos(a),y2:r+(cap-4)*Math.sin(a),class:'knurl'}));}
  g.appendChild(svgEl('circle',{cx:r,cy:r,r:cap-(o.big?4:0),class:'cap'}));
  g.appendChild(svgEl('line',{x1:r,y1:r-(o.big?8:3),x2:r,y2:r-cap+(o.big?7:4),class:'ptr'}));
  sv.appendChild(g);wrap.appendChild(sv);
  if(o.under){const u=document.createElement('div');u.className='under';u.innerHTML=o.under;wrap.appendChild(u);}
  const ro=document.createElement('div');ro.className='ro';wrap.appendChild(ro);host.appendChild(wrap);
  const span=o.max-o.min;
  const draw=()=>{const v=o.get();g.setAttribute('transform',`rotate(${-135+270*(v-o.min)/span} ${r} ${r})`);sv.setAttribute('aria-valuenow',v.toFixed(3));ro.textContent=o.fmt?o.fmt(v):'';};
  const put=v=>{o.set(clamp(v,o.min,o.max));draw();};
  let y0=null,v0=0;
  sv.addEventListener('pointerdown',e=>{sv.setPointerCapture(e.pointerId);y0=e.clientY;v0=o.get();e.preventDefault();e.stopPropagation();});
  sv.addEventListener('pointermove',e=>{if(y0===null)return;put(v0+(y0-e.clientY)/(o.big?260:180)*span*(e.shiftKey?0.2:1));});
  const end=()=>{y0=null;};sv.addEventListener('pointerup',end);sv.addEventListener('pointercancel',end);
  sv.addEventListener('dblclick',()=>put(o.def));
  sv.addEventListener('wheel',e=>{e.preventDefault();put(o.get()-Math.sign(e.deltaY)*span/(e.shiftKey?400:100));},{passive:false});
  sv.addEventListener('keydown',e=>{const d={ArrowUp:1,ArrowRight:1,ArrowDown:-1,ArrowLeft:-1}[e.key];if(d){e.preventDefault();put(o.get()+d*span/(e.shiftKey?400:100));}});
  draw();REDRAW.push(draw);return{draw};
}
function toggle(host,o){
  const b=document.createElement('button');b.className='tg';b.type='button';b.setAttribute('aria-label',o.label);const n=o.pos.length;
  b.innerHTML=`<span class="tl">${o.pos[0]}</span><svg width="26" height="54" viewBox="0 0 26 54" aria-hidden="true"><rect x="2" y="2" width="22" height="50" rx="5" class="tgb"></rect><circle cx="13" cy="27" r="7" class="tgn"></circle><rect class="lev" x="9" width="8" height="20" rx="4"></rect></svg>${n===3?`<span class="tm">${o.pos[1]}</span>`:''}<span class="tl">${o.pos[n-1]}</span>`;
  const lev=b.querySelector('.lev');
  const draw=()=>{const p=o.get();lev.setAttribute('y',n===3?[5,17,29][p]:[5,29][p]);b.title=o.label+': '+o.pos[p];};
  b.onclick=()=>{o.set((o.get()+1)%n);draw();};host.appendChild(b);draw();REDRAW.push(draw);
}
function jack(host,lbl,port,out){
  const d=document.createElement('div');d.className='jk'+(out?' out':'');d.dataset.port=port;d.dataset.kind=out?'out':'in';
  d.innerHTML=`<span>${lbl}</span><svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true"><polygon points="20,2 35.6,11 35.6,29 20,38 4.4,29 4.4,11" class="nut"></polygon><circle cx="20" cy="20" r="10.5" class="sock"></circle><circle cx="20" cy="20" r="5" class="hole"></circle></svg>`;
  d.setAttribute('role','button');d.setAttribute('aria-label',(out?'Output ':'Input ')+lbl);host.appendChild(d);
}
const fmtHz=f=>f>=1000?(f/1000).toFixed(2)+' kHz':f.toFixed(f<100?2:1)+' Hz';
const mark=k=>{if(k&1)dirty[1]=true;if(k&2)dirty[2]=true;};
const lfoHz=x=>WDF.OTALFO.hz(x,0);
const lagT=x=>(WDF.P.lag.Rmin+WDF.P.lag.Rpot*(Math.expm1(3*x)/Math.expm1(3)))*WDF.P.lag.C;
const ms=t=>t<1?(t*1000).toFixed(0)+' ms':t.toFixed(2)+' s';
// ---------------- modules ----------------
(function(){ // LFO over voltage processor
  const m=$('#lv'),l=S.lfo,p=S.vp;
  knob(m.querySelector('.a1'),{label:'Rate',size:58,min:0,max:1,def:0.55,get:()=>l.rate,set:x=>l.rate=x,fmt:x=>{const f=lfoHz(x);return(f<0.1?f.toFixed(3):f<1?f.toFixed(2):f.toFixed(1))+' Hz';}});
  knob(m.querySelector('.a1'),{label:'Symmetry',size:46,min:-1,max:1,def:0,get:()=>l.sym,set:x=>l.sym=x,fmt:x=>(x>=0?'+':'')+(x*100).toFixed(0)+'%'});
  knob(m.querySelector('.a1'),{label:'VCO mod',size:46,min:0,max:1,def:0,get:()=>l.vmod,set:x=>l.vmod=x,fmt:x=>'±'+(x*2).toFixed(2)+' st'});
  const ja=m.querySelector('.ja');jack(ja,'Rate CV','lfo.rate');jack(ja,'Tri','lfo.tri',1);jack(ja,'Square','lfo.sq',1);
  knob(m.querySelector('.b1'),{label:'In gain',size:46,min:0,max:2,def:1,get:()=>p.gin,set:x=>p.gin=x,fmt:x=>'x'+x.toFixed(2)});
  knob(m.querySelector('.b1'),{label:'Lag time',size:46,min:0,max:1,def:0,get:()=>p.lag,set:x=>p.lag=x,fmt:x=>ms(lagT(x))});
  knob(m.querySelector('.b1'),{label:'Out gain',size:46,min:0,max:2,def:1,get:()=>p.gout,set:x=>p.gout=x,fmt:x=>'x'+x.toFixed(2)});
  const jb=m.querySelector('.jb');jack(jb,'In 1','vp.in1');jack(jb,'In 2','vp.in2');jack(jb,'Out','vp.out',1);jack(jb,'Rev','vp.rev',1);
})();
(function(){ // noise
  const m=$('#nz');knob(m.querySelector('.c1'),{label:'Level',size:58,min:0,max:1,def:0,get:()=>S.noise.lvl,set:x=>S.noise.lvl=x,fmt:x=>(x*10).toFixed(1)});
  toggle(m.querySelector('.c2'),{label:'Noise colour',pos:['White','Pink'],get:()=>S.noise.pink,set:p=>S.noise.pink=p});
  jack(m.querySelector('.jacks'),'Out','noise.out',1);
})();
const scopes={};
function waveIcons(){return `<div class="wic">${['M2 12 L8 3 L14 12 L20 3','M2 12 L10 3 L10 12 L18 3 L18 12','M2 12 V3 H9 V12 H16 V3 H20','M2 12 H7 V3 H9 V12 H16 V3 H18 V12'].map(d=>`<svg width="22" height="15" viewBox="0 0 22 15" aria-hidden="true"><path d="${d}"></path></svg>`).join('')}</div>`;}
function vco(n){
  const m=$('#vco'+n),st=n===1?S.v1:S.v2,top=m.querySelector('.top'),big=m.querySelector('.bigrow'),small=m.querySelector('.smallrow'),jk=m.querySelector('.jacks');
  const cv=document.createElement('canvas');cv.width=440;cv.height=170;cv.className='scope';top.prepend(cv);scopes[n]=cv;
  knob(big,{label:'Frequency',big:true,size:112,min:CVF_MIN,max:CVF_MAX,def:n===1?6.27:5.604,get:()=>st.freq,set:v=>{st.freq=v;mark(n);},fmt:v=>v.toFixed(2)+' V',under:'<div class="rng"><span>10 Hz</span><span>12 kHz</span></div>'});
  knob(big,{label:'Waveform',big:true,size:112,min:0,max:9,def:n===1?4.5:2.5,get:()=>st.wave,set:v=>{st.wave=v;mark(n);},fmt:v=>NAMES[Math.round(v/0.5)],under:waveIcons()});
  if(n===1){knob(small,{label:'Fine',size:46,min:-0.055,max:0.055,def:0,get:()=>st.fine,set:v=>{st.fine=v;mark(1);},fmt:v=>(v>=0?'+':'')+(v*OCT_PER_V*1200).toFixed(0)+' ct'});}
  else{knob(small,{label:'Beat',size:46,min:-0.035,max:0.035,def:0,get:()=>st.beat,set:v=>{st.beat=v;mark(2);},fmt:v=>(v>=0?'+':'')+(v*OCT_PER_V*1200).toFixed(0)+' ct'});}
  knob(small,{label:'Volume',size:46,min:0,max:1,def:n===1?0.8:0.5,get:()=>st.vol,set:v=>st.vol=v,fmt:v=>(v*10).toFixed(1)});
  if(n===1)toggle(small,{label:'VCO 1 modulation',pos:['Mod','Off'],get:()=>st.mod?0:1,set:p=>st.mod=p===0?1:0});
  else toggle(small,{label:'Synchro to VCO 1',pos:['Free','Sync'],get:()=>st.sync,set:p=>{st.sync=p;m.querySelector('.led').classList.toggle('on',p===1);}});
  jack(jk,'Freq','v'+n+'.fcv');jack(jk,'Wave','v'+n+'.wcv');jack(jk,'Vol','v'+n+'.vcv');
  if(n===2)jack(jk,'Sync','v2.sync');
  jack(jk,'Out','v'+n+'.out',1);
}
vco(1);vco(2);
(function(){ // output VCA with ADS
  const m=$('#vca'),v=S.vca;
  toggle(m.querySelector('.d1'),{label:'VCO VCA modulation source',pos:['LFO','ADS','Both'],get:()=>v.src,set:p=>v.src=p});
  knob(m.querySelector('.d1'),{label:'Depth',size:46,min:0,max:1,def:0,get:()=>v.depth,set:x=>v.depth=x,fmt:x=>(x*100).toFixed(0)+'%'});
  knob(m.querySelector('.d1'),{label:'Volume',size:58,min:0,max:1,def:0.7,get:()=>v.out,set:x=>v.out=x,fmt:x=>(x*10).toFixed(1)});
  knob(m.querySelector('.d2'),{label:'Attack',size:46,min:0,max:1,def:0.12,get:()=>v.att,set:x=>v.att=x,fmt:x=>ms(KVCF.SSM2050.time(x))});
  knob(m.querySelector('.d2'),{label:'Decay',size:46,min:0,max:1,def:0.35,get:()=>v.dec,set:x=>v.dec=x,fmt:x=>ms(KVCF.SSM2050.time(x))});
  knob(m.querySelector('.d2'),{label:'Sustain',size:46,min:0,max:1,def:0.7,get:()=>v.sus,set:x=>v.sus=x,fmt:x=>(x*100).toFixed(0)+'%'});
  const j=m.querySelector('.jacks');jack(j,'In','vca.in');jack(j,'CV','vca.cv');jack(j,'Gate','vca.gate');
  jack(j,'Key CV','kb.cv',1);jack(j,'Key gate','kb.gate',1);jack(j,'Env','vca.env',1);jack(j,'Out','vca.out',1);
  toggle(m.querySelector('.d1'),{label:'Decay (release)',pos:['On','Off'],get:()=>v.decayOn?0:1,set:p=>v.decayOn=p===0?1:0});
})();
(function(){ // VCF (SSM2040) with ADS2
  const m=$('#vcf'),v=S.vcf;
  knob(m.querySelector('.f1'),{label:'Frequency',size:70,min:0,max:10,def:6,get:()=>v.freq,set:x=>v.freq=x,fmt:x=>Math.round(KVCF.law(x,0).fc)+' Hz'});
  knob(m.querySelector('.f1'),{label:'Resonance',size:70,min:0,max:10,def:3,get:()=>v.res,set:x=>v.res=x,fmt:x=>(x*10).toFixed(0)+'%'+(KVCF.law(6,x).K>4?' osc':'')});
  knob(m.querySelector('.f2'),{label:'ADS2 amount',size:46,min:-10,max:10,def:3,get:()=>v.amt,set:x=>v.amt=x,fmt:x=>(x>=0?'+':'')+x.toFixed(1)+' V'});
  knob(m.querySelector('.f2'),{label:'Kbd track',size:46,min:0,max:1,def:1,get:()=>v.kbd,set:x=>v.kbd=x,fmt:x=>(x*100).toFixed(0)+'%'});
  knob(m.querySelector('.f3'),{label:'Attack',size:46,min:0,max:1,def:0.1,get:()=>v.att,set:x=>v.att=x,fmt:x=>ms(KVCF.SSM2050.time(x))});
  knob(m.querySelector('.f3'),{label:'Decay',size:46,min:0,max:1,def:0.4,get:()=>v.dec,set:x=>v.dec=x,fmt:x=>ms(KVCF.SSM2050.time(x))});
  knob(m.querySelector('.f3'),{label:'Sustain',size:46,min:0,max:1,def:0.3,get:()=>v.sus,set:x=>v.sus=x,fmt:x=>(x*100).toFixed(0)+'%'});
  const j=m.querySelector('.jacks');jack(j,'In','vcf.in');jack(j,'Freq','vcf.fcv');jack(j,'Res','vcf.rcv');
  jack(j,'ADS2','vcf.env',1);jack(j,'Out','vcf.out',1);
})();
// ---------------- cables ----------------
const rack=$('#rack'),wrapR=$('#rackwrap'),svgC=$('#cables');let scale=1;
const COLORS=['#E2401C','#F2B33D','#2F7DD1','#3BA55C','#C64FB6','#E7E9EA','#1FB3B3'];let ci=0;const ccol={};
function jackPos(port){const el=document.querySelector(`.jk[data-port="${port}"] svg`);const r=el.getBoundingClientRect(),q=rack.getBoundingClientRect();
  return[(r.left-q.left+r.width/2)/scale,(r.top-q.top+r.height/2)/scale];}
function cablePath(x1,y1,x2,y2){const d=Math.hypot(x2-x1,y2-y1),sag=30+d*0.22;return `M${x1} ${y1} C${x1} ${y1+sag} ${x2} ${y2+sag} ${x2} ${y2}`;}
function drawCables(tmp){
  svgC.setAttribute('width',rack.offsetWidth);svgC.setAttribute('height',rack.offsetHeight);svgC.innerHTML='';
  const add=(a,b,col,dash)=>{const[x1,y1]=a,[x2,y2]=b;
    svgC.appendChild(svgEl('path',{d:cablePath(x1,y1,x2,y2),stroke:'rgba(0,0,0,.35)','stroke-width':8,fill:'none',transform:'translate(2 3)'}));
    svgC.appendChild(svgEl('path',{d:cablePath(x1,y1,x2,y2),stroke:col,'stroke-width':6,fill:'none','stroke-linecap':'round','stroke-dasharray':dash||''}));
    for(const[x,y]of[a,b])svgC.appendChild(svgEl('circle',{cx:x,cy:y,r:9,fill:col,stroke:'rgba(0,0,0,.5)','stroke-width':1.5}));};
  for(const i in conn)add(jackPos(conn[i]),jackPos(i),ccol[i]);
  if(tmp)add(tmp.a,tmp.b,tmp.col);
  document.querySelectorAll('.jk').forEach(j=>j.classList.toggle('plugged',conn[j.dataset.port]!==undefined||Object.values(conn).includes(j.dataset.port)));
}
let drag=null;
rack.addEventListener('pointerdown',e=>{
  const j=e.target.closest('.jk');if(!j)return;e.preventDefault();
  let from=j.dataset.port,kind=j.dataset.kind,col=COLORS[ci%COLORS.length];
  if(kind==='in'&&conn[from]!==undefined){col=ccol[from];const o=conn[from];delete conn[from];from=o;kind='out';}  // pick the cable up
  else ci++;
  drag={from,kind,col};
});
window.addEventListener('pointermove',e=>{if(!drag)return;const q=rack.getBoundingClientRect();drawCables({a:jackPos(drag.from),b:[(e.clientX-q.left)/scale,(e.clientY-q.top)/scale],col:drag.col});});
window.addEventListener('pointerup',e=>{
  if(!drag)return;const el=document.elementFromPoint(e.clientX,e.clientY),j=el&&el.closest?el.closest('.jk'):null;
  if(j&&j.dataset.kind!==drag.kind){const inP=drag.kind==='in'?drag.from:j.dataset.port,outP=drag.kind==='out'?drag.from:j.dataset.port;conn[inP]=outP;ccol[inP]=drag.col;
    if(inP==='v1.fcv'||inP==='v1.wcv')dirty[1]=true;if(inP==='v2.fcv'||inP==='v2.wcv')dirty[2]=true;}
  drag=null;drawCables();ensureAudio();
});
window.__patch=(i,o)=>{if(o)conn[i]=o;else delete conn[i];mark(3);};
$('#clear').onclick=()=>{for(const k in conn)delete conn[k];drawCables();mark(3);};
// ---------------- keyboard ----------------
const kb=$('#keys'),KEYMAP='awsedftgyhujk'.split(''),BLACK=[1,3,6,8,10];
for(let i=0;i<=24;i++){const b=document.createElement('button');b.type='button';b.className='key'+(BLACK.includes(i%12)?' blk':'');b.dataset.s=i-12;b.setAttribute('aria-label','Note '+(i-12));kb.appendChild(b);}
function noteOn(s){S.note=s;S.gate=true;kb.querySelectorAll('.key').forEach(k=>k.classList.toggle('dn',+k.dataset.s===s));}
function noteOff(){S.gate=false;kb.querySelectorAll('.key.dn').forEach(k=>k.classList.remove('dn'));}
kb.addEventListener('pointerdown',e=>{const k=e.target.closest('.key');if(!k)return;ensureAudio();noteOn(+k.dataset.s);});
window.addEventListener('pointerup',()=>{if(S.gate)noteOff();});
const held=new Set();
window.addEventListener('keydown',e=>{if(e.target.closest&&e.target.closest('.knob'))return;if(e.repeat)return;const k=e.key.toLowerCase(),i=KEYMAP.indexOf(k);
  if(i>=0){ensureAudio();held.add(k);noteOn(i);} if(k==='z'){S.oct=Math.max(-2,S.oct-1);octLbl();} if(k==='x'){S.oct=Math.min(2,S.oct+1);octLbl();}});
window.addEventListener('keyup',e=>{const k=e.key.toLowerCase();held.delete(k);if(KEYMAP.includes(k)&&held.size===0)noteOff();});
const octLbl=()=>{$('#oct').textContent=(S.oct>0?'+':'')+S.oct;};
$('#octd').onclick=()=>{S.oct=Math.max(-2,S.oct-1);octLbl();};$('#octu').onclick=()=>{S.oct=Math.min(2,S.oct+1);octLbl();};
$('#hold').onclick=()=>{ensureAudio();S.hold=!S.hold;$('#hold').setAttribute('aria-pressed',String(S.hold));};
function ensureAudio(){if(!ctx){startAudio();$('#power').setAttribute('aria-pressed','true');$('#power').textContent='Audio on';}if(ctx.state!=='running')ctx.resume();}
$('#power').onclick=()=>{if(ctx&&ctx.state==='running'){ctx.suspend();$('#power').setAttribute('aria-pressed','false');$('#power').textContent='Audio off';}else ensureAudio();};
// ---------------- render ----------------
const drawn={1:-1,2:-1};
function drawScope(n){const cv=scopes[n],g=cv.getContext('2d'),w=cv.width,h=cv.height,v=V[n];if(!v.snaps)return;
  g.fillStyle='#07090A';g.fillRect(0,0,w,h);g.strokeStyle='#162024';g.lineWidth=1;
  for(let x=0;x<=w;x+=w/10){g.beginPath();g.moveTo(x,0);g.lineTo(x,h);g.stroke();}for(let y=0;y<=h;y+=h/4){g.beginPath();g.moveTo(0,y);g.lineTo(w,y);g.stroke();}
  const t=pickTable(v.snaps,v.f,ctx?ctx.sampleRate:48000);let pk=0.05;for(let i=0;i<N;i+=4)pk=Math.max(pk,Math.abs(t[i]));
  g.strokeStyle='#F2B33D';g.lineWidth=3;g.beginPath();for(let x=0;x<w;x++){const y=h/2-t[Math.floor(x/w*N)]/pk*(h/2-12);x?g.lineTo(x,y):g.moveTo(x,y);}g.stroke();}
function frame(){
  for(const n of[1,2]){if(!ctx&&dirty[n]){dirty[n]=false;evalVoice(n,0,0);}
    if(V[n].stamp!==drawn[n]){drawn[n]=V[n].stamp;drawScope(n);$(`#vco${n} .hz`).textContent=fmtHz(V[n].f);}}
  $('#vca .led').style.opacity=(0.15+0.85*Math.min(1,meter)).toFixed(2);$('#lv .led').classList.toggle('on',lfoLed>0);
  requestAnimationFrame(frame);}
evalVoice(1,0,0);evalVoice(2,0,0);dirty[1]=dirty[2]=false;requestAnimationFrame(frame);
function fit(){const w=wrapR.clientWidth-24;scale=clamp(w/rack.offsetWidth,0.5,1);rack.style.transform=`scale(${scale})`;
  wrapR.style.height=(rack.offsetHeight*scale)+'px';wrapR.style.overflowX=w/rack.offsetWidth<0.5?'auto':'hidden';drawCables();}
// refit when the window changes and when the rack itself grows (web fonts load after the first fit)
new ResizeObserver(fit).observe(wrapR);new ResizeObserver(fit).observe(rack);if(document.fonts)document.fonts.ready.then(fit);fit();

// ---------------- model info per module ----------------
const INFO={
  lv:{tags:[['wb','White-box']],h:'LFO and voltage processor',p:['<b>LFO:</b> CA3080 integrator and Schmitt comparator, wave digital filter (WDF) model of the schematic. Frequency within 0.7 % of an ngspice simulation of the same circuit, 0.01 to 100 Hz.','<b>Voltage processor:</b> WDF RC lag with a log pot. Step response within 0.7 mV of ngspice.','Component values are placeholders. The board\'s parts list has been found but is not applied yet.']},
  nz:{tags:[['wb','White-box']],h:'Noise generator',p:['Bit-exact model of the MM5837 shift-register noise chip (17 stages, period 131 071, confirmed by the parts list).','WDF output network within 0.7 dB of ngspice. The 100 kHz clock and the pink filter are assumptions.']},
  vco1:{tags:[['ml','Learned'],['wb','VCA: white-box']],h:'VCO 1',p:['Grey-box model learned from 437 recordings of the real oscillator.','<b>Pitch:</b> physical law, 1.4 cents error on held-out settings.','<b>Waveform:</b> saw core, fold and two comparators, set by a small network from the two knobs. 17.4 dB median harmonic SNR held out, against 13.6 dB for interpolation. Exact at every recorded setting.','Its VCA is a CA3080 WDF model.']},
  vco2:{tags:[['ml','Learned'],['as','Assumed same as VCO 1']],h:'VCO 2',p:['Same model as VCO 1. The schematic shows an identical circuit, but VCO 2 was never recorded.','Sync resets it on every cycle of VCO 1.']},
  vcf:{tags:[['ml','Learned'],['as','ADS2: assumed times']],h:'VCF (SSM2040)',p:['Four OTA stages with a CA3080 resonance limiter, from the schematic. The knob-to-filter law (physics formula plus a small network) is learned from 256 recordings.','<b>Held out:</b> 0.3 dB harmonic error, waveform error −26 dB, self-oscillation pitch within about 10 cents.','<b>ADS2:</b> SSM2050 circuit-behavioural model. Time range (1 ms to 10 s) is assumed, not measured.']},
  vca:{tags:[['wb','VCA: white-box'],['as','ADS1: assumed times']],h:'Output VCA and ADS1',p:['<b>VCA:</b> CA3080 with a WDF output network. DC transfer within 14 µV of ngspice.','<b>ADS1:</b> SSM2050 circuit-behavioural model. Time range (1 ms to 10 s) is assumed, not measured.']},
};
for(const id in INFO){const m=$('#'+id),inf=INFO[id];if(!m)continue;
  const b=document.createElement('button');b.type='button';b.className='ib';b.textContent='i';b.setAttribute('aria-label','How '+inf.h+' is modelled');b.setAttribute('aria-expanded','false');
  const pn=document.createElement('div');pn.className='ip';pn.hidden=true;
  pn.innerHTML=`<div class="ih">${inf.h}</div><div class="tags">${inf.tags.map(([c,t])=>`<span class="tag ${c}">${t}</span>`).join('')}</div>${inf.p.map(t=>`<p>${t}</p>`).join('')}`;
  b.onclick=e=>{e.stopPropagation();const open=pn.hidden;document.querySelectorAll('.ip').forEach(x=>x.hidden=true);document.querySelectorAll('.ib').forEach(x=>x.setAttribute('aria-expanded','false'));
    pn.hidden=!open;b.setAttribute('aria-expanded',String(open));};
  m.appendChild(b);m.appendChild(pn);}
document.addEventListener('click',e=>{if(!e.target.closest('.ip'))document.querySelectorAll('.ip').forEach(x=>x.hidden=true);});
// ---------------- presets ----------------
const DEF=JSON.parse(JSON.stringify(S));
const PRESETS=[
  {name:'Init',hint:'Default patch. Play the keys A to K.',set(){}},
  {name:'Waveform morph',hold:true,hint:'Hold is on: the LFO sweeps VCO 1 from triangle to pulse. Press Hold to stop.',
   set(){S.lfo.rate=0.3;S.v1.wave=4.5;S.v2.vol=0;Object.assign(S.vcf,{freq:9.5,res:0,amt:0});conn['v1.wcv']='lfo.tri';}},
  {name:'Filter sweep',hint:'Play the keys: ADS2 opens the resonant filter on every note.',
   set(){Object.assign(S.vcf,{freq:2.5,res:6.5,amt:6,att:0.15,dec:0.45,sus:0.15});Object.assign(S.vca,{att:0.05,dec:0.5,sus:0.7});S.v2.vol=0.4;}},
  {name:'Self-oscillation',hold:true,hint:'Both VCOs are off: you hear the filter whistling by itself. Play the keys and it follows the keyboard.',
   set(){S.v1.vol=0;S.v2.vol=0;Object.assign(S.vcf,{freq:5.5,res:10,amt:0,kbd:1});Object.assign(S.vca,{att:0.05,sus:1});}},
  {name:'LFO vibrato',hint:'Play the keys: the CA3080 LFO adds vibrato to VCO 1.',
   set(){S.lfo.rate=0.68;S.lfo.vmod=0.3;S.v1.mod=1;S.v2.vol=0;Object.assign(S.vcf,{freq:8,res:2,amt:0});}},
  {name:'Noise + filter',hint:'MM5837 noise through the resonant filter. Play the keys for wind-like sweeps.',
   set(){S.v1.vol=0;S.v2.vol=0;S.noise.lvl=0.8;Object.assign(S.vcf,{freq:4,res:8.5,amt:4,att:0.3,dec:0.6,sus:0.2});Object.assign(S.vca,{att:0.25,dec:0.6,sus:0.6});}},
  {name:'VCO 2 sync',hint:'VCO 2 is hard-synced to VCO 1. Play the keys and turn VCO 2 Frequency to sweep the sync.',
   set(){S.v1.vol=0;S.v2.vol=0.8;S.v2.sync=1;S.v2.freq=7.0;Object.assign(S.vcf,{freq:8,res:3,amt:2});}},
];
const pbar=$('#presets'),phint=$('#phint');
PRESETS.forEach(p=>{const b=document.createElement('button');b.type='button';b.className='btn pre';b.textContent=p.name;
  b.onclick=()=>{for(const k of['lfo','vp','noise','v1','v2','vca','vcf'])Object.assign(S[k],JSON.parse(JSON.stringify(DEF[k])));
    for(const k in conn)delete conn[k];p.set();
    S.hold=!!p.hold;$('#hold').setAttribute('aria-pressed',String(S.hold));
    $('#vco2 .led').classList.toggle('on',S.v2.sync===1);
    REDRAW.forEach(f=>f());mark(3);drawCables();ensureAudio();
    pbar.querySelectorAll('.pre').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));phint.textContent=p.hint;};
  pbar.appendChild(b);});
})();
