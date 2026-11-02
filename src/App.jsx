import { useState, useCallback, useMemo } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area, ScatterChart, Scatter, Cell } from "recharts";

function normalCDF(x) {
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign=x<0?-1:1; x=Math.abs(x)/Math.sqrt(2);
  const t=1/(1+p*x);
  const y=1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5*(1+sign*y);
}
function normalPDF(x){return Math.exp(-0.5*x*x)/Math.sqrt(2*Math.PI);}
function gaussianPDF(x,mu,sigma){return normalPDF((x-mu)/Math.max(sigma,1e-8))/Math.max(sigma,1e-8);}

function blackScholes(S,K,T,r,sigma,type='call'){
  if(T<=0)return{price:Math.max(0,type==='call'?S-K:K-S),delta:0,gamma:0,vega:0,theta:0,rho:0};
  const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
  const d2=d1-sigma*Math.sqrt(T);
  let price,delta,rho;
  if(type==='call'){price=S*normalCDF(d1)-K*Math.exp(-r*T)*normalCDF(d2);delta=normalCDF(d1);rho=K*T*Math.exp(-r*T)*normalCDF(d2)/100;}
  else{price=K*Math.exp(-r*T)*normalCDF(-d2)-S*normalCDF(-d1);delta=normalCDF(d1)-1;rho=-K*T*Math.exp(-r*T)*normalCDF(-d2)/100;}
  const gamma=normalPDF(d1)/(S*sigma*Math.sqrt(T));
  const vega=S*normalPDF(d1)*Math.sqrt(T)/100;
  const theta=type==='call'?(-S*normalPDF(d1)*sigma/(2*Math.sqrt(T))-r*K*Math.exp(-r*T)*normalCDF(d2))/365:(-S*normalPDF(d1)*sigma/(2*Math.sqrt(T))+r*K*Math.exp(-r*T)*normalCDF(-d2))/365;
  return{price,delta,gamma,vega,theta,rho};
}

function cleanDate(raw){
  // Handle ISO 8601: "2026-03-05T00:00:00+05:30" → "2026-03-05"
  if(!raw)return raw;
  const s=raw.trim().replace(/['"]/g,'');
  const tIdx=s.indexOf('T');
  return tIdx>0?s.slice(0,tIdx):s;
}

function parseCSV(text){
  const lines=text.trim().split('\n');
  const headers=lines[0].split(',').map(h=>h.trim().toLowerCase().replace(/['"]/g,''));
  const data=[];
  for(let i=1;i<lines.length;i++){
    const vals=lines[i].split(',').map(v=>v.trim().replace(/['"]/g,''));
    if(vals.length<2)continue;
    const row={};headers.forEach((h,j)=>{row[h]=vals[j];});data.push(row);
  }
  // Detect if newest-first by comparing first two date strings; if so, reverse
  const dateCol=findCol(headers,['date','time','datetime']);
  if(dateCol&&data.length>1){
    const d0=new Date(cleanDate(data[0][dateCol]));
    const d1=new Date(cleanDate(data[1][dateCol]));
    if(d0>d1)data.reverse(); // flip to oldest-first
  }
  // Clean date field in every row
  if(dateCol)data.forEach(r=>{r[dateCol]=cleanDate(r[dateCol]);});
  return{headers,data};
}
function findCol(headers,candidates){for(const c of candidates){const f=headers.find(h=>h.includes(c));if(f)return f;}return null;}
function computeReturns(closes){const r=[];for(let i=1;i<closes.length;i++)r.push(Math.log(closes[i]/closes[i-1]));return r;}

function computeVaRCVaR(returns,confidence=0.95){
  const sorted=[...returns].sort((a,b)=>a-b);
  const idx=Math.floor((1-confidence)*sorted.length);
  const varH=-sorted[idx];
  const tail=sorted.slice(0,idx);
  const cvarH=tail.length>0?-(tail.reduce((a,b)=>a+b,0)/tail.length):varH;
  const mean=returns.reduce((a,b)=>a+b,0)/returns.length;
  const std=Math.sqrt(returns.reduce((a,b)=>a+(b-mean)**2,0)/returns.length);
  const z={0.95:1.645,0.99:2.326}[confidence]||1.645;
  return{varHistorical:varH,cvarHistorical:cvarH,varParametric:-(mean-z*std),mean,std};
}
function computeEWMA(returns,lambda=0.94){const vols=[];let v=returns[0]**2;for(const r of returns){v=lambda*v+(1-lambda)*r**2;vols.push(Math.sqrt(v*252));}return vols;}
function computeRollingVol(returns,window=20){
  const vols=[];
  for(let i=0;i<returns.length;i++){
    if(i<window-1){vols.push(null);continue;}
    const s=returns.slice(i-window+1,i+1);const m=s.reduce((a,b)=>a+b,0)/s.length;
    vols.push(Math.sqrt(s.reduce((a,b)=>a+(b-m)**2,0)/s.length*252));
  }
  return vols;
}
function estimateGARCH(returns){
  const mean=returns.reduce((a,b)=>a+b,0)/returns.length;
  const variance=returns.reduce((a,b)=>a+(b-mean)**2,0)/returns.length;
  const alpha=0.09,beta=0.90,omega=variance*(1-alpha-beta);
  const forecasts=[];let h=variance;
  for(let t=1;t<=30;t++){h=omega+(alpha+beta)*h;forecasts.push({day:t,vol:parseFloat((Math.sqrt(h*252)*100).toFixed(2))});}
  const histVols=[];h=variance;
  for(const r of returns){h=omega+alpha*r**2+beta*h;histVols.push(Math.sqrt(h*252));}
  return{alpha,beta,omega,longRunVol:Math.sqrt(variance*252),forecasts,histVols};
}
function computeHistogram(returns,bins=40){
  const min=Math.min(...returns),max=Math.max(...returns),binSize=(max-min)/bins;
  const hist=Array(bins).fill(0).map((_,i)=>({x:parseFloat(((min+i*binSize+binSize/2)*100).toFixed(3)),count:0}));
  for(const r of returns){const idx=Math.min(Math.floor((r-min)/binSize),bins-1);hist[idx].count++;}
  return hist;
}
function computeSkewKurt(returns){
  const n=returns.length,m=returns.reduce((a,b)=>a+b,0)/n;
  const s=Math.sqrt(returns.reduce((a,b)=>a+(b-m)**2,0)/n);
  const skew=returns.reduce((a,b)=>a+((b-m)/s)**3,0)/n;
  const kurt=returns.reduce((a,b)=>a+((b-m)/s)**4,0)/n-3;
  return{skew,kurt};
}

// ── HMM: Baum-Welch EM + Viterbi ─────────────────────────────────────────────
function baumWelch(returns,N=3,maxIter=50){
  const T=returns.length;
  const EPS=1e-10;
  // Init params by splitting sorted returns into N quantile groups
  const sorted=[...returns].sort((a,b)=>a-b);
  const chunk=Math.floor(T/N);
  let means=Array.from({length:N},(_,i)=>{const s=sorted.slice(i*chunk,(i+1)*chunk);return s.reduce((a,b)=>a+b,0)/s.length;});
  means.sort((a,b)=>b-a); // highest mean = state 0
  const globalVar=returns.reduce((a,r)=>a+r**2,0)/T;
  let sigmas=means.map((_,i)=>Math.sqrt(globalVar)*(0.5+i*0.5));
  let A=Array.from({length:N},(_,i)=>Array.from({length:N},(_,j)=>{const base=0.15/(N-1);return i===j?1-0.15:base;}));
  let pi=Array(N).fill(1/N);

  for(let iter=0;iter<maxIter;iter++){
    const B=returns.map(r=>Array.from({length:N},(_,j)=>Math.max(gaussianPDF(r,means[j],sigmas[j]),EPS)));
    // Forward
    const alpha=Array.from({length:T},()=>new Array(N).fill(0));
    const scale=new Array(T);
    for(let j=0;j<N;j++)alpha[0][j]=pi[j]*B[0][j];
    scale[0]=alpha[0].reduce((a,b)=>a+b,0)||EPS;
    for(let j=0;j<N;j++)alpha[0][j]/=scale[0];
    for(let t=1;t<T;t++){
      for(let j=0;j<N;j++)alpha[t][j]=B[t][j]*alpha[t-1].reduce((s,a,i)=>s+a*A[i][j],0);
      scale[t]=alpha[t].reduce((a,b)=>a+b,0)||EPS;
      for(let j=0;j<N;j++)alpha[t][j]/=scale[t];
    }
    // Backward
    const beta=Array.from({length:T},()=>new Array(N).fill(1));
    for(let t=T-2;t>=0;t--){
      for(let i=0;i<N;i++)beta[t][i]=A[i].reduce((s,aij,j)=>s+aij*B[t+1][j]*beta[t+1][j],0)/(scale[t+1]||EPS);
    }
    // Gamma
    const gamma=Array.from({length:T},(_,t)=>{
      const g=Array.from({length:N},(_,j)=>alpha[t][j]*beta[t][j]);
      const s=g.reduce((a,b)=>a+b,0)||EPS;return g.map(v=>v/s);
    });
    // Xi
    const xi=Array.from({length:T-1},(_,t)=>{
      const m=Array.from({length:N},(_,i)=>Array.from({length:N},(_,j)=>alpha[t][i]*A[i][j]*B[t+1][j]*beta[t+1][j]));
      const s=m.reduce((a,row)=>a+row.reduce((b,v)=>b+v,0),0)||EPS;
      return m.map(row=>row.map(v=>v/s));
    });
    // Update
    pi=gamma[0].map(v=>Math.max(v,EPS));
    const pSum=pi.reduce((a,b)=>a+b,0);pi=pi.map(v=>v/pSum);
    for(let i=0;i<N;i++){
      const gS=gamma.slice(0,T-1).reduce((s,g)=>s+g[i],0)||EPS;
      for(let j=0;j<N;j++)A[i][j]=Math.max(xi.reduce((s,xt)=>s+xt[i][j],0)/gS,EPS);
      const rs=A[i].reduce((a,b)=>a+b,0);A[i]=A[i].map(v=>v/rs);
    }
    for(let j=0;j<N;j++){
      const gS=gamma.reduce((s,g)=>s+g[j],0)||EPS;
      means[j]=gamma.reduce((s,g,t)=>s+g[j]*returns[t],0)/gS;
      sigmas[j]=Math.max(0.0005,Math.sqrt(gamma.reduce((s,g,t)=>s+g[j]*(returns[t]-means[j])**2,0)/gS));
    }
  }

  // Viterbi decode
  const B=returns.map(r=>Array.from({length:N},(_,j)=>Math.max(gaussianPDF(r,means[j],sigmas[j]),1e-300)));
  const vit=Array.from({length:T},()=>new Array(N).fill(0));
  const psi=Array.from({length:T},()=>new Array(N).fill(0));
  for(let j=0;j<N;j++)vit[0][j]=Math.log(pi[j]+EPS)+Math.log(B[0][j]);
  for(let t=1;t<T;t++){
    for(let j=0;j<N;j++){
      let best=-Infinity,bs=0;
      for(let i=0;i<N;i++){const v=vit[t-1][i]+Math.log(A[i][j]+EPS);if(v>best){best=v;bs=i;}}
      vit[t][j]=best+Math.log(B[t][j]);psi[t][j]=bs;
    }
  }
  const states=new Array(T);
  states[T-1]=vit[T-1].indexOf(Math.max(...vit[T-1]));
  for(let t=T-2;t>=0;t--)states[t]=psi[t+1][states[t+1]];

  // Relabel: 0=Bull(highest mean),1=Sideways,2=Bear(lowest)
  const order=[...Array(N).keys()].sort((a,b)=>means[b]-means[a]);
  const remap=new Array(N);order.forEach((orig,newL)=>{remap[orig]=newL;});
  return{
    states:states.map(s=>remap[s]),
    means:order.map(o=>means[o]),
    sigmas:order.map(o=>sigmas[o]),
    A:order.map(o=>order.map(p=>A[o][p])),
    N
  };
}

function computeHMM(returns,dates,closes){
  const hmm=baumWelch(returns,3,50);
  const NAMES=['BULL','SIDEWAYS','BEAR'];
  const COLORS=['#00e8a0','#f0b040','#f04a4a'];
  const stateStats=NAMES.map((_,s)=>{
    const idx=hmm.states.reduce((a,st,i)=>st===s?[...a,i]:a,[]);
    const ret=idx.map(i=>returns[i]);
    if(!ret.length)return{name:NAMES[s],color:COLORS[s],count:0,pct:'0.0',mean:'0.00',vol:'0.00',sharpe:'—'};
    const m=ret.reduce((a,b)=>a+b,0)/ret.length;
    const v=Math.sqrt(ret.reduce((a,b)=>a+(b-m)**2,0)/ret.length*252);
    return{name:NAMES[s],color:COLORS[s],count:idx.length,pct:((idx.length/returns.length)*100).toFixed(1),
      mean:(m*252*100).toFixed(2),vol:(v*100).toFixed(2),sharpe:v>0?((m*252)/v).toFixed(3):'—'};
  });
  const trans=hmm.A.map((row,i)=>({from:NAMES[i],probs:row.map((p,j)=>({to:NAMES[j],p:(p*100).toFixed(1)}))}));
  const step=Math.max(1,Math.floor(closes.length/150));
  const regimeChart=hmm.states.map((s,i)=>({date:dates[i+1]||`D${i+1}`,price:parseFloat(closes[i+1].toFixed(2)),regime:s})).filter((_,i)=>i%step===0);
  const runs=[];let cur=hmm.states[0],start=0;
  for(let i=1;i<=hmm.states.length;i++){if(i===hmm.states.length||hmm.states[i]!==cur){runs.push({regime:cur,len:i-start});cur=hmm.states[i];start=i;}}
  const avgRun=NAMES.map((_,s)=>{const r=runs.filter(x=>x.regime===s);return r.length?r.reduce((a,b)=>a+b.len,0)/r.length:0;});
  const currentRegime=hmm.states[hmm.states.length-1];
  return{stateStats,trans,regimeChart,currentRegime,currentName:NAMES[currentRegime],currentColor:COLORS[currentRegime],avgRun,NAMES,COLORS,means:hmm.means,sigmas:hmm.sigmas};
}

// ── MAIN ──────────────────────────────────────────────────────────────────────
export default function QuantRiskTerminal(){
  const [data,setData]=useState(null);
  const [activeTab,setActiveTab]=useState('overview');
  const [opts,setOpts]=useState({S:22000,K:22000,T:30,r:6.5,sigma:15,type:'call'});
  const [dragging,setDragging]=useState(false);
  const [error,setError]=useState(null);

  const processFile=useCallback((file)=>{
    const reader=new FileReader();
    reader.onload=(e)=>{
      try{
        const{headers,data:rows}=parseCSV(e.target.result);
        const closeCol=findCol(headers,['close','ltp','last','price']);
        const dateCol=findCol(headers,['date','time','datetime']);
        if(!closeCol){setError("Could not find 'Close' column.");return;}
        const closes=rows.map(r=>parseFloat(r[closeCol])).filter(v=>!isNaN(v)&&v>0);
        if(closes.length<50){setError("Need at least 50 data points.");return;}
        const dates=dateCol?rows.map(r=>r[dateCol]):rows.map((_,i)=>`D${i+1}`);
        setData({closes,dates,closeCol,dateCol,headers,rowCount:rows.length});
        setOpts(p=>({...p,S:Math.round(closes[closes.length-1])}));
        setError(null);
      }catch(err){setError("Parse error: "+err.message);}
    };
    reader.readAsText(file);
  },[]);

  const C=useMemo(()=>{
    if(!data)return null;
    const returns=computeReturns(data.closes);
    const var95=computeVaRCVaR(returns,0.95);
    const var99=computeVaRCVaR(returns,0.99);
    const ewmaVols=computeEWMA(returns);
    const rollingVols=computeRollingVol(returns,20);
    const garch=estimateGARCH(returns);
    const histogram=computeHistogram(returns);
    const{skew,kurt}=computeSkewKurt(returns);
    const step=Math.max(1,Math.floor(data.closes.length/150));
    const priceData=data.closes.map((c,i)=>({date:data.dates[i]||`D${i}`,price:parseFloat(c.toFixed(2))})).filter((_,i)=>i%step===0);
    const volData=returns.map((r,i)=>({date:data.dates[i+1]||`D${i+1}`,ewma:ewmaVols[i]?parseFloat((ewmaVols[i]*100).toFixed(2)):null,rolling:rollingVols[i]?parseFloat((rollingVols[i]*100).toFixed(2)):null,garch:garch.histVols[i]?parseFloat((garch.histVols[i]*100).toFixed(2)):null})).filter((_,i)=>i%step===0);
    const hmm=computeHMM(returns,data.dates,data.closes);
    return{returns,var95,var99,ewmaVols,rollingVols,garch,histogram,priceData,volData,skew,kurt,currentVol:ewmaVols[ewmaVols.length-1],hmm};
  },[data]);

  const greeks=useMemo(()=>blackScholes(opts.S,opts.K,opts.T/365,opts.r/100,opts.sigma/100,opts.type),[opts]);
  const TABS=[['overview','PRICE & RETURNS'],['risk','VAR / CVAR'],['volatility','VOLATILITY'],['regimes','HMM REGIMES'],['options','OPTIONS GREEKS']];

  return(
    <div style={{
  background:'#060a0f',
  minHeight:'100vh',
  width:'100vw',
  overflowX:'hidden',
  display:'flex',
  flexDirection:'column',
  fontFamily:"'IBM Plex Mono','Courier New',monospace",
  color:'#b8ccd8',
  padding:0
}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-track{background:#0a0f18;}::-webkit-scrollbar-thumb{background:#1a3050;}
        .card{background:#0a1018;border:1px solid #162030;padding:20px;border-radius:3px;}
        .tab{background:none;border:none;border-bottom:2px solid transparent;cursor:pointer;font-family:inherit;letter-spacing:.12em;font-size:10px;padding:10px 18px;transition:all .2s;color:#3a6080;}
        .tab.on{color:#00c8f0;border-bottom-color:#00c8f0;}
        .tab:hover:not(.on){color:#6a9ab0;}
        .inp{background:#0a1018;border:1px solid #162030;color:#b8ccd8;padding:7px 10px;font-family:inherit;font-size:12px;width:100%;border-radius:2px;outline:none;}
        .inp:focus{border-color:#00c8f0;}
        .gbox{background:#080e18;border:1px solid #162030;padding:14px;border-radius:3px;text-align:center;}
        .lbl{font-size:9px;letter-spacing:.18em;color:#2a5070;margin-bottom:5px;text-transform:uppercase;}
        .upload{border:1px dashed #162030;border-radius:4px;padding:50px 30px;text-align:center;cursor:pointer;transition:all .3s;background:#08101a;}
        .upload.drag{border-color:#00c8f0;background:#0a1828;}
        .upload:hover{border-color:#2a5878;}
        tr:hover td{background:#0d1828 !important;}
      `}</style>

      <div style={{background:'#080d14',borderBottom:'1px solid #162030',padding:'14px 28px',display:'flex',alignItems:'center',justifyContent:'space-between'}}>
        <div>
          <div style={{fontSize:'9px',letterSpacing:'.3em',color:'#2a6080',marginBottom:'3px'}}>QUANTITATIVE RISK ANALYTICS</div>
          <div style={{fontSize:'19px',fontWeight:600,color:'#00c8f0',letterSpacing:'.04em',textShadow:'0 0 25px rgba(0,200,240,.4)'}}>NIFTY RISK TERMINAL</div>
        </div>
        <div style={{textAlign:'right',fontSize:'10px',color:'#2a5070',letterSpacing:'.08em',lineHeight:1.8}}>
          <div>VaR · CVaR · GARCH(1,1) · HMM(3-STATE) · BLACK-SCHOLES</div>
          <div style={{color:'#1a3a50'}}>BAUM-WELCH EM · VITERBI DECODING · GAUSSIAN EMISSIONS</div>
        </div>
      </div>

      <div style={{padding:'20px 28px'}}>
        {!data&&(
          <div style={{maxWidth:'560px',margin:'50px auto'}}>
            <div className="lbl" style={{textAlign:'center',marginBottom:'14px'}}>LOAD NIFTY OHLCV DATA</div>
            <div className={`upload ${dragging?'drag':''}`}
              onDragOver={e=>{e.preventDefault();setDragging(true);}}
              onDragLeave={()=>setDragging(false)}
              onDrop={e=>{e.preventDefault();setDragging(false);if(e.dataTransfer.files[0])processFile(e.dataTransfer.files[0]);}}
              onClick={()=>document.getElementById('fi').click()}>
              <div style={{fontSize:'28px',opacity:.3,marginBottom:'14px'}}>⬆</div>
              <div style={{color:'#5a8aaa',fontSize:'12px',marginBottom:'6px'}}>DROP CSV FILE HERE</div>
              <div style={{color:'#1a3a50',fontSize:'10px',letterSpacing:'.12em'}}>OR CLICK TO BROWSE</div>
              <input id="fi" type="file" accept=".csv" style={{display:'none'}} onChange={e=>{if(e.target.files[0])processFile(e.target.files[0]);}}/>
            </div>
            {error&&<div style={{color:'#f04a4a',fontSize:'11px',textAlign:'center',marginTop:'14px'}}>{error}</div>}
            <div style={{marginTop:'20px',background:'#0a1018',border:'1px solid #162030',borderRadius:'3px',padding:'14px',fontSize:'10px',color:'#1a3a50',lineHeight:2}}>
              <div style={{color:'#3a6a80',marginBottom:'6px',letterSpacing:'.1em'}}>EXPECTED CSV FORMAT:</div>
              <div>Date, Open, High, Low, <span style={{color:'#00c8f0'}}>Close</span>, Volume</div>
              <div>NSE / Zerodha / Kite / TradingView export supported</div>
            </div>
          </div>
        )}

        {data&&C&&(
          <>
            <div style={{display:'grid',gridTemplateColumns:'repeat(6,1fr)',gap:'10px',marginBottom:'20px'}}>
              {[
                {l:'LAST CLOSE',v:`₹${data.closes[data.closes.length-1].toLocaleString('en-IN',{maximumFractionDigits:0})}`,c:'#00c8f0'},
                {l:'EWMA VOL (ANN)',v:`${(C.currentVol*100).toFixed(1)}%`,c:'#00e8a0'},
                {l:'VaR 95% (1D)',v:`${(C.var95.varHistorical*100).toFixed(3)}%`,c:'#f0b040'},
                {l:'CVaR 95% (1D)',v:`${(C.var95.cvarHistorical*100).toFixed(3)}%`,c:'#f06040'},
                {l:'HMM REGIME',v:C.hmm.currentName,c:C.hmm.currentColor},
                {l:'OBSERVATIONS',v:C.returns.length.toLocaleString(),c:'#b8ccd8'},
              ].map(m=>(
                <div key={m.l} className="card">
                  <div className="lbl">{m.l}</div>
                  <div style={{fontSize:'15px',fontWeight:600,color:m.c}}>{m.v}</div>
                </div>
              ))}
            </div>

            <div style={{borderBottom:'1px solid #162030',marginBottom:'20px',display:'flex',alignItems:'center'}}>
              {TABS.map(([id,label])=>(
                <button key={id} className={`tab ${activeTab===id?'on':''}`} onClick={()=>setActiveTab(id)}>{label}</button>
              ))}
              <div style={{marginLeft:'auto',display:'flex',gap:'10px',alignItems:'center'}}>
                <span style={{fontSize:'9px',color:'#1a3a50'}}>{data.rowCount} ROWS</span>
                <button onClick={()=>setData(null)} style={{background:'none',border:'1px solid #162030',color:'#3a6080',fontSize:'9px',padding:'3px 10px',cursor:'pointer',fontFamily:'inherit',borderRadius:'2px'}}>RESET</button>
              </div>
            </div>

            {/* OVERVIEW */}
            {activeTab==='overview'&&(
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px'}}>
                <div className="card">
                  <div className="lbl">NIFTY CLOSE PRICE</div>
                  <ResponsiveContainer width="100%" height={210}>
                    <AreaChart data={C.priceData}>
                      <defs><linearGradient id="pg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#00c8f0" stopOpacity={.18}/><stop offset="95%" stopColor="#00c8f0" stopOpacity={0}/></linearGradient></defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#0d1a28"/>
                      <XAxis dataKey="date" tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} interval="preserveStartEnd"/>
                      <YAxis tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} axisLine={false} tickFormatter={v=>`₹${(v/1000).toFixed(0)}k`}/>
                      <Tooltip contentStyle={{background:'#0a1018',border:'1px solid #162030',fontSize:'11px'}} labelStyle={{color:'#3a6080'}}/>
                      <Area type="monotone" dataKey="price" stroke="#00c8f0" strokeWidth={1.5} fill="url(#pg)" dot={false}/>
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="card">
                  <div className="lbl">LOG RETURN DISTRIBUTION (%)</div>
                  <ResponsiveContainer width="100%" height={210}>
                    <BarChart data={C.histogram}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#0d1a28" vertical={false}/>
                      <XAxis dataKey="x" tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} interval={7} tickFormatter={v=>`${v}%`}/>
                      <YAxis tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} axisLine={false}/>
                      <Tooltip contentStyle={{background:'#0a1018',border:'1px solid #162030',fontSize:'11px'}} labelStyle={{color:'#3a6080'}} formatter={(v)=>[v,'Freq']}/>
                      <Bar dataKey="count" fill="#00e8a0" opacity={0.65} radius={[1,1,0,0]}/>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="card" style={{gridColumn:'span 2'}}>
                  <div className="lbl">DESCRIPTIVE STATISTICS</div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(6,1fr)',gap:'14px'}}>
                    {[
                      {l:'MEAN RETURN P.A.',v:`${(C.var95.mean*100*252).toFixed(2)}%`},
                      {l:'DAILY STD DEV',v:`${(C.var95.std*100).toFixed(3)}%`},
                      {l:'ANN VOLATILITY',v:`${(C.var95.std*Math.sqrt(252)*100).toFixed(2)}%`},
                      {l:'SKEWNESS',v:C.skew.toFixed(4),c:C.skew<-0.5?'#f06040':C.skew>0.5?'#00e8a0':'#b8ccd8'},
                      {l:'EXCESS KURTOSIS',v:C.kurt.toFixed(4),c:C.kurt>3?'#f0b040':'#b8ccd8'},
                      {l:'MAX 1D LOSS',v:`${(Math.min(...C.returns)*100).toFixed(3)}%`,c:'#f06040'},
                    ].map(s=>(
                      <div key={s.l} style={{textAlign:'center',background:'#080e18',border:'1px solid #162030',padding:'12px',borderRadius:'3px'}}>
                        <div className="lbl">{s.l}</div>
                        <div style={{fontSize:'13px',color:s.c||'#b8ccd8',fontWeight:500}}>{s.v}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* VAR */}
            {activeTab==='risk'&&(
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px'}}>
                <div className="card">
                  <div className="lbl">VALUE AT RISK METRICS</div>
                  <table style={{width:'100%',borderCollapse:'collapse',fontSize:'11px'}}>
                    <thead><tr>{['METRIC','CONF','HISTORICAL','PARAMETRIC','10D SCALED'].map(h=>(
                      <th key={h} style={{textAlign:'left',padding:'7px 5px',color:'#2a5070',fontSize:'8px',letterSpacing:'.1em',borderBottom:'1px solid #162030'}}>{h}</th>
                    ))}</tr></thead>
                    <tbody>
                      {[
                        {m:'VaR',c:'95%',h:C.var95.varHistorical,p:C.var95.varParametric},
                        {m:'VaR',c:'99%',h:C.var99.varHistorical,p:C.var99.varParametric},
                        {m:'CVaR',c:'95%',h:C.var95.cvarHistorical,p:C.var95.cvarHistorical*1.15},
                        {m:'CVaR',c:'99%',h:C.var99.cvarHistorical,p:C.var99.cvarHistorical*1.15},
                      ].map((row,i)=>(
                        <tr key={i} style={{borderBottom:'1px solid #0d1828'}}>
                          <td style={{padding:'9px 5px',color:'#7aaac0'}}>{row.m}</td>
                          <td style={{padding:'9px 5px',color:'#3a6080'}}>{row.c}</td>
                          <td style={{padding:'9px 5px',color:'#f0b040',fontWeight:600}}>{(row.h*100).toFixed(3)}%</td>
                          <td style={{padding:'9px 5px',color:'#f08040'}}>{(row.p*100).toFixed(3)}%</td>
                          <td style={{padding:'9px 5px',color:'#f06040'}}>{(row.h*Math.sqrt(10)*100).toFixed(3)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{marginTop:'12px',fontSize:'9px',color:'#1a3a50',lineHeight:2.2}}>
                    <div>▸ Historical VaR: empirical distribution, no normality assumption</div>
                    <div>▸ Parametric: Gaussian — underestimates fat-tail risk</div>
                    <div>▸ 10D: Basel √10 scaling rule</div>
                  </div>
                </div>
                <div className="card">
                  <div className="lbl">DISTRIBUTION WITH VAR THRESHOLDS</div>
                  <ResponsiveContainer width="100%" height={230}>
                    <BarChart data={C.histogram}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#0d1828" vertical={false}/>
                      <XAxis dataKey="x" tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} interval={7}/>
                      <YAxis tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} axisLine={false}/>
                      <Tooltip contentStyle={{background:'#0a1018',border:'1px solid #162030',fontSize:'10px'}} labelStyle={{color:'#3a6080'}} formatter={(v)=>[v,'Freq']}/>
                      <Bar dataKey="count" fill="#1a4060" radius={[1,1,0,0]}/>
                      <ReferenceLine x={parseFloat((-C.var95.varHistorical*100).toFixed(2))} stroke="#f0b040" strokeDasharray="4 2" label={{value:'VaR95',fill:'#f0b040',fontSize:9,position:'top'}}/>
                      <ReferenceLine x={parseFloat((-C.var99.varHistorical*100).toFixed(2))} stroke="#f04040" strokeDasharray="4 2" label={{value:'VaR99',fill:'#f04040',fontSize:9,position:'top'}}/>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="card" style={{gridColumn:'span 2'}}>
                  <div className="lbl">RUPEE CAPITAL AT RISK — NIFTY FUTURES (LOT = 25 UNITS)</div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:'12px'}}>
                    {[1,5,10,25].map(lots=>{
                      const exp=data.closes[data.closes.length-1]*lots*25;
                      return(
                        <div key={lots} style={{background:'#080e18',border:'1px solid #162030',padding:'14px',borderRadius:'3px'}}>
                          <div className="lbl">{lots} LOT{lots>1?'S':''}</div>
                          <div style={{fontSize:'10px',color:'#2a5070',marginBottom:'3px'}}>Notional</div>
                          <div style={{fontSize:'12px',color:'#b8ccd8',marginBottom:'10px'}}>₹{exp.toLocaleString('en-IN',{maximumFractionDigits:0})}</div>
                          <div style={{fontSize:'10px',color:'#2a5070'}}>VaR 95%</div>
                          <div style={{fontSize:'14px',color:'#f0b040',fontWeight:600,marginBottom:'6px'}}>₹{(exp*C.var95.varHistorical).toLocaleString('en-IN',{maximumFractionDigits:0})}</div>
                          <div style={{fontSize:'10px',color:'#2a5070'}}>VaR 99%</div>
                          <div style={{fontSize:'14px',color:'#f04040',fontWeight:600,marginBottom:'6px'}}>₹{(exp*C.var99.varHistorical).toLocaleString('en-IN',{maximumFractionDigits:0})}</div>
                          <div style={{fontSize:'10px',color:'#2a5070'}}>CVaR 95%</div>
                          <div style={{fontSize:'13px',color:'#f07040',fontWeight:600}}>₹{(exp*C.var95.cvarHistorical).toLocaleString('en-IN',{maximumFractionDigits:0})}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* VOLATILITY */}
            {activeTab==='volatility'&&(
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px'}}>
                <div className="card" style={{gridColumn:'span 2'}}>
                  <div className="lbl">VOLATILITY MODELS — ANNUALIZED (%)</div>
                  <ResponsiveContainer width="100%" height={230}>
                    <LineChart data={C.volData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#0d1828"/>
                      <XAxis dataKey="date" tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} interval="preserveStartEnd"/>
                      <YAxis tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} axisLine={false} tickFormatter={v=>`${v}%`}/>
                      <Tooltip contentStyle={{background:'#0a1018',border:'1px solid #162030',fontSize:'11px'}} labelStyle={{color:'#3a6080'}} formatter={(v,n)=>[`${v}%`,n]}/>
                      <Line type="monotone" dataKey="ewma" stroke="#00c8f0" strokeWidth={1.5} dot={false} name="EWMA"/>
                      <Line type="monotone" dataKey="rolling" stroke="#00e8a0" strokeWidth={1.5} dot={false} name="Rolling 20D"/>
                      <Line type="monotone" dataKey="garch" stroke="#f0b040" strokeWidth={1.5} dot={false} name="GARCH(1,1)"/>
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{display:'flex',gap:'20px',marginTop:'8px'}}>
                    {[['EWMA (λ=0.94)','#00c8f0'],['ROLLING 20D','#00e8a0'],['GARCH(1,1)','#f0b040']].map(([l,c])=>(
                      <div key={l} style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'9px',color:'#3a6080'}}>
                        <div style={{width:'18px',height:'2px',background:c}}/>{l}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="card">
                  <div className="lbl">GARCH(1,1) PARAMETERS</div>
                  <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'10px',marginBottom:'14px'}}>
                    {[
                      {l:'ω (OMEGA)',v:(C.garch.omega*1e6).toFixed(4),sub:'×10⁻⁶'},
                      {l:'α (ARCH)',v:C.garch.alpha.toFixed(4)},
                      {l:'β (GARCH)',v:C.garch.beta.toFixed(4)},
                      {l:'PERSISTENCE α+β',v:(C.garch.alpha+C.garch.beta).toFixed(4),c:(C.garch.alpha+C.garch.beta)>0.99?'#f04040':'#00e8a0'},
                    ].map(p=>(
                      <div key={p.l} className="gbox">
                        <div className="lbl">{p.l}</div>
                        <div style={{fontSize:'17px',color:p.c||'#b8ccd8',fontWeight:600}}>{p.v}<span style={{fontSize:'10px',color:'#3a6080'}}>{p.sub||''}</span></div>
                      </div>
                    ))}
                  </div>
                  <div style={{background:'#080e18',border:'1px solid #162030',padding:'12px',borderRadius:'3px',fontSize:'10px',color:'#1a3a50',lineHeight:2.2}}>
                    <div>σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁</div>
                    <div>Long-run vol: <span style={{color:'#00c8f0'}}>{(C.garch.longRunVol*100).toFixed(2)}%</span></div>
                    <div>EWMA now: <span style={{color:'#00e8a0'}}>{(C.currentVol*100).toFixed(2)}%</span></div>
                  </div>
                </div>
                <div className="card">
                  <div className="lbl">GARCH 30-DAY FORECAST</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={C.garch.forecasts}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#0d1828"/>
                      <XAxis dataKey="day" tick={{fill:'#1a3a50',fontSize:8}} tickLine={false}/>
                      <YAxis tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} axisLine={false} tickFormatter={v=>`${v.toFixed(0)}%`}/>
                      <Tooltip contentStyle={{background:'#0a1018',border:'1px solid #162030',fontSize:'11px'}} formatter={(v)=>[`${v}%`,'Forecast']}/>
                      <ReferenceLine y={C.garch.longRunVol*100} stroke="#1a4060" strokeDasharray="6 3" label={{value:'LR vol',fill:'#2a5a80',fontSize:9}}/>
                      <Line type="monotone" dataKey="vol" stroke="#f0b040" strokeWidth={2} dot={false}/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* HMM REGIMES */}
            {activeTab==='regimes'&&(
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px'}}>
                {/* Banner */}
                <div style={{gridColumn:'span 2',background:'#0a1018',border:`1px solid ${C.hmm.currentColor}44`,borderLeft:`3px solid ${C.hmm.currentColor}`,padding:'16px 22px',borderRadius:'3px',display:'flex',alignItems:'center',justifyContent:'space-between'}}>
                  <div>
                    <div className="lbl">CURRENT DETECTED REGIME</div>
                    <div style={{fontSize:'28px',fontWeight:600,color:C.hmm.currentColor,textShadow:`0 0 24px ${C.hmm.currentColor}55`,letterSpacing:'.06em'}}>{C.hmm.currentName} MARKET</div>
                  </div>
                  <div style={{textAlign:'right',fontSize:'10px',color:'#2a5070',lineHeight:2}}>
                    <div>Model: <span style={{color:'#b8ccd8'}}>3-State Gaussian HMM</span></div>
                    <div>Learning: <span style={{color:'#b8ccd8'}}>Baum-Welch EM ({'>'}50 iterations)</span></div>
                    <div>Decoding: <span style={{color:'#b8ccd8'}}>Viterbi algorithm</span></div>
                    <div>Emissions: <span style={{color:'#b8ccd8'}}>Gaussian on log-returns</span></div>
                  </div>
                </div>

                {/* Regime-colored scatter price chart */}
                <div className="card" style={{gridColumn:'span 2'}}>
                  <div className="lbl">NIFTY PRICE — COLORED BY HMM REGIME</div>
                  <div style={{display:'flex',gap:'22px',marginBottom:'10px'}}>
                    {C.hmm.NAMES.map((n,i)=>(
                      <div key={n} style={{display:'flex',alignItems:'center',gap:'7px',fontSize:'9px',color:'#3a6080'}}>
                        <div style={{width:'10px',height:'10px',background:C.hmm.COLORS[i],borderRadius:'2px'}}/>
                        {n} REGIME
                      </div>
                    ))}
                  </div>
                  <ResponsiveContainer width="100%" height={220}>
                    <ScatterChart margin={{top:5,right:5,bottom:5,left:5}}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#0d1828"/>
                      <XAxis dataKey="date" tick={{fill:'#1a3a50',fontSize:7}} tickLine={false} interval="preserveStartEnd"/>
                      <YAxis tick={{fill:'#1a3a50',fontSize:8}} tickLine={false} axisLine={false} tickFormatter={v=>`₹${(v/1000).toFixed(0)}k`}/>
                      <Tooltip cursor={false} content={({payload})=>{
                        if(!payload||!payload.length)return null;
                        const d=payload[0]?.payload;if(!d)return null;
                        return<div style={{background:'#0a1018',border:'1px solid #162030',padding:'8px 12px',borderRadius:'3px',fontSize:'10px'}}>
                          <div style={{color:'#3a6080',marginBottom:'3px'}}>{d.date}</div>
                          <div style={{color:'#b8ccd8'}}>₹{d.price?.toLocaleString('en-IN')}</div>
                          <div style={{color:C.hmm.COLORS[d.regime],fontWeight:600,marginTop:'3px'}}>{C.hmm.NAMES[d.regime]}</div>
                        </div>;
                      }}/>
                      {[0,1,2].map(s=>(
                        <Scatter key={s} name={C.hmm.NAMES[s]}
                          data={C.hmm.regimeChart.filter(d=>d.regime===s)}
                          fill={C.hmm.COLORS[s]} opacity={0.8} r={2}>
                          {C.hmm.regimeChart.filter(d=>d.regime===s).map((_,i)=><Cell key={i} fill={C.hmm.COLORS[s]}/>)}
                        </Scatter>
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* Regime statistics */}
                <div className="card">
                  <div className="lbl">REGIME STATISTICS</div>
                  <table style={{width:'100%',borderCollapse:'collapse',fontSize:'11px',marginBottom:'16px'}}>
                    <thead><tr>{['REGIME','DAYS','TIME %','ANN RET','ANN VOL','SHARPE'].map(h=>(
                      <th key={h} style={{textAlign:'left',padding:'7px 5px',color:'#2a5070',fontSize:'8px',letterSpacing:'.1em',borderBottom:'1px solid #162030'}}>{h}</th>
                    ))}</tr></thead>
                    <tbody>
                      {C.hmm.stateStats.map((s,i)=>(
                        <tr key={i} style={{borderBottom:'1px solid #0d1828'}}>
                          <td style={{padding:'9px 5px',color:s.color,fontWeight:600}}>{s.name}</td>
                          <td style={{padding:'9px 5px',color:'#7aaac0'}}>{s.count}</td>
                          <td style={{padding:'9px 5px',color:'#3a6080'}}>{s.pct}%</td>
                          <td style={{padding:'9px 5px',color:parseFloat(s.mean)>=0?'#00e8a0':'#f06040',fontWeight:600}}>{s.mean}%</td>
                          <td style={{padding:'9px 5px',color:'#f0b040'}}>{s.vol}%</td>
                          <td style={{padding:'9px 5px',color:parseFloat(s.sharpe)>=0?'#00c8f0':'#f06040'}}>{s.sharpe}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="lbl">AVG REGIME DURATION</div>
                  <div style={{display:'flex',gap:'10px',marginTop:'6px'}}>
                    {C.hmm.NAMES.map((n,i)=>(
                      <div key={n} style={{flex:1,background:'#080e18',border:`1px solid ${C.hmm.COLORS[i]}22`,padding:'12px',borderRadius:'3px',textAlign:'center'}}>
                        <div style={{fontSize:'9px',color:C.hmm.COLORS[i],letterSpacing:'.1em',marginBottom:'5px'}}>{n}</div>
                        <div style={{fontSize:'18px',color:C.hmm.COLORS[i],fontWeight:600}}>{C.hmm.avgRun[i].toFixed(1)}<span style={{fontSize:'10px'}}>d</span></div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Transition matrix */}
                <div className="card">
                  <div className="lbl">STATE TRANSITION MATRIX (%)</div>
                  <div style={{fontSize:'9px',color:'#1a3a50',marginBottom:'12px'}}>Row → Col probability of next-day regime</div>
                  <table style={{width:'100%',borderCollapse:'collapse',fontSize:'12px',marginBottom:'16px'}}>
                    <thead>
                      <tr>
                        <th style={{padding:'8px 6px',color:'#2a5070',fontSize:'9px',borderBottom:'1px solid #162030'}}>FROM \ TO</th>
                        {C.hmm.NAMES.map((n,i)=><th key={n} style={{padding:'8px 6px',color:C.hmm.COLORS[i],fontSize:'9px',letterSpacing:'.1em',borderBottom:'1px solid #162030'}}>{n}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {C.hmm.trans.map((row,i)=>(
                        <tr key={i} style={{borderBottom:'1px solid #0d1828'}}>
                          <td style={{padding:'10px 6px',color:C.hmm.COLORS[i],fontWeight:600,fontSize:'10px'}}>{row.from}</td>
                          {row.probs.map((p,j)=>(
                            <td key={j} style={{padding:'10px 6px',textAlign:'center',color:i===j?C.hmm.COLORS[i]:'#7aaac0',fontWeight:i===j?600:400,background:i===j?`${C.hmm.COLORS[i]}10`:'transparent',borderRadius:'2px'}}>{p.p}%</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{background:'#080e18',border:'1px solid #162030',padding:'10px',borderRadius:'3px',fontSize:'9px',color:'#1a3050',lineHeight:2}}>
                    <div>▸ Diagonal = persistence (staying in regime tomorrow)</div>
                    <div>▸ High diagonal → Bouchaud volatility clustering confirmed</div>
                    <div>▸ Off-diagonal = probability of regime transition</div>
                  </div>
                </div>

                {/* Emission parameters */}
                <div className="card" style={{gridColumn:'span 2'}}>
                  <div className="lbl">LEARNED HMM EMISSION PARAMETERS + TRADING SIGNAL</div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'14px',marginBottom:'14px'}}>
                    {C.hmm.NAMES.map((n,i)=>(
                      <div key={n} style={{background:'#080e18',border:`1px solid ${C.hmm.COLORS[i]}22`,padding:'16px',borderRadius:'3px'}}>
                        <div style={{fontSize:'10px',letterSpacing:'.15em',color:C.hmm.COLORS[i],marginBottom:'12px',fontWeight:600,display:'flex',alignItems:'center',gap:'8px'}}>
                          {n} {i===C.hmm.currentRegime&&<span style={{fontSize:'8px',background:`${C.hmm.COLORS[i]}22`,border:`1px solid ${C.hmm.COLORS[i]}44`,padding:'2px 6px',borderRadius:'2px'}}>CURRENT</span>}
                        </div>
                        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'8px'}}>
                          {[
                            {l:'DAILY μ',v:`${(C.hmm.means[i]*100).toFixed(4)}%`,c:C.hmm.means[i]>=0?'#00e8a0':'#f06040'},
                            {l:'ANN. μ',v:`${(C.hmm.means[i]*252*100).toFixed(2)}%`,c:C.hmm.means[i]>=0?'#00e8a0':'#f06040'},
                            {l:'DAILY σ',v:`${(C.hmm.sigmas[i]*100).toFixed(4)}%`,c:'#f0b040'},
                            {l:'ANN. σ',v:`${(C.hmm.sigmas[i]*Math.sqrt(252)*100).toFixed(2)}%`,c:'#f0b040'},
                          ].map(p=>(
                            <div key={p.l} style={{background:'#060b12',border:'1px solid #0d1828',padding:'9px',borderRadius:'2px',textAlign:'center'}}>
                              <div className="lbl">{p.l}</div>
                              <div style={{fontSize:'12px',color:p.c,fontWeight:600}}>{p.v}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{background:'#080e18',border:`1px solid ${C.hmm.currentColor}33`,padding:'14px',borderRadius:'3px',fontSize:'10px',lineHeight:2.2}}>
                    <span style={{color:C.hmm.currentColor,fontWeight:600,letterSpacing:'.08em'}}>▸ {C.hmm.currentName} REGIME SIGNAL: </span>
                    {C.hmm.currentName==='BULL'&&<span style={{color:'#7aaac0'}}>Positive drift, low volatility environment. Consider trend-following on NIFTY futures. If India VIX is elevated vs HMM regime vol, selling straddles/strangles may offer positive edge. Theta decay works in your favor.</span>}
                    {C.hmm.currentName==='SIDEWAYS'&&<span style={{color:'#7aaac0'}}>Mean-reverting, moderate volatility. Iron condors and short strangles on NIFTY weekly options are structurally favored. Watch transition probabilities — sideways regimes often precede directional moves.</span>}
                    {C.hmm.currentName==='BEAR'&&<span style={{color:'#7aaac0'}}>Negative drift, high volatility. Avoid naked short vol — tail risk is elevated. Protective puts on portfolio and long vega positions are warranted. Consider debit spreads rather than naked calls/puts.</span>}
                  </div>
                </div>
              </div>
            )}

            {/* OPTIONS */}
            {activeTab==='options'&&(
              <div style={{display:'grid',gridTemplateColumns:'300px 1fr',gap:'16px'}}>
                <div>
                  <div className="card" style={{marginBottom:'12px'}}>
                    <div className="lbl">BLACK-SCHOLES INPUTS</div>
                    {[{l:'SPOT PRICE (S) ₹',k:'S',step:50},{l:'STRIKE PRICE (K) ₹',k:'K',step:50},{l:'DAYS TO EXPIRY',k:'T',step:1},{l:'RISK-FREE RATE (%) r',k:'r',step:.1},{l:'IMPLIED VOL (%) σ',k:'sigma',step:.5}].map(f=>(
                      <div key={f.k} style={{marginBottom:'10px'}}>
                        <div className="lbl">{f.l}</div>
                        <input type="number" className="inp" value={opts[f.k]} step={f.step} onChange={e=>setOpts(p=>({...p,[f.k]:parseFloat(e.target.value)||0}))}/>
                      </div>
                    ))}
                    <div>
                      <div className="lbl">OPTION TYPE</div>
                      <select className="inp" value={opts.type} onChange={e=>setOpts(p=>({...p,type:e.target.value}))}>
                        <option value="call">CALL</option><option value="put">PUT</option>
                      </select>
                    </div>
                  </div>
                  <div className="card">
                    <div className="lbl">REGIME-ADJUSTED VOL REFERENCE</div>
                    <div style={{fontSize:'10px',lineHeight:2.4,color:'#1a3a50'}}>
                      <div>EWMA: <span style={{color:'#00c8f0'}}>{(C.currentVol*100).toFixed(1)}%</span></div>
                      <div>GARCH LR: <span style={{color:'#f0b040'}}>{(C.garch.longRunVol*100).toFixed(1)}%</span></div>
                      {C.hmm.stateStats.map((s,i)=>(
                        <div key={i} style={{color:i===C.hmm.currentRegime?s.color:'#1a3a50'}}>HMM {s.name}: <span style={{color:C.hmm.COLORS[i]}}>{s.vol}%</span>{i===C.hmm.currentRegime?' ◀':''}</div>
                      ))}
                    </div>
                  </div>
                </div>
                <div>
                  <div className="card" style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'14px'}}>
                    <div>
                      <div className="lbl">{opts.type.toUpperCase()} THEORETICAL PRICE</div>
                      <div style={{fontSize:'38px',fontWeight:600,color:'#00c8f0',textShadow:'0 0 30px rgba(0,200,240,.4)'}}>₹{greeks.price.toFixed(2)}</div>
                    </div>
                    <div style={{textAlign:'right',fontSize:'11px',lineHeight:2.2}}>
                      <div style={{color:'#2a5070'}}>Moneyness: <span style={{color:opts.S>opts.K?'#00e8a0':'#f06040'}}>{opts.S>opts.K?(opts.type==='call'?'ITM':'OTM'):(opts.type==='call'?'OTM':'ITM')}</span></div>
                      <div style={{color:'#2a5070'}}>Intrinsic: <span style={{color:'#b8ccd8'}}>₹{Math.max(0,opts.type==='call'?opts.S-opts.K:opts.K-opts.S).toFixed(2)}</span></div>
                      <div style={{color:'#2a5070'}}>Time Value: <span style={{color:'#f0b040'}}>₹{Math.max(0,greeks.price-Math.max(0,opts.type==='call'?opts.S-opts.K:opts.K-opts.S)).toFixed(2)}</span></div>
                    </div>
                  </div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:'10px',marginBottom:'14px'}}>
                    {[
                      {l:'DELTA (Δ)',v:greeks.delta.toFixed(5),d:'Directional',c:'#00c8f0'},
                      {l:'GAMMA (Γ)',v:greeks.gamma.toFixed(7),d:'Convexity',c:'#00e8a0'},
                      {l:'VEGA (ν)',v:greeks.vega.toFixed(4),d:'Per 1% IV',c:'#f0b040'},
                      {l:'THETA (Θ)',v:greeks.theta.toFixed(4),d:'Daily decay',c:'#f06040'},
                      {l:'RHO (ρ)',v:greeks.rho.toFixed(5),d:'Per 1% rate',c:'#c070f0'},
                    ].map(g=>(
                      <div key={g.l} className="gbox">
                        <div className="lbl">{g.l}</div>
                        <div style={{fontSize:'17px',fontWeight:600,color:g.c,marginBottom:'4px'}}>{g.v}</div>
                        <div style={{fontSize:'8px',color:'#1a3a50'}}>{g.d}</div>
                      </div>
                    ))}
                  </div>
                  <div className="card">
                    <div className="lbl">GREEKS IN PLAIN ENGLISH</div>
                    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px',fontSize:'11px',lineHeight:2,marginBottom:'12px'}}>
                      <div><span style={{color:'#00c8f0'}}>Δ={greeks.delta.toFixed(4)} → </span><span style={{color:'#2a5070'}}>₹100 move = <span style={{color:'#b8ccd8'}}>₹{(Math.abs(greeks.delta)*100).toFixed(2)}</span> P&L</span></div>
                      <div><span style={{color:'#f06040'}}>Θ={greeks.theta.toFixed(4)} → </span><span style={{color:'#2a5070'}}>Loses <span style={{color:'#f06040'}}>₹{Math.abs(greeks.theta).toFixed(2)}</span>/day</span></div>
                      <div><span style={{color:'#f0b040'}}>ν={greeks.vega.toFixed(4)} → </span><span style={{color:'#2a5070'}}>1% IV rise = <span style={{color:'#b8ccd8'}}>₹{greeks.vega.toFixed(2)}</span></span></div>
                      <div><span style={{color:'#00e8a0'}}>Γ={greeks.gamma.toFixed(6)} → </span><span style={{color:'#2a5070'}}>Delta change per ₹100 = <span style={{color:'#b8ccd8'}}>{(greeks.gamma*100).toFixed(5)}</span></span></div>
                    </div>
                    <div style={{padding:'12px',background:'#080e18',border:`1px solid ${C.hmm.currentColor}22`,borderRadius:'3px',fontSize:'9px',color:'#1a3050',lineHeight:2}}>
                      <span style={{color:'#2a5a70'}}>HMM IV EDGE CHECK: </span>
                      {C.hmm.currentName} regime vol = {C.hmm.stateStats[C.hmm.currentRegime]?.vol}% · Input σ = {opts.sigma}% ·
                      {opts.sigma>parseFloat(C.hmm.stateStats[C.hmm.currentRegime]?.vol||0)*1.1
                        ?<span style={{color:'#f0b040'}}> Options are RICH vs regime realized vol → selling edge exists</span>
                        :opts.sigma<parseFloat(C.hmm.stateStats[C.hmm.currentRegime]?.vol||0)*0.9
                        ?<span style={{color:'#00e8a0'}}> Options are CHEAP vs regime realized vol → buying edge exists</span>
                        :<span style={{color:'#b8ccd8'}}> IV roughly fair vs current regime vol</span>}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
