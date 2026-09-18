"""Render a Report as one self-contained HTML page.

Inline CSS with light and dark tokens, inline SVG charts and a small amount of
vanilla JavaScript for sorting, filtering, tooltips and navigation; no external
assets. Colours and mark specs follow the dataviz skill's reference palette.
"""

from __future__ import annotations

import html
import json
import math
from typing import NamedTuple

from cost_data import (
    ACCELERATORS,
    Agg,
    CPU_CLASSES,
    DayAgg,
    FAMILIES,
    HW_CLASSES,
    HW_FAMILY,
    ratio,
    Report,
    TRIGGERS,
)


SECTION_IDS = (
    "summary",
    "charts",
    "owners",
    "files",
    "hardware",
    "coverage",
    "unmapped",
    "methodology",
)
SECTION_TITLES = {
    "summary": "Summary",
    "charts": "Charts",
    "owners": "Owners",
    "files": "Files",
    "hardware": "Hardware",
    "coverage": "Coverage",
    "unmapped": "Unmapped",
    "methodology": "Methodology",
}
TRIGGER_LABELS = {
    "main": "from pushes to main",
    "pr": "from pull requests",
    "scheduled": "from scheduled runs",
    "other": "from other triggers",
}
FAMILY_CLASSES = {
    fam: [c for c in HW_CLASSES if HW_FAMILY[c] == fam] for fam in FAMILIES
}
FAMILY_FILL = {fam: f"var(--s{i})" for i, fam in enumerate(FAMILIES, 1)}
CHART_W = 960
BAR = 18
PITCH = 28
SPARK_W = 600
SPARK_H = 72
MIN_CELL_H = 0.05  # below this a table cell would print as 0.0, so leave it blank
DOT = "\u00b7"
esc = html.escape

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--dim:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--accent:#2a78d6;--accent-soft:#cde2fb;--bar:#9ec5f4;--shadow:0 1px 2px rgba(0,0,0,.04),0 12px 32px -20px rgba(0,0,0,.18);--radius:12px;--t1:#104281;--t2:#256abf;--t3:#5598e7;--t4:#86b6ef;--tt1:#ffffff;--tt2:#ffffff;--tt3:#0b0b0b;--tt4:#0b0b0b}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--dim:#52514e;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--accent:#3987e5;--accent-soft:#0d366b;--bar:#1c5cab;--shadow:none;--t1:#b7d3f6;--t2:#6da7ec;--t3:#2a78d6;--t4:#184f95;--tt1:#0b0b0b;--tt2:#0b0b0b;--tt3:#ffffff;--tt4:#ffffff}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 "Inter","SF Pro Text","Segoe UI Variable Text","Segoe UI",Roboto,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:1600px;margin:0 auto;padding:1.5rem 1.5rem 5rem}
section{scroll-margin-top:4rem}
h1{font-size:1.9rem;line-height:1.15;font-weight:700;letter-spacing:-.02em;margin:.3rem 0 .5rem}
h2{font-size:1.45rem;line-height:1.2;font-weight:700;letter-spacing:-.015em;margin:.2rem 0 .3rem}
h3{font-size:1.02rem;font-weight:650;margin:0 0 .3rem}
p{margin:.35rem 0}
.sub{color:var(--ink2)}
.eyebrow{display:inline-flex;align-items:center;gap:.55rem;font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2)}
.eyebrow::before{content:"";width:1.25rem;height:2px;border-radius:1px;background:var(--accent)}
.sec{margin:3rem 0 1rem;max-width:72ch}
.sec .lede{color:var(--ink2);margin:.1rem 0 0;font-size:1rem}
details.about{margin:.45rem 0 0;font-size:.9rem;color:var(--ink2)}
details.about summary{cursor:pointer;color:var(--accent);font-weight:600;list-style:none;width:max-content}
details.about summary::-webkit-details-marker{display:none}
details.about summary::after{content:" +"}
details.about[open] summary::after{content:" \\2212"}
details.about p{margin:.35rem 0 0}
.top{position:sticky;top:0;z-index:3;background:var(--surface);border-bottom:1px solid var(--border)}
.top nav{max-width:1600px;margin:0 auto;padding:.5rem 1.5rem;display:flex;gap:.2rem;flex-wrap:wrap;align-items:center}
.top b{margin-right:1rem;font-weight:700;letter-spacing:-.01em;display:inline-flex;align-items:center;gap:.5rem}
.top b::before{content:"";width:9px;height:9px;border-radius:50%;background:var(--accent)}
.top a{color:var(--ink2);text-decoration:none;padding:.3rem .75rem;border-radius:999px;font-size:.86rem;font-weight:500}
.top a:hover{background:var(--page);color:var(--ink)}
.top a.active{background:var(--accent-soft);color:var(--ink)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.15rem 1.4rem;margin:.9rem 0;box-shadow:var(--shadow)}
.card-head{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;margin:.2rem 0 .6rem}
.card-head h3{margin:0}
.card-head .sub{margin:.1rem 0 0;font-size:.88rem;max-width:80ch}
.warnings{border-left:4px solid var(--s4);padding:.75rem 1rem;margin-top:1rem}
.warnings ul{margin:.35rem 0 0;padding-left:1.2rem}
.mast{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(300px,.85fr);gap:2.5rem;align-items:end}
@media (max-width:900px){.mast{grid-template-columns:1fr}}
.mast .lead{font-size:1.05rem;color:var(--ink2);max-width:60ch;margin:0}
.hero{font-size:56px;font-weight:600;line-height:1;letter-spacing:-.02em}
.hero-l{color:var(--ink2);margin-top:.35rem}
.spark{display:block;width:100%;height:72px;margin-top:.9rem}
.spark .cur{display:none;pointer-events:none}
.spark.hover .cur{display:inline}
.spark-cap{display:flex;justify-content:space-between;gap:1rem;color:var(--muted);font-size:.78rem;margin-top:.15rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.75rem;margin-top:1.4rem}
.tile{background:var(--page);border:1px solid var(--border);border-radius:10px;padding:.85rem 1rem;display:flex;flex-direction:column;gap:.3rem}
.tile .l{color:var(--ink2);font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.tile .v{font-size:1.6rem;font-weight:600;letter-spacing:-.01em;line-height:1.15}
.tile .c{color:var(--muted);font-size:.8rem}
.meter{height:4px;border-radius:2px;background:var(--accent-soft);overflow:hidden;margin:.15rem 0}
.meter i{display:block;height:100%;background:var(--accent);border-radius:2px}
.tile.wide{margin-top:.75rem}
.tbar{display:flex;height:24px;border-radius:6px;overflow:hidden;background:var(--grid);margin:.15rem 0}
.tbar i{display:flex;align-items:center;min-width:0;margin-right:2px;padding:0 .55rem;font-size:.74rem;font-weight:600;font-style:normal;white-space:nowrap;overflow:hidden}
.tbar i:last-child{margin-right:0}
.tbar .mark:hover{filter:brightness(1.08)}
.tkeys{display:flex;flex-wrap:wrap;gap:.3rem 1.4rem;font-size:.82rem;color:var(--ink2);margin-top:.3rem}
.tkeys i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.4rem;vertical-align:-1px}
.tkeys b{color:var(--ink);font-weight:600;margin-left:.3rem}
.tkeys small{color:var(--muted);margin-left:.4rem}
figure{margin:0 0 1rem}
figure h3{margin:0 0 .15rem}
figure .sub{font-size:.88rem;margin:0 0 .5rem;max-width:110ch}
figure .sub b{color:var(--ink);font-weight:600}
.legend{display:flex;gap:.4rem;flex-wrap:wrap;font-size:.8rem;color:var(--ink2);margin:.3rem 0 .6rem}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:.4rem}
.legend span{display:inline-flex;align-items:center;padding:.18rem .6rem .18rem .45rem;border-radius:999px;background:var(--page);border:1px solid var(--border)}
.legend span.on{color:var(--ink);font-weight:600;border-color:var(--accent)}
.legend span.off{opacity:.45}
figure[data-sortable] .legend span{cursor:pointer}
.legend span.sel{color:var(--ink);font-weight:600;background:var(--accent-soft);border-color:var(--accent)}
svg .mark.dim{opacity:.25}
svg .band{fill:var(--page);opacity:0}
svg .band.on{opacity:1}
.tip{position:fixed;z-index:5;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:10px;padding:.5rem .7rem;font-size:.82rem;line-height:1.4;box-shadow:0 6px 24px rgba(0,0,0,.16);max-width:24rem}
.tip .tv{font-size:1rem;font-weight:600;font-variant-numeric:tabular-nums}
.tip .ts{color:var(--ink2)}
.tip .ts i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:.4rem;vertical-align:middle}
.tip .tn{color:var(--muted);font-size:.78rem}
svg text{font:12px "Inter","SF Pro Text","Segoe UI Variable Text","Segoe UI",Roboto,system-ui,sans-serif;fill:var(--ink2)}
svg .tick{fill:var(--muted);font-variant-numeric:tabular-nums}
svg .val{fill:var(--ink2);font-variant-numeric:tabular-nums}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .mark:hover,svg .mark:focus{filter:brightness(1.12);outline:none}
svg .mark:focus-visible{outline:2px solid var(--ink);outline-offset:1px}
.scroll{overflow-x:auto}
.tall{max-height:75vh;overflow:auto}
.tall thead th{position:sticky;top:0;z-index:2}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:.86rem}
th,td{padding:.4rem .55rem;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600;font-size:.78rem;letter-spacing:.02em;background:var(--surface);cursor:pointer;user-select:none}
th:hover,th:focus-visible{color:var(--ink);outline:none}
th.t,td.t{text-align:left}
th[data-dir=asc]::after{content:" \\25B4";color:var(--accent)}
th[data-dir=desc]::after{content:" \\25BE";color:var(--accent)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:.35rem;vertical-align:1px}
tbody tr:hover td{background-color:var(--page)}
tfoot td{font-weight:600;border-top:1px solid var(--axis);border-bottom:none;background-color:var(--surface)}
td.m,span.m{color:var(--muted)}
table.sticky-first th:first-child,table.sticky-first td:first-child{position:sticky;left:0;z-index:1;background-color:var(--surface);box-shadow:1px 0 0 var(--grid)}
table.sticky-first thead th:first-child{z-index:3}
table.sticky-first tbody tr:hover td:first-child{background-color:var(--page)}
td.db{background-image:linear-gradient(var(--bar),var(--bar));background-repeat:no-repeat;background-size:var(--w) 3px;background-position:0 calc(100% - 3px)}
td.mix{text-align:left;padding-left:.75rem}
.mixbar{display:inline-flex;width:96px;height:8px;border-radius:2px;overflow:hidden;vertical-align:middle;background:var(--grid)}
.mixbar i{display:block;height:100%;margin-right:1px}
.mixbar i:last-child{margin-right:0}
.mixbar .mark:hover{filter:brightness(1.12)}
.tools{display:flex;gap:.6rem;align-items:center;margin:.2rem 0 .4rem}
.tools input{font:inherit;font-size:.88rem;padding:.4rem .6rem .4rem 2rem;border:1px solid var(--axis);border-radius:999px;background:var(--surface) no-repeat .65rem center/14px 14px url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23898781' stroke-width='2.2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.6-3.6'/%3E%3C/svg%3E");color:var(--ink);min-width:17rem}
.tools input:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
.tools .n{color:var(--ink2);font-size:.78rem;font-weight:600;padding:.2rem .6rem;border-radius:999px;background:var(--page);border:1px solid var(--border)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;background:var(--page);padding:.1em .35em;border-radius:4px}
.method{padding-left:1.1rem;max-width:110ch}
.method li{margin:.5rem 0}
.method li b{color:var(--ink)}
.cmd{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem;background:var(--page);border:1px solid var(--border);border-radius:8px;padding:.6rem .8rem;white-space:pre-wrap;word-break:break-all;margin:.6rem 0 .2rem}
.bad{font-weight:600}
@media print{.top,.tools,.tip{display:none}.tall{max-height:none;overflow:visible}.card,figure{break-inside:avoid;box-shadow:none}body{background:#fff;color:#000}}
"""

JS = """
(function(){
  function val(td){var v=td.getAttribute('data-v');return v===null?td.textContent.trim().toLowerCase():parseFloat(v);}
  document.querySelectorAll('table.sortable').forEach(function(t){
    var heads=t.tHead.rows[0].cells,body=t.tBodies[0];
    Array.prototype.forEach.call(heads,function(th,i){
      function sort(){
        var asc=th.getAttribute('data-dir')==='desc';
        Array.prototype.forEach.call(heads,function(h){h.removeAttribute('data-dir');h.setAttribute('aria-sort','none');});
        th.setAttribute('data-dir',asc?'asc':'desc');th.setAttribute('aria-sort',asc?'ascending':'descending');
        var rows=Array.prototype.slice.call(body.rows);
        rows.sort(function(a,b){var x=val(a.cells[i]),y=val(b.cells[i]);
          if(typeof x==='number'&&typeof y==='number'){return asc?x-y:y-x;}
          return asc?String(x).localeCompare(String(y)):String(y).localeCompare(String(x));});
        rows.forEach(function(r){body.appendChild(r);});
      }
      th.addEventListener('click',sort);
      th.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();sort();}});
    });
  });
  document.querySelectorAll('input[data-filter]').forEach(function(inp){
    var t=document.getElementById(inp.getAttribute('data-filter'));
    var n=document.getElementById(inp.getAttribute('data-filter')+'-n');
    var rows=Array.prototype.slice.call(t.tBodies[0].rows);
    inp.addEventListener('input',function(){
      var q=inp.value.trim().toLowerCase(),k=0;
      rows.forEach(function(r){var show=!q||r.textContent.toLowerCase().indexOf(q)>=0;r.style.display=show?'':'none';if(show){k++;}});
      n.textContent=k.toLocaleString('en-US')+' of '+rows.length.toLocaleString('en-US')+' rows';
    });
  });
  var tip=document.createElement('div');tip.className='tip';tip.hidden=true;document.body.appendChild(tip);
  function line(cls,text){var d=document.createElement('div');d.className=cls;d.textContent=text;return d;}
  function clearBands(root){root.querySelectorAll('.band.on').forEach(function(b){b.classList.remove('on');});}
  function show(m,x,y){
    var series=m.getAttribute('data-series'),fig=m.closest('figure'),svg=m.closest('svg');
    tip.textContent='';
    tip.appendChild(line('tv',m.getAttribute('data-value')+' ('+m.getAttribute('data-pct')+')'));
    var who=line('ts',series?series+' on '+m.getAttribute('data-bar'):m.getAttribute('data-bar'));
    if(series){var key=document.createElement('i');key.style.background=m.getAttribute('data-fill');who.insertBefore(key,who.firstChild);}
    tip.appendChild(who);
    var note=m.getAttribute('data-note');if(note){tip.appendChild(line('tn',note));}
    tip.style.left='0px';tip.style.top='0px';tip.hidden=false;
    var r=tip.getBoundingClientRect(),px=x+14,py=y+14;
    if(px+r.width>window.innerWidth-8){px=x-r.width-14;}
    if(py+r.height>window.innerHeight-8){py=y-r.height-14;}
    tip.style.left=px+'px';tip.style.top=py+'px';
    if(fig){fig.querySelectorAll('.legend span:not([data-total])').forEach(function(el){var k=el.getAttribute('data-series');el.classList.toggle('on',k===series);el.classList.toggle('off',!!series&&k!==series);});}
    if(svg){clearBands(svg);var b=svg.querySelector('.band[data-row="'+m.getAttribute('data-row')+'"]');if(b){b.classList.add('on');}}
    if(svg&&m.hasAttribute('data-x')){var cx=m.getAttribute('data-x'),cy=m.getAttribute('data-y'),ca=m.getAttribute('data-ya');svg.classList.add('hover');var ln=svg.querySelector('.cur-line');ln.setAttribute('x1',cx);ln.setAttribute('x2',cx);svg.querySelector('.cur-attr').setAttribute('d','M'+cx+','+ca+'l0.01,0');svg.querySelector('.cur-ring').setAttribute('d','M'+cx+','+cy+'l0.01,0');svg.querySelector('.cur-wall').setAttribute('d','M'+cx+','+cy+'l0.01,0');}
  }
  function hide(el){tip.hidden=true;clearBands(el);el.classList.remove('hover');var fig=el.closest('figure');if(fig){fig.querySelectorAll('.legend span').forEach(function(s){s.classList.remove('on','off');});}}
  window.addEventListener('scroll',function(){if(!tip.hidden){tip.hidden=true;clearBands(document);document.querySelectorAll('.spark.hover').forEach(function(sp){sp.classList.remove('hover');});document.querySelectorAll('.legend span.on,.legend span.off').forEach(function(s){s.classList.remove('on','off');});}},{passive:true});
  document.querySelectorAll('figure svg,.spark,.mixbar,.tbar').forEach(function(host){
    host.addEventListener('pointermove',function(e){var m=e.target.closest('.mark');if(m){show(m,e.clientX,e.clientY);}else{hide(host);}});
    host.addEventListener('pointerleave',function(){hide(host);});
    host.addEventListener('focusin',function(e){var m=e.target.closest('.mark');if(m){var r=m.getBoundingClientRect();show(m,r.right,r.top);}});
    host.addEventListener('focusout',function(){hide(host);});
  });
  function applyDim(fig,key){fig.querySelectorAll('.mark').forEach(function(m){m.classList.toggle('dim',!!key&&m.getAttribute('data-series')!==key);});}
  document.querySelectorAll('figure .legend span[data-series]').forEach(function(el){
    var fig=el.closest('figure'),key=el.getAttribute('data-series');
    el.addEventListener('pointerenter',function(){if(!key){return;}el.classList.add('on');applyDim(fig,key);});
    el.addEventListener('pointerleave',function(){el.classList.remove('on');applyDim(fig,fig.getAttribute('data-sort')||'');});
  });
  var PITCH=28,BAR=18,W=960,NS='http://www.w3.org/2000/svg';
  function fmtK(v){if(v>=999500){return (v/1e6).toFixed(2)+'M';}if(v>=9995){return Math.round(v/1e3)+'K';}if(v>=999.5){return (v/1e3).toFixed(1)+'K';}if(v>=10||v===0){return Math.round(v).toString();}return v.toFixed(1);}
  function fmtH(v){return v>=100?Math.round(v).toLocaleString('en-US'):v.toFixed(1);}
  function fmtPct(p){return p<0.1?(p*100).toFixed(1)+'%':Math.round(p*100)+'%';}
  function barPath(x,y,w,h,r){if(r<=0||w<2*r){return 'M'+x.toFixed(1)+','+y.toFixed(1)+'h'+w.toFixed(1)+'v'+h.toFixed(1)+'h'+(-w).toFixed(1)+'z';}return 'M'+x.toFixed(1)+','+y.toFixed(1)+'h'+(w-r).toFixed(1)+'a'+r+','+r+' 0 0 1 '+r+','+r+'v'+(h-2*r).toFixed(1)+'a'+r+','+r+' 0 0 1 '+(-r)+','+r+'h'+(-(w-r)).toFixed(1)+'z';}
  function node(svg,tag,attrs,text){var e=document.createElementNS(NS,tag);Object.keys(attrs).forEach(function(k){e.setAttribute(k,attrs[k]);});if(text!==undefined){e.textContent=text;}svg.appendChild(e);return e;}
  function drawChart(fig,key){
    var data=JSON.parse(fig.querySelector('script.chart-data').textContent),svg=fig.querySelector('svg'),ki=-1;
    data.series.forEach(function(s,i){if(s.key===key){ki=i;}});
    var rows=data.rows.map(function(r){var total=r.v.reduce(function(a,b){return a+b;},0);return {r:r,total:total,sel:ki>=0?r.v[ki]:total};});
    if(ki>=0){rows=rows.filter(function(x){return x.sel>0;});}
    rows.sort(function(a,b){return b.sel-a.sel||a.r.label.localeCompare(b.r.label);});
    rows=rows.slice(0,data.limit);
    var n=rows.length,g=data.gutter,scale=(W-g-120)/data.top,baseY=8+n*PITCH,lim=Math.floor(g/7);
    svg.setAttribute('viewBox','0 0 '+W+' '+(baseY+26));
    while(svg.firstChild){svg.removeChild(svg.firstChild);}
    rows.forEach(function(x,i){node(svg,'rect',{class:'band','data-row':i,x:0,y:8+i*PITCH,width:W,height:PITCH,rx:6});});
    for(var t=0;t<=Math.round(data.top/data.step);t++){var tx=(g+t*data.step*scale).toFixed(1);node(svg,'line',{class:'grid',x1:tx,y1:4,x2:tx,y2:baseY});node(svg,'text',{class:'tick',x:tx,y:baseY+16,'text-anchor':'middle'},fmtK(t*data.step));}
    node(svg,'line',{class:'axis',x1:g,y1:4,x2:g,y2:baseY});
    var order=data.series.map(function(_,i){return i;});
    if(ki>=0){order=[ki].concat(order.filter(function(i){return i!==ki;}));}
    rows.forEach(function(x,i){
      var y=8+i*PITCH+(PITCH-BAR)/2,ty=(y+BAR/2+4).toFixed(1),label=x.r.label;
      node(svg,'text',{x:g-8,y:ty,'text-anchor':'end'},label.length>lim?label.slice(0,lim-3)+'...':label);
      var segs=order.filter(function(si){return x.r.v[si]*scale>=1;});
      if(!segs.length&&x.total>0){var best=order[0];order.forEach(function(si){if(x.r.v[si]>x.r.v[best]){best=si;}});segs=[best];}
      var px=g;
      segs.forEach(function(si,k){
        var v=x.r.v[si],w=Math.max(v*scale,1),last=k===segs.length-1,s=data.series[si];
        var pct=fmtPct(x.total?v/x.total:0)+' of '+label,value=fmtH(v)+' '+data.unit,note=(x.r.n&&x.r.n[si])||'';
        node(svg,'path',{class:'mark'+(ki>=0&&si!==ki?' dim':''),d:barPath(px,y,last?w:Math.max(w-2,0.5),BAR,last?4:0),fill:s.fill,stroke:'transparent','stroke-width':8,tabindex:0,role:'img','aria-label':label+' / '+s.key+': '+value+' ('+pct+(note?'; '+note:'')+')','data-series':s.key,'data-bar':label,'data-value':value,'data-pct':pct,'data-note':note,'data-fill':s.fill,'data-row':i});
        px+=w;
      });
      var val=node(svg,'text',{class:'val',x:(g+x.total*scale+6).toFixed(1),y:ty},(ki>=0?fmtK(x.sel):fmtK(x.total))+' '+data.unit);
      if(ki>=0){var of=document.createElementNS(NS,'tspan');of.setAttribute('class','tick');of.textContent=' of '+fmtK(x.total);val.appendChild(of);}
    });
    fig.setAttribute('data-sort',key);
    fig.querySelectorAll('.legend span[data-series]').forEach(function(p){p.classList.toggle('sel',p.getAttribute('data-series')===key);});
  }
  document.querySelectorAll('figure[data-sortable]').forEach(function(fig){
    fig.querySelectorAll('.legend span[data-series]').forEach(function(p){
      p.addEventListener('click',function(){var k=p.getAttribute('data-series');drawChart(fig,fig.getAttribute('data-sort')===k?'':k);});
    });
  });
  var links={};
  document.querySelectorAll('.top nav a[href^="#"]').forEach(function(a){links[a.getAttribute('href').slice(1)]=a;});
  if('IntersectionObserver' in window){
    var obs=new IntersectionObserver(function(entries){entries.forEach(function(en){if(en.isIntersecting){Object.keys(links).forEach(function(k){links[k].classList.toggle('active',k===en.target.id);});}});},{rootMargin:'-40% 0px -55% 0px',threshold:0});
    document.querySelectorAll('main section[id]').forEach(function(s){obs.observe(s);});
  }
})();
"""


def fmt_k(value: float) -> str:
    if value >= 999_500:
        return f"{value / 1e6:.2f}M"
    if value >= 9_995:
        return f"{value / 1e3:.0f}K"
    if value >= 999.5:
        return f"{value / 1e3:.1f}K"
    if value >= 10 or value == 0:
        return f"{value:.0f}"
    return f"{value:.1f}"


def fmt_h(hours: float) -> str:
    return f"{hours:,.0f}" if hours >= 100 else f"{hours:,.1f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1%}" if value < 0.1 else f"{value:.0%}"


def nice_step(max_value: float, target: int = 4) -> float:
    if max_value <= 0:
        return 1.0
    raw = max_value / target
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * magnitude >= raw:
            return m * magnitude
    return 10 * magnitude


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def bar_path(x: float, y: float, w: float, h: float, r: float) -> str:
    if r <= 0 or w < 2 * r:
        return f"M{x:.1f},{y:.1f}h{w:.1f}v{h:.1f}h{-w:.1f}z"
    return f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z"


class Seg(NamedTuple):
    fill: str
    value: float
    series: str = ""  # legend key; empty for a single-series chart
    note: str = ""  # extra detail shown in the tooltip


def mark_attrs(bar: str, seg: Seg, value: str, pct: str, row: int | None = None) -> str:
    """The tooltip contract shared by chart marks, sparkline days and mix segments."""
    who = f"{bar} / {seg.series}" if seg.series else bar
    detail = f"{pct}; {seg.note}" if seg.note else pct
    label = f"{who}: {value} ({detail})"
    attrs = (
        f'aria-label="{esc(label)}" data-series="{esc(seg.series)}" '
        f'data-bar="{esc(bar)}" data-value="{esc(value)}" data-pct="{esc(pct)}" '
        f'data-note="{esc(seg.note)}" data-fill="{seg.fill}"'
    )
    return attrs if row is None else f'{attrs} data-row="{row}"'


def chart_scale(rows: list[tuple[str, list[Seg], str]]) -> tuple[float, float]:
    max_v = max((sum(seg.value for seg in segs) for _, segs, _ in rows), default=0.0)
    step = nice_step(max_v)
    return step, step * math.ceil(max_v / step) if max_v > 0 else step


def hbar_svg(
    rows: list[tuple[str, list[Seg], str]],
    gutter: int,
    aria: str,
    unit: str = "h",
    share: tuple[str, float] | None = None,
) -> str:
    """Horizontal bars. rows = (label, segments, value-label suffix html). Each mark
    carries its tooltip as data attributes: the share is of the bar, or of the
    (name, total) in `share` for single-series charts."""
    n = len(rows)
    plot_w = CHART_W - gutter - 120
    step, top = chart_scale(rows)
    scale = plot_w / top
    base_y = 8 + n * PITCH
    out = [
        f'<svg viewBox="0 0 {CHART_W} {base_y + 26}" width="100%" role="img" aria-label="{esc(aria)}">'
    ]
    for r in range(n):
        out.append(
            f'<rect class="band" data-row="{r}" x="0" y="{8 + r * PITCH}" width="{CHART_W}" height="{PITCH}" rx="6"/>'
        )
    for i in range(int(round(top / step)) + 1):
        x = gutter + i * step * scale
        out.append(
            f'<line class="grid" x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{base_y}"/>'
        )
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{base_y + 16}" text-anchor="middle">{fmt_k(i * step)}</text>'
        )
    out.append(f'<line class="axis" x1="{gutter}" y1="4" x2="{gutter}" y2="{base_y}"/>')
    for r, (label, segs, suffix) in enumerate(rows):
        y = 8 + r * PITCH + (PITCH - BAR) / 2
        text_y = y + BAR / 2 + 4
        out.append(
            f'<text x="{gutter - 8}" y="{text_y:.1f}" text-anchor="end">{esc(clip(label, gutter // 7))}</text>'
        )
        total = sum(seg.value for seg in segs)
        visible = [seg for seg in segs if seg.value * scale >= 1]
        if not visible and total > 0:
            visible = [max(segs, key=lambda seg: seg.value)]  # 1px sliver as hit target
        x = float(gutter)
        for k, seg in enumerate(visible):
            w = max(seg.value * scale, 1.0)
            last = k == len(visible) - 1
            d = bar_path(x, y, w if last else max(w - 2, 0.5), BAR, 4 if last else 0)
            name, whole = share if share else (label, total)
            pct = f"{fmt_pct(ratio(seg.value, whole))} of {name}"
            attrs = mark_attrs(label, seg, f"{fmt_h(seg.value)} {unit}", pct, r)
            out.append(
                f'<path class="mark" d="{d}" fill="{seg.fill}" stroke="transparent" stroke-width="8" tabindex="0" role="img" {attrs}/>'
            )
            x += w
        out.append(
            f'<text class="val" x="{gutter + total * scale + 6:.1f}" y="{text_y:.1f}">{fmt_k(total)} {unit}{suffix}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def sparkline_svg(days: list[DayAgg], total: float) -> str:
    """Daily test-job hours (accent line and wash) over attributed hours (dim line),
    with one invisible hit rect per day carrying the tooltip contract."""
    n = len(days)
    pad = 8
    top = max((d.wall_h for d in days), default=0.0) or 1.0
    step = (SPARK_W - 2 * pad) / max(n - 1, 1)
    base = SPARK_H - pad

    def point(i: int, hours: float) -> tuple[float, float]:
        return pad + i * step, base - ratio(hours, top) * (SPARK_H - 2 * pad)

    wall = [point(i, d.wall_h) for i, d in enumerate(days)]
    attr = [point(i, d.attr_h) for i, d in enumerate(days)]
    wall_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in wall)
    attr_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in attr)
    area = f"M{wall[0][0]:.1f},{base} L{wall_pts.replace(' ', ' L')} L{wall[-1][0]:.1f},{base} Z"
    out = [
        f'<svg class="spark" viewBox="0 0 {SPARK_W} {SPARK_H}" preserveAspectRatio="none" role="img" aria-label="Test-job hours per UTC day">',
        f'<path d="{area}" fill="var(--accent-soft)" opacity=".45"/>',
        f'<polyline points="{attr_pts}" fill="none" stroke="var(--dim)" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        f'<polyline points="{wall_pts}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        f'<path d="M{wall[-1][0]:.1f},{wall[-1][1]:.1f}l0.01,0" stroke="var(--surface)" stroke-width="13" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        f'<path d="M{wall[-1][0]:.1f},{wall[-1][1]:.1f}l0.01,0" stroke="var(--accent)" stroke-width="9" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        f'<line class="cur cur-line" x1="0" y1="0" x2="0" y2="{SPARK_H}" stroke="var(--axis)" stroke-width="1" vector-effect="non-scaling-stroke"/>',
        '<path class="cur cur-attr" d="M0,0" stroke="var(--dim)" stroke-width="7" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        '<path class="cur cur-ring" d="M0,0" stroke="var(--surface)" stroke-width="13" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        '<path class="cur cur-wall" d="M0,0" stroke="var(--accent)" stroke-width="9" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
    ]
    for i, d in enumerate(days):
        seg = Seg(
            "var(--accent)",
            d.wall_h,
            "",
            f"{fmt_h(d.attr_h)} h attributed ({ratio(d.attr_h, d.wall_h):.0%}); {'settled' if d.settled else 'unsettled'}",
        )
        pct = f"{fmt_pct(ratio(d.wall_h, total))} of the window"
        attrs = mark_attrs(str(d.day), seg, f"{fmt_h(d.wall_h)} h", pct)
        point = f'data-x="{wall[i][0]:.1f}" data-y="{wall[i][1]:.1f}" data-ya="{attr[i][1]:.1f}"'
        out.append(
            f'<rect class="mark" x="{pad + (i - 0.5) * step:.1f}" y="0" width="{step:.1f}" height="{SPARK_H}" fill="transparent" tabindex="0" role="img" {attrs} {point}/>'
        )
    out.append("</svg>")
    return "\n".join(out)


def legend(items: list[tuple[str, str, str]]) -> str:
    """items = (fill, label, series key matching the marks' data-series); an empty fill
    renders the "sort by total" pill of a sortable chart."""
    spans = []
    for fill, label, key in items:
        if fill:
            spans.append(
                f'<span data-series="{esc(key)}"><i style="background:{fill}"></i>{esc(label)}</span>'
            )
        else:
            spans.append(
                f'<span data-series="" data-total="1" class="sel">{esc(label)}</span>'
            )
    return '<div class="legend">' + "".join(spans) + "</div>"


def figure(title: str, headline: str, detail: str, body: str, attrs: str = "") -> str:
    return f'<figure class="card"{attrs}><h3>{esc(title)}</h3><p class="sub"><b>{esc(headline)}.</b> {esc(detail)}</p>{body}</figure>'


def sortable_figure(
    title: str,
    headline: str,
    detail: str,
    series: list[tuple[str, str]],
    legend_items: list[tuple[str, str, str]],
    rows: list[tuple[str, list[Seg], str]],
    gutter: int,
    aria: str,
    unit: str,
    limit: int,
) -> str:
    """Stacked bars whose legend pills re-sort and re-select the rows client-side. The
    server renders the default order (by total); every row is embedded as JSON so the
    script can pick the top rows for any series. series = (key, fill) in segment order."""
    step, top = chart_scale(rows[:limit])
    data_rows = []
    for label, segs, _ in rows:
        row: dict[str, object] = {
            "label": label,
            "v": [round(seg.value, 3) for seg in segs],
        }
        if any(seg.note for seg in segs):
            row["n"] = [seg.note for seg in segs]
        data_rows.append(row)
    data = {
        "unit": unit,
        "gutter": gutter,
        "top": top,
        "step": step,
        "limit": limit,
        "series": [{"key": key, "fill": fill} for key, fill in series],
        "rows": data_rows,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = (
        legend([("", "Total", ""), *legend_items])
        + hbar_svg(rows[:limit], gutter, aria, unit)
        + f'<script type="application/json" class="chart-data">{payload}</script>'
    )
    hint = f"{detail}; click a legend entry to sort by that series"
    return figure(title, headline, hint, body, ' data-sortable="1" data-sort=""')


def section_head(sid: str, lede: str, about: str = "") -> str:
    """Numbered eyebrow, title and one-sentence lede; longer notes fold into a details block."""
    eyebrow = f"{SECTION_IDS.index(sid):02d} {SECTION_TITLES[sid]}"
    more = (
        f'<details class="about"><summary>About these numbers</summary><p>{about}</p></details>'
        if about
        else ""
    )
    return f'<header class="sec"><span class="eyebrow">{esc(eyebrow)}</span><h2>{SECTION_TITLES[sid]}</h2><p class="lede">{lede}</p>{more}</header>'


def td(
    text: str,
    value: float | None = None,
    cls: str = "",
    title: str = "",
    style: str = "",
) -> str:
    attrs = f' class="{cls}"' if cls else ""
    if value is not None:
        attrs += f' data-v="{value:.4f}"'
    if title:
        attrs += f' title="{esc(title)}"'
    if style:
        attrs += f' style="{style}"'
    return f"<td{attrs}>{text}</td>"


def num(value: float, title: str = "", bar: float | None = None) -> str:
    if bar is None:
        return td(fmt_h(value), value, title=title)
    width = f"--w:{min(max(bar, 0.0), 1.0):.1%}"
    return td(fmt_h(value), value, "db", title, width)


def count(value: int) -> str:
    return td(f"{value:,}", value)


def pct(value: float) -> str:
    return td(f"{value:.1%}", value)


def text(value: str, cls: str = "t") -> str:
    return td(esc(value), None, cls)


def table(
    tid: str,
    headers: list[tuple],
    rows: list[list[str]],
    foot: list[str] | None = None,
    filterable: bool = False,
    sticky: bool = False,
    title: str = "",
    note: str = "",
) -> str:
    """headers = (label, is_text[, prefix html[, hover hint]]); note may hold markup."""
    tools = ""
    if filterable:
        tools = f'<div class="tools"><input id="{tid}-filter" name="{tid}-filter" data-filter="{tid}" placeholder="Filter rows" aria-label="Filter rows"><span class="n" id="{tid}-n">{len(rows):,} of {len(rows):,} rows</span></div>'
    if title:
        sub = f'<p class="sub">{note}</p>' if note else ""
        head = (
            f'<div class="card-head"><div><h3>{esc(title)}</h3>{sub}</div>{tools}</div>'
        )
    else:
        head = tools
    heads = []
    for h in headers:
        cls = ' class="t"' if h[1] else ""
        prefix = h[2] if len(h) > 2 else ""
        hint = f' title="{esc(h[3])}"' if len(h) > 3 else ""
        heads.append(
            f'<th{cls} tabindex="0" aria-sort="none"{hint}>{prefix}{esc(h[0])}</th>'
        )
    wrap = "scroll tall" if filterable else "scroll"
    kind = "sortable sticky-first" if sticky else "sortable"
    out = [
        head,
        f'<div class="{wrap}"><table id="{tid}" class="{kind}"><thead><tr>{"".join(heads)}</tr></thead><tbody>',
    ]
    out.extend("<tr>" + "".join(cells) + "</tr>\n" for cells in rows)
    out.append("</tbody>")
    if foot:
        out.append("<tfoot><tr>" + "".join(foot) + "</tr></tfoot>")
    out.append("</table></div>")
    return "".join(out)


def class_columns(report: Report) -> list[str]:
    return [c for c in HW_CLASSES if report.class_attr_h[c] > 0]


def class_headers(classes: list[str]) -> list[tuple]:
    return [
        (
            c,
            False,
            f'<i class="dot" style="background:{FAMILY_FILL[HW_FAMILY[c]]}"></i>',
        )
        for c in classes
    ]


def subclass_split(agg: Agg, hw_class: str, report: Report) -> str:
    """Tooltip for an accelerator cell: the cell's hours per sub-class."""
    if hw_class not in ACCELERATORS:
        return ""
    return "; ".join(
        f"{s.hw.subclass} {fmt_h(agg.by_subclass[s.hw.subclass])} h"
        for s in report.subclasses
        if s.hw.cls == hw_class
        and agg.by_subclass.get(s.hw.subclass, 0.0) >= MIN_CELL_H
    )


def mix_cell(agg: Agg, rich: bool) -> str:
    """Hardware-family mix of one row as a mini stacked bar; sorts by accelerator share.
    Rich segments carry the tooltip contract; compact ones (the long files table) only a title."""
    total = agg.hours
    parts = [
        (fam, hours)
        for fam in FAMILIES
        if (hours := sum(agg.by_class[c] for c in FAMILY_CLASSES[fam])) >= 0.01 * total
        and hours > 0
    ]
    label = ", ".join(f"{fam} {ratio(h, total):.0%}" for fam, h in parts)
    segs = []
    for fam, hours in parts:
        style = f"width:{ratio(hours, total):.1%};background:{FAMILY_FILL[fam]}"
        if rich:
            pct = f"{fmt_pct(ratio(hours, total))} of {agg.key}"
            attrs = mark_attrs(
                agg.key, Seg(FAMILY_FILL[fam], hours, fam), f"{fmt_h(hours)} h", pct
            )
            segs.append(f'<i class="mark" style="{style}" {attrs}></i>')
        else:
            segs.append(f'<i style="{style}"></i>')
    name = f'aria-label="{esc(label)}"' if rich else f'title="{esc(label)}"'
    span = f'<span class="mixbar" role="img" {name}>{"".join(segs)}</span>'
    accel = sum(agg.by_class[c] for c in ACCELERATORS)
    return td(span, ratio(accel, total), "mix")


def agg_row(
    agg: Agg,
    report: Report,
    classes: list[str],
    label_cell: str,
    with_jobs: bool,
    top_hours: float,
) -> list[str]:
    hours = agg.hours
    row = (
        [label_cell, text(agg.owner)]
        if with_jobs
        else [
            label_cell,
            count(len(agg.members)) if agg.key != "unmapped" else td("", 0.0, "m"),
        ]
    )
    row += [
        num(hours, bar=ratio(hours, top_hours)),
        mix_cell(agg, rich=not with_jobs),
        pct(ratio(hours, report.attr_h)),
    ]
    if with_jobs:
        row.append(count(agg.jobs))
    row.append(pct(ratio(agg.by_trigger["pr"], hours)))
    row += [
        num(agg.by_class[c], subclass_split(agg, c, report))
        if agg.by_class[c] >= MIN_CELL_H
        else td("", 0.0, "m")
        for c in classes
    ]
    row.append(num(agg.test_h))
    return row


MIX_HEADER = (
    "Mix",
    False,
    "",
    "Hardware family mix of this row's hours, in the owner chart's colours; sorts by accelerator share",
)


def tile_grid(tiles: list[tuple[str, str, str, float | None]], extra: str = "") -> str:
    cells = []
    for value, label, cap, share in tiles:
        meter = (
            f'<div class="meter"><i style="width:{min(max(share, 0.0), 1.0):.1%}"></i></div>'
            if share is not None
            else ""
        )
        cells.append(
            f'<div class="tile"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>{meter}<div class="c">{esc(cap)}</div></div>'
        )
    return f'<div class="tiles">{"".join(cells)}</div>{extra}'


def trigger_tile(report: Report) -> str:
    """One wide tile: a joined bar of test-job hours by trigger, a key per segment."""
    total = report.wall_h
    segs, keys, parts = [], [], []
    for i, t in enumerate(TRIGGERS, 1):
        hours = report.trigger_wall_h[t]
        if hours <= 0:
            continue
        share = ratio(hours, total)
        name = TRIGGER_LABELS[t].removeprefix("from ")
        note = f"{report.trigger_attr_h[t]:,.0f} h attributed"
        seg = Seg(f"var(--t{i})", hours, "", note)
        attrs = mark_attrs(
            TRIGGER_LABELS[t], seg, f"{hours:,.0f} h", f"{share:.1%} of test-job hours"
        )
        inner = f"<b>{esc(name)} {share:.0%}</b>" if share >= 0.14 else ""
        segs.append(
            f'<i class="mark" style="flex:{share:.4f} 1 0%;background:var(--t{i});color:var(--tt{i})" {attrs}>{inner}</i>'
        )
        keys.append(
            f'<span><i style="background:var(--t{i})"></i>{esc(name)}<b>{hours:,.0f} h</b> {share:.1%}<small>{note}</small></span>'
        )
        parts.append(f"{name} {share:.1%}")
    return (
        '<div class="tile wide"><div class="l">test-job hours by trigger</div>'
        f'<div class="tbar" role="img" aria-label="{esc(", ".join(parts))}">{"".join(segs)}</div>'
        f'<div class="tkeys">{"".join(keys)}</div></div>'
    )


def summary_section(report: Report) -> str:
    m = report.meta
    w = m.window
    unsettled = sum(1 for d in report.days if not d.settled)
    owners = [o for o in report.owners if o.key not in ("unmapped", "no-header")]
    tiles: list[tuple[str, str, str, float | None]] = [
        (
            f"{report.attr_h:,.0f} h",
            "attributed to test files",
            f"{report.coverage:.1%} of test-job hours; non-successful jobs and missing results remain unattributed",
            report.coverage,
        ),
        (
            f"{report.total_jobs:,}",
            "test jobs",
            "completed test / test-osdc jobs, reruns counted separately",
            None,
        ),
        (
            f"{report.accelerator_share:.1%}",
            "accelerator share",
            "of test-job hours on NVIDIA, ROCm, XPU or TPU runners",
            report.accelerator_share,
        ),
        (
            f"{report.gpu_h:,.0f}",
            "accelerator GPU-hours",
            "test-job hours x GPUs per runner on accelerator runners",
            None,
        ),
        (
            f"{len(report.files):,}",
            "test files with attributed hours",
            f"{len(owners):,} owners",
            None,
        ),
        (
            f"{len(report.days)}",
            "UTC days",
            f"{unsettled} unsettled, refetched on every run"
            if unsettled
            else "all days settled",
            None,
        ),
    ]
    daily = [d.wall_h for d in report.days]
    eyebrow = f" {DOT} ".join(
        [
            "pytorch/pytorch CI",
            f"{len(report.days)} UTC days",
            f"{w.start} to {w.last}",
        ]
    )
    spark_cap = (
        f"low {fmt_h(min(daily))} h {DOT} high {fmt_h(max(daily))} h" if daily else ""
    )
    warnings = ""
    if report.warnings:
        items = "".join(f"<li>{esc(warning)}</li>" for warning in report.warnings)
        warnings = (
            '<aside class="warnings" aria-labelledby="data-quality-title">'
            f'<h3 id="data-quality-title">Data quality warnings</h3><ul>{items}</ul></aside>'
        )
    return (
        '<section id="summary"><div class="card"><div class="mast"><div>'
        f'<span class="eyebrow">{esc(eyebrow)}</span><h1>CI test cost</h1>'
        f'<p class="lead">Machine-hours of pytorch/pytorch CI test jobs, attributed to test files and their owners. Owners come from the checkout at {esc(m.checkout)}. Hours are counted per hardware class and are not price-weighted.</p>'
        "</div><div>"
        f'<div class="hero">{report.wall_h:,.0f}</div><div class="hero-l">machine-hours spent in test jobs</div>'
        + sparkline_svg(report.days, report.wall_h)
        + f'<div class="spark-cap"><span>test-job hours per UTC day; attributed hours in gray</span><span>{esc(spark_cap)}</span></div>'
        "</div></div>"
        + warnings
        + tile_grid(tiles, trigger_tile(report))
        + "</div></section>"
    )


def owner_stack(parts: list[Seg], attr: float, wall: float) -> list[Seg]:
    """Segments for one bar: the named owners, then every other owner and the unattributed remainder."""
    other = Seg("var(--muted)", attr - sum(seg.value for seg in parts), "other owners")
    gap = max(wall - attr, 0.0)
    note = "non-successful jobs or missing per-test results"
    return [*parts, other, Seg("var(--dim)", gap, "unattributed", note)]


def owner_series(named: list[Agg], slots: dict[str, str]) -> list[tuple[str, str]]:
    """(key, fill) per segment of an owner-stacked bar, matching owner_stack's order."""
    return [(o.key, slots[o.key]) for o in named] + [
        ("other owners", "var(--muted)"),
        ("unattributed", "var(--dim)"),
    ]


def charts_section(report: Report) -> str:
    head = section_head(
        "charts",
        "Four views of the same hours: who uses them, on which CPU classes, on which GPU pools, and in which files. Hover a segment for the exact value and its share; click a legend entry to sort a chart by that series.",
    )
    if not report.owners:
        return f'<section id="charts">{head}<p class="sub">No attributed test results in this window.</p></section>'
    top_owner = report.owners[0]
    owner_rows = []
    for owner in report.owners:
        segs = []
        for fam in FAMILIES:
            hours = sum(owner.by_class[c] for c in FAMILY_CLASSES[fam])
            segs.append(Seg(FAMILY_FILL[fam], hours, fam))
        owner_rows.append((owner.key, segs, ""))
    shown = {
        seg.series for _, segs, _ in owner_rows[:15] for seg in segs if seg.value > 0
    }
    legend_items = [
        (
            FAMILY_FILL[fam],
            f"{fam} ({', '.join(FAMILY_CLASSES[fam])})"
            if len(FAMILY_CLASSES[fam]) > 1
            else fam,
            fam,
        )
        for fam in FAMILIES
        if fam in shown
    ]
    owners_fig = sortable_figure(
        "Attributed test-job hours by owner",
        f"{top_owner.key} tests use {ratio(top_owner.hours, report.attr_h):.0%} of attributed test-job hours",
        f"Attributed machine-hours by owner and hardware family, top 15 of {len(report.owners)} owners; full table below",
        [(fam, FAMILY_FILL[fam]) for fam in FAMILIES],
        legend_items,
        owner_rows,
        180,
        "Attributed hours per owner, stacked by hardware family",
        "h",
        15,
    )
    top_file = report.files[0] if report.files else None
    file_rows = [
        (
            f.key.removeprefix("test/"),
            [Seg("var(--s1)", f.hours, "", f"owner {f.owner}")],
            f'<tspan class="tick"> {esc(f.owner)}</tspan>',
        )
        for f in report.files[:20]
    ]
    files_fig = figure(
        "Attributed test-job hours by test file",
        f"{top_file.key} alone uses {fmt_k(top_file.hours)} h ({ratio(top_file.hours, report.attr_h):.1%} of attributed hours)"
        if top_file
        else "No file could be mapped",
        f"Attributed machine-hours by test file with its owner, top {len(file_rows)} of {len(report.files)} files",
        hbar_svg(
            file_rows,
            370,
            "Attributed hours per test file",
            share=("attributed hours", report.attr_h),
        ),
    )
    named = [o for o in report.owners if o.key not in ("unmapped", "no-header")][:6]
    slots = {o.key: f"var(--s{i})" for i, o in enumerate(named, 1)}
    cpu_classes = sorted(
        (c for c in CPU_CLASSES if report.class_wall_h[c] > 0),
        key=lambda c: (-report.class_wall_h[c], HW_CLASSES.index(c)),
    )
    cpu_rows = []
    for c in cpu_classes:
        wall, attr = report.class_wall_h[c], report.class_attr_h[c]
        parts = [Seg(slots[o.key], o.by_class[c], o.key) for o in named]
        cpu_rows.append((c, owner_stack(parts, attr, wall), ""))
    cpu_fig = (
        sortable_figure(
            "Test-job hours by CPU class and owner",
            f"{cpu_classes[0]} runners account for {ratio(report.class_wall_h[cpu_classes[0]], report.wall_h):.0%} of all test-job hours",
            f"Test-job machine-hours on CPU-only runner classes, stacked by owner: the {len(named)} owners with the most attributed hours, every other owner, and hours no test file reported; accelerator classes are in the next chart and an unknown class, if any, only in the Hardware tables",
            owner_series(named, slots),
            [(fill, key, key) for key, fill in owner_series(named, slots)],
            cpu_rows,
            110,
            "Test-job hours per CPU class by owner",
            "h",
            len(cpu_rows),
        )
        if cpu_classes
        else ""
    )
    accel = sorted(
        (s for s in report.subclasses if s.hw.cls in ACCELERATORS and s.wall_h > 0),
        key=lambda s: (-s.wall_h * s.hw.gpus, s.hw.subclass),
    )
    gpu_subs = [s.hw.subclass for s in accel]
    gpu_named = [o for o in named if any(o.by_subclass.get(k, 0) > 0 for k in gpu_subs)]
    gpu_rows = []
    for s in accel:
        sub, gpus = s.hw.subclass, s.hw.gpus
        parts = [
            Seg(
                slots[o.key],
                o.by_subclass.get(sub, 0.0) * gpus,
                o.key,
                f"{fmt_h(o.by_subclass.get(sub, 0.0))} runner h x {gpus} GPUs",
            )
            for o in gpu_named
        ]
        stack = owner_stack(parts, s.attr_h * gpus, s.wall_h * gpus)
        gpu_rows.append((sub, stack, ""))
    gpu_fig = (
        sortable_figure(
            "GPU-hours by accelerator sub-class and owner",
            f"{accel[0].hw.subclass} runners use {ratio(accel[0].wall_h * accel[0].hw.gpus, report.gpu_h):.0%} of accelerator GPU-hours",
            f"GPU-hours by accelerator sub-class (runner hours x GPUs per runner), stacked by owner: the {len(gpu_named)} owners with the most attributed hours, every other owner, and hours no test file reported; top {min(20, len(accel))} of {len(accel)} sub-classes",
            owner_series(gpu_named, slots),
            [(fill, key, key) for key, fill in owner_series(gpu_named, slots)],
            gpu_rows,
            120,
            "GPU-hours per accelerator sub-class by owner",
            "GPU-h",
            20,
        )
        if accel
        else ""
    )
    return f'<section id="charts">{head}{owners_fig}{cpu_fig}{gpu_fig}{files_fig}</section>'


def owners_section(report: Report) -> str:
    classes = class_columns(report)
    headers = (
        [
            ("Owner", True),
            ("Files", False),
            ("Hours", False),
            MIX_HEADER,
            ("Share", False),
            ("PR share", False),
        ]
        + class_headers(classes)
        + [("Raw test h", False)]
    )
    top = report.owners[0].hours if report.owners else 0.0
    rows = [agg_row(o, report, classes, text(o.key), False, top) for o in report.owners]
    foot = [
        td("Total", None, "t"),
        count(len(report.files)),
        num(report.attr_h),
        td("", None, "m"),
        pct(1.0 if report.attr_h else 0.0),
        pct(ratio(report.trigger_attr_h["pr"], report.attr_h)),
    ]
    foot += [num(report.class_attr_h[c]) for c in classes] + [
        num(sum(o.test_h for o in report.owners))
    ]
    about = (
        "The owner is the first <code># Owner(s)</code> label of each file with the module: / oncall: prefix removed. "
        "<code>unknown</code> is the literal <code>module: unknown</code> label; <code>no-header</code> files have no header; <code>unmapped</code> invoking names have no file in this checkout. "
        "The Mix bar shows the hardware-family split of the row in the owner chart's colours and sorts by accelerator share. Click a header or press Enter on it to sort; type to filter."
    )
    return (
        '<section id="owners">'
        + section_head(
            "owners",
            "Attributed machine-hours per owner, with the hardware mix and the hours on each class.",
            about,
        )
        + '<div class="card">'
        + table(
            "owners-table",
            headers,
            rows,
            foot,
            filterable=True,
            sticky=True,
            title="All owners",
            note="Sorted by attributed hours; the bar under each figure is its share of the largest.",
        )
        + "</div></section>"
    )


def files_section(report: Report) -> str:
    classes = class_columns(report)
    headers = (
        [
            ("File", True),
            ("Owner", True),
            ("Hours", False),
            MIX_HEADER,
            ("Share", False),
            ("Jobs", False),
            ("PR share", False),
        ]
        + class_headers(classes)
        + [("Raw test h", False)]
    )
    top = report.files[0].hours if report.files else 0.0
    rows = []
    for f in report.files:
        default_name = f.key.removeprefix("test/").removesuffix(".py").replace("/", ".")
        aliases = sorted(f.members - {default_name})
        label = esc(f.key) + (
            f' <span class="m">via {esc(", ".join(aliases))}</span>' if aliases else ""
        )
        rows.append(agg_row(f, report, classes, td(label, None, "t"), True, top))
    about = (
        "Raw test hours are the summed per-test durations; they exceed attributed hours when tests run in parallel inside a job. "
        '"via" lists invoking names that differ from the file name. The Mix bar shows the hardware-family split of the row and sorts by accelerator share.'
    )
    return (
        '<section id="files">'
        + section_head(
            "files",
            "Every test file with attributed hours, with its owner and hardware mix.",
            about,
        )
        + '<div class="card">'
        + table(
            "files-table",
            headers,
            rows,
            filterable=True,
            sticky=True,
            title="All files",
            note="Sorted by attributed hours; the bar under each figure is its share of the largest.",
        )
        + "</div></section>"
    )


def hardware_section(report: Report) -> str:
    max_wall = max(report.class_wall_h.values(), default=0.0)
    class_rows = []
    for c in HW_CLASSES:
        wall, attr = report.class_wall_h[c], report.class_attr_h[c]
        if wall <= 0 and attr <= 0:
            continue
        labels = [lab for lab in report.labels if lab.hw.cls == c]
        class_rows.append(
            [
                td(
                    f'<i class="dot" style="background:{FAMILY_FILL[HW_FAMILY[c]]}"></i>{esc(c)}',
                    None,
                    "t",
                ),
                text(HW_FAMILY[c]),
                num(wall, bar=ratio(wall, max_wall)),
                pct(ratio(wall, report.wall_h)),
                num(attr),
                pct(ratio(attr, wall)),
                count(sum(lab.jobs for lab in labels)),
                count(len(labels)),
            ]
        )
    blank = td("", 0.0, "m")
    max_sub = max((s.wall_h for s in report.subclasses), default=0.0)
    sub_rows = [
        [
            text(s.hw.subclass),
            text(s.hw.cls),
            count(s.hw.gpus) if s.hw.gpus else blank,
            num(s.wall_h, bar=ratio(s.wall_h, max_sub)),
            pct(ratio(s.wall_h, report.wall_h)),
            num(s.wall_h * s.hw.gpus) if s.hw.gpus else blank,
            num(s.attr_h),
            pct(ratio(s.attr_h, s.wall_h)),
            count(s.jobs),
            count(sum(1 for lab in report.labels if lab.hw == s.hw)),
        ]
        for s in report.subclasses
    ]
    sub_foot = [
        td("Total", None, "t"),
        td("", None, "t"),
        blank,
        num(report.wall_h),
        pct(1.0 if report.wall_h else 0.0),
        num(report.gpu_h),
        num(report.attr_h),
        pct(report.coverage),
        count(report.total_jobs),
        count(len(report.labels)),
    ]
    accel = [s for s in report.subclasses if s.hw.cls in ACCELERATORS and s.attr_h > 0]
    gpus_of = {s.hw.subclass: s.hw.gpus for s in accel}
    owner_rows = []
    for o in report.owners:
        split = {k: h for k, h in o.by_subclass.items() if k in gpus_of and h > 0}
        if not split:
            continue
        hours = sum(split.values())
        gpu_h = sum(h * gpus_of[k] for k, h in split.items())
        cells = [text(o.key), num(hours), num(gpu_h)]
        cells += [num(split[k]) if k in split else blank for k in gpus_of]
        owner_rows.append((-hours, o.key, cells))
    owner_rows.sort()
    owner_foot = [
        td("Total", None, "t"),
        num(sum(s.attr_h for s in accel)),
        num(report.attr_gpu_h),
    ] + [num(s.attr_h) for s in accel]
    label_rows = [
        [
            text(lab.label),
            text(lab.hw.cls),
            text(lab.hw.subclass),
            count(lab.hw.gpus) if lab.hw.gpus else blank,
            count(lab.jobs),
            num(lab.wall_h),
            num(lab.attr_h),
            pct(ratio(lab.attr_h, lab.wall_h)),
        ]
        for lab in report.labels
    ]
    accel_classes = ", ".join(c for c in HW_CLASSES if c in ACCELERATORS)
    about = (
        f"Accelerator classes: {accel_classes}. GPU-hours multiply runner hours by GPUs per runner. "
        "Hours are not price-weighted; an H100 hour costs far more than a CPU hour, and donated hardware has no price in the CI cost tables."
    )
    return (
        '<section id="hardware">'
        + section_head(
            "hardware",
            "Test-job hours by hardware class, by GPU sub-class and by runner label.",
            about,
        )
        + '<div class="card">'
        + table(
            "class-table",
            [
                ("Class", True),
                ("Family", True),
                ("Test-job h", False),
                ("Share", False),
                ("Attributed h", False),
                ("Coverage", False),
                ("Jobs", False),
                ("Labels", False),
            ],
            class_rows,
            title="By hardware class",
            note="Dots use the owner chart's family colours.",
        )
        + "</div>"
        + '<div class="card">'
        + table(
            "subclass-table",
            [
                ("Sub-class", True),
                ("Class", True),
                ("GPUs", False),
                ("Test-job h", False),
                ("Share", False),
                ("GPU-hours", False),
                ("Attributed h", False),
                ("Coverage", False),
                ("Jobs", False),
                ("Labels", False),
            ],
            sub_rows,
            sub_foot,
            title="By sub-class",
            note="Sub-class = GPU model plus GPUs per runner, both parsed from the runner label; accelerator labels without a count are single-GPU. Non-accelerator classes have one sub-class equal to the class.",
        )
        + "</div>"
        + '<div class="card">'
        + table(
            "owner-subclass-table",
            [("Owner", True), ("Accelerator h", False), ("GPU-hours", False)]
            + [(s.hw.subclass, False) for s in accel],
            [cells for _, _, cells in owner_rows],
            owner_foot,
            filterable=True,
            sticky=True,
            title="Owners by sub-class",
            note="Attributed hours per owner on each accelerator sub-class, with GPU-hours; owners without accelerator hours are omitted.",
        )
        + "</div>"
        + '<div class="card">'
        + table(
            "label-table",
            [
                ("Runner label", True),
                ("Class", True),
                ("Sub-class", True),
                ("GPUs", False),
                ("Jobs", False),
                ("Test-job h", False),
                ("Attributed h", False),
                ("Coverage", False),
            ],
            label_rows,
            filterable=True,
            title="By runner label",
            note="The mapping actually used; a label in the wrong class or sub-class means the classifier needs a new pattern.",
        )
        + "</div></section>"
    )


def coverage_section(report: Report) -> str:
    day_rows = [
        [
            text(str(d.day)),
            count(d.jobs),
            num(d.wall_h),
            num(d.attr_h),
            pct(ratio(d.attr_h, d.wall_h)),
            text("settled" if d.settled else "unsettled"),
        ]
        for d in report.days
    ]
    workflows = sorted(
        report.workflows, key=lambda wf: (-(wf.wall_h - wf.attr_h), wf.workflow)
    )
    wf_rows = [
        [
            text(wf.workflow),
            count(wf.jobs),
            num(wf.wall_h),
            num(wf.attr_h),
            num(wf.wall_h - wf.attr_h),
            pct(ratio(wf.attr_h, wf.wall_h)),
        ]
        for wf in workflows
    ]
    if not report.verify_requested:
        verify_html = (
            '<p class="sub">GitHub cross-check disabled (--verify-sample 0).</p>'
        )
    elif report.verify_error:
        verify_html = f'<p class="sub bad">GitHub cross-check failed: {esc(report.verify_error)}</p>'
    else:
        ok = sum(1 for v in report.verify if v.status == "OK")
        verdict = f"{ok} of {len(report.verify)} sampled jobs match the GitHub Actions API on runner label and wall seconds (within 1 s)."
        if ok != len(report.verify):
            verdict += " Mismatches or errors are listed below; a mismatch means the ClickHouse mirror and GitHub disagree for that job."
        verify_rows = []
        for v in report.verify:
            url = f"https://github.com/pytorch/pytorch/actions/runs/{v.run_id}/job/{v.job_id}"
            delta = "" if v.gh_wall_s is None else f"{v.gh_wall_s - v.ch_wall_s:+d}"
            verify_rows.append(
                [
                    td(f'<a href="{url}">{v.job_id}</a>', v.job_id, "t"),
                    text(v.ch_label),
                    text(v.gh_label),
                    count(v.ch_wall_s),
                    td(
                        "" if v.gh_wall_s is None else f"{v.gh_wall_s:,}",
                        v.gh_wall_s or 0,
                    ),
                    td(delta, 0 if v.gh_wall_s is None else v.gh_wall_s - v.ch_wall_s),
                    text(v.status, "t" if v.status == "OK" else "t bad"),
                    text(v.note),
                ]
            )
        verify_html = f'<p class="sub">{esc(verdict)}</p>' + table(
            "verify-table",
            [
                ("Job", True),
                ("Label (ClickHouse)", True),
                ("Label (GitHub)", True),
                ("Wall s (CH)", False),
                ("Wall s (GH)", False),
                ("Delta", False),
                ("Status", True),
                ("Note", True),
            ],
            verify_rows,
        )
    about = "Unsettled days are younger than the settle period and are refetched on every run because test results can arrive up to two days late; settled days are cached."
    return (
        '<section id="coverage">'
        + section_head(
            "coverage",
            "How much of the test-job time could be attributed to test files, per day and per workflow, plus the GitHub cross-check of the job data.",
            about,
        )
        + '<div class="card">'
        + table(
            "day-table",
            [
                ("Day", True),
                ("Jobs", False),
                ("Test-job h", False),
                ("Attributed h", False),
                ("Coverage", False),
                ("State", True),
            ],
            day_rows,
            title="By day",
        )
        + "</div>"
        + '<div class="card">'
        + table(
            "workflow-table",
            [
                ("Workflow", True),
                ("Jobs", False),
                ("Test-job h", False),
                ("Attributed h", False),
                ("Unattributed h", False),
                ("Coverage", False),
            ],
            wf_rows,
            filterable=True,
            title="By workflow",
            note="Sorted by unattributed hours. Non-successful jobs remain unattributed even if some test results exist. Missing reports, upload lag and ingestion gaps also reduce coverage.",
        )
        + "</div>"
        + '<div class="card"><div class="card-head"><div><h3>GitHub cross-check</h3></div></div>'
        + verify_html
        + "</div></section>"
    )


def unmapped_section(report: Report) -> str:
    unmapped_rows = [
        [text(u.invoking_file), num(u.attr_h), count(u.jobs)] for u in report.unmapped
    ]
    no_header = [f for f in report.files if f.owner == "no-header"]
    no_header_rows = [[text(f.key), num(f.hours), count(f.jobs)] for f in no_header]
    unmapped_h = sum(u.attr_h for u in report.unmapped)
    lede = (
        f"{len(report.unmapped)} invoking names ({unmapped_h:,.1f} h) have no test/&lt;name&gt;.py in this checkout: C++ gtest launchers such as test_libtorch, or files that only exist on a PR branch. "
        f"{len(no_header)} files ({sum(f.hours for f in no_header):,.1f} h) have no Owner(s) header."
    )
    none = '<p class="sub">None.</p>'
    return (
        '<section id="unmapped">'
        + section_head("unmapped", lede)
        + '<div class="card">'
        + (
            table(
                "unmapped-table",
                [("Invoking file", True), ("Hours", False), ("Jobs", False)],
                unmapped_rows,
                title="Invoking names without a file",
            )
            if unmapped_rows
            else '<div class="card-head"><div><h3>Invoking names without a file</h3></div></div>'
            + none
        )
        + "</div>"
        + '<div class="card">'
        + (
            table(
                "noheader-table",
                [("File", True), ("Hours", False), ("Jobs", False)],
                no_header_rows,
                title="Files without an Owner(s) header",
            )
            if no_header_rows
            else '<div class="card-head"><div><h3>Files without an Owner(s) header</h3></div></div>'
            + none
        )
        + "</div></section>"
    )


def methodology_section(report: Report) -> str:
    m = report.meta
    items = [
        "Jobs: <code>default.workflow_job</code> rows named <code>... / test (...)</code> or <code>... / test-osdc (...)</code> for pytorch/pytorch with status completed and a runner assigned, deduplicated by job id and bucketed by <code>completed_at</code> UTC day. All conclusions count toward test-job hours. Wall seconds are completed_at minus started_at, so queue time is excluded; a rerun is a new job id and counts separately. Rows whose started_at is after completed_at or more than 7 days before the window are dropped as corrupt timestamps.",
        "Per-test seconds: <code>tests.all_test_runs.time</code> grouped by job and invoking file, taking rows inserted between one day before and two days after the job day. Each source report contributes only its latest ingestion snapshot; repeated testcases within that snapshot and separate rerun reports are preserved. File job counts deduplicate job IDs after resolving invoking-file aliases.",
        "Attribution: only successful jobs are eligible. hours(file, job) = job wall hours x file test seconds / job test seconds; when every test in a job reports zero seconds the split uses test counts instead. Non-successful jobs remain entirely unattributed because interrupted files may never upload a report; this also excludes failed jobs with complete reports. File job counts and raw test seconds cover only successful jobs. Attributed hours never exceed test-job hours.",
        "Trigger (from <code>default.workflow_run</code>): main = pushes to main plus workflow_dispatch runs on trunk/&lt;sha&gt; tags; pr = pull_request events plus ciflow/* tag pushes; scheduled = schedule events; other = everything else.",
        f"Owner: first label of the <code># Owner(s): [...]</code> header in the checkout at {esc(m.checkout)}, with the module: or oncall: prefix removed. <code>unknown</code> is the literal <code>module: unknown</code> label, <code>no-header</code> marks files without the header, <code>unmapped</code> marks invoking names with no file in the checkout. Tests imported by another file are charged to the importing file.",
        "Hardware class: an ordered regex table over the runner label, shown in the Hardware section; accelerator labels that match no known GPU stay <code>unknown</code> rather than counting as CPU. Sub-class = GPU model plus GPUs per runner, both parsed from the label: the model token (mi300, gfx950, idc, tpuv7x; NVIDIA classes already name the model) and the count from the ARC <code>-h100-4</code> suffix, the dotted <code>.4</code> suffix or the legacy EC2 size; accelerator labels without a count are single-GPU. GPU-hours = runner hours x GPUs per runner. Neither unit is price-weighted: an H100 hour costs far more than a CPU hour, and donated hardware (ROCm, B200, XPU, TPU, s390x) has no price in the CI cost tables, so hours by class are the honest unit.",
        "Coverage: non-successful jobs and jobs without per-test results remain unattributed. Some workflows never upload results (XPU, s390x, TSan, torchtitan, most perf jobs); upload lag and ingestion gaps can also reduce coverage. A successful conclusion does not guarantee every report arrived. Check the data quality warnings alongside the daily totals.",
        "Determinism: days at least three days old are cached per query hash under agent_space/test-cost/cache and never refetched; younger days are refetched on every run, so only settled windows reproduce byte for byte. Owners reflect the checkout at report time.",
        f"GitHub cross-check: {report.verify_requested} jobs chosen by cityHash64(id) over the window are fetched from the GitHub Actions API and compared on runner label and wall seconds."
        if report.verify_requested
        else "GitHub cross-check: disabled for this report.",
    ]
    bullets = []
    for item in items:
        term, _, rest = item.partition(": ")
        bullets.append(f"<li><b>{term}.</b> {rest}</li>")
    return (
        '<section id="methodology">'
        + section_head(
            "methodology",
            "How the numbers are produced, and how to reproduce this page.",
        )
        + '<div class="card"><ul class="method">'
        + "".join(bullets)
        + f'</ul><p class="sub">Reproduce with the command below (query hash {esc(m.query_hash)}, checkout {esc(m.checkout)}).</p>'
        + f'<pre class="cmd">{esc(m.command)}</pre></div></section>'
    )


def render(report: Report) -> str:
    w = report.meta.window
    nav = "".join(f'<a href="#{sid}">{SECTION_TITLES[sid]}</a>' for sid in SECTION_IDS)
    sections = [
        summary_section(report),
        charts_section(report),
        owners_section(report),
        files_section(report),
        hardware_section(report),
        coverage_section(report),
        unmapped_section(report),
        methodology_section(report),
    ]
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark">'
        f"<title>CI test cost {w.start} to {w.last}</title><style>{CSS}</style></head><body>"
        f'<header class="top"><nav><b>test-cost</b>{nav}</nav></header><main>'
        + "\n".join(sections)
        + f"</main><script>{JS}</script></body></html>\n"
    )
