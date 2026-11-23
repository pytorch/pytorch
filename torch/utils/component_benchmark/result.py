from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


@dataclass
class ComponentResult:
    module_result: dict[str, Any]
    plugin_name: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)

    def update(self, result: dict[str, Any]) -> None:
        self.module_result.update(result)


@dataclass
class BenchmarkResult:
    """Measured component metrics, failures and local artifact paths."""

    result: dict[str, ComponentResult] = field(default_factory=dict)
    errors: dict[str, dict[str, str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_component_metrics(
        self, component_name: str, result: ComponentResult
    ) -> None:
        target = self.result.setdefault(component_name, ComponentResult({}))
        prefix = f"{result.plugin_name}." if result.plugin_name else ""
        metrics = {prefix + key: value for key, value in result.module_result.items()}
        duplicates = target.module_result.keys() & metrics.keys()
        if duplicates:
            raise ValueError(
                f"Duplicate metrics for {component_name}: {sorted(duplicates)}"
            )
        target.module_result.update(metrics)
        target.artifacts.update(
            {prefix + key: value for key, value in result.artifacts.items()}
        )

    def add_error(
        self, component_name: str, error: str, plugin: str = "capture"
    ) -> None:
        self.errors.setdefault(component_name, {})[plugin] = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "components": {
                name: {"metrics": value.module_result, "artifacts": value.artifacts}
                for name, value in self.result.items()
            },
            "errors": self.errors,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )

    def save_csv(self, path: str | Path) -> None:
        keys = sorted(
            {key for result in self.result.values() for key in result.module_result}
        )
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["component", *keys, "errors"])
            writer.writeheader()
            for name in dict.fromkeys([*self.result, *self.errors]):
                metrics = self.result.get(name, ComponentResult({})).module_result
                writer.writerow(
                    {
                        "component": name,
                        **metrics,
                        "errors": json.dumps(self.errors.get(name, {})),
                    }
                )

    def save_html(self, path: str | Path) -> None:
        """Write a standalone interactive hierarchy and metric table."""
        payload = json.dumps(self.to_dict(), allow_nan=False).replace("<", "\\u003c")
        keys = sorted(
            {
                key
                for value in self.result.values()
                for key, metric in value.module_result.items()
                if key == "status"
                or isinstance(metric, (int, float, bool))
                or metric is None
            }
        )
        headers = "".join(f"<th>{html.escape(key)}</th>" for key in keys)
        rows = []
        for name in dict.fromkeys([*self.result, *self.errors]):
            value = self.result.get(name, ComponentResult({}))
            links = " ".join(
                f'<a href="{html.escape(Path(target).resolve().as_uri(), quote=True)}">{html.escape(label)}</a>'
                for label, target in value.artifacts.items()
            )
            metrics = "".join(
                f"<td>{html.escape(str(value.module_result.get(key, '')))}</td>"
                for key in keys
            )
            rows.append(
                f"<tr><td>{html.escape(name)}</td>{metrics}<td>{links}</td>"
                f"<td><details><summary>{len(self.errors.get(name, {}))} failures</summary>"
                f"<pre>{html.escape(json.dumps(self.errors.get(name, {})))}</pre></details></td></tr>"
            )
        title = "Component Benchmark"
        document = (
            _REPORT.replace("__TITLE__", title)
            .replace("__HEADERS__", headers)
            .replace("__ROWS__", "".join(rows))
            .replace("__DATA__", payload)
        )
        Path(path).write_text(document, encoding="utf-8")


def run_metadata() -> dict[str, str]:
    return {"started_at_utc": datetime.now(UTC).isoformat()}


_REPORT = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1200px;margin:24px auto;padding:0 16px}
button,input,select{font:inherit;padding:6px}svg{width:100%;min-height:70px}rect{stroke:Canvas;stroke-width:2;cursor:pointer}
text{fill:CanvasText;pointer-events:none;font-size:13px}table{border-collapse:collapse;width:100%}td,th{border:1px solid GrayText;padding:8px;text-align:left;vertical-align:top}
pre{margin:0;white-space:pre-wrap;font-size:12px}.scroll{overflow-x:auto}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
</style>
<h1>__TITLE__</h1>
<p>Every component is replayed independently. Parent and child latencies overlap in scope and are not additive.
Rectangle widths use relative sibling measurements for navigation, not an additive model-time decomposition.</p>
<div class="controls"><label>Metric <select id="metric"></select></label><button id="up" type="button">Whole model</button>
<label>Filter table <input id="filter" type="search"></label></div>
<p id="selection" aria-live="polite"></p>
<svg id="tree" role="img" aria-label="Interactive module hierarchy"><title>Module hierarchy</title><desc>Choose a metric and select a module to zoom.</desc></svg>
<div class="scroll"><table><thead><tr><th>Component</th>__HEADERS__<th>Artifacts</th><th>Errors</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), ns='http://www.w3.org/2000/svg';
const nodes=new Map(), metric=document.getElementById('metric'), svg=document.getElementById('tree');
for(const name of new Set([...Object.keys(data.components),...Object.keys(data.errors)])) {
 const parts=name.split('.'); for(let i=1;i<=parts.length;i++) {const path=parts.slice(0,i).join('.');if(!nodes.has(path))nodes.set(path,{path,children:[],metrics:data.components[path]?.metrics||{}});}
}
const root={path:'All components',children:[],metrics:{}};
for(const node of nodes.values()){const parent=nodes.get(node.path.slice(0,node.path.lastIndexOf('.')));(parent||root).children.push(node);}
const keys=[...new Set([...nodes.values()].flatMap(n=>Object.keys(n.metrics).filter(k=>typeof n.metrics[k]==='number')))].sort();
for(const key of keys){const option=document.createElement('option');option.value=key;option.textContent=key;metric.appendChild(option);}
let selected=root;
function weight(node){const own=node.metrics[metric.value];if(typeof own==='number'&&own>0)return own;return node.children.reduce((s,n)=>s+weight(n),0)||1;}
function render(){svg.replaceChildren();let maxDepth=0;const maximum=Math.max(1,...[...nodes.values()].map(n=>Number(n.metrics[metric.value])||0));
 function draw(node,x,width,depth){maxDepth=Math.max(maxDepth,depth);const g=document.createElementNS(ns,'g'),r=document.createElementNS(ns,'rect');
  for(const [key,value] of Object.entries({x,y:depth*48,width,height:46,rx:3}))r.setAttribute(key,value);
  const value=node.metrics[metric.value],hue=typeof value==='number'?120*Math.max(0,value)/maximum:210;
  r.setAttribute('fill',`hsl(${hue} 48% 65%)`);r.setAttribute('tabindex','0');r.setAttribute('role','button');
  const label=node.path+(value===undefined?'':`: ${value}`);r.setAttribute('aria-label',label);
  const title=document.createElementNS(ns,'title');title.textContent=label;r.appendChild(title);
  const choose=()=>{selected=node;render();};r.addEventListener('click',choose);r.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});g.appendChild(r);
  if(width>90){const t=document.createElementNS(ns,'text');t.setAttribute('x',x+7);t.setAttribute('y',depth*48+27);t.textContent=node.path.split('.').at(-1).slice(0,Math.floor(width/8));g.appendChild(t);}svg.appendChild(g);
  const total=node.children.reduce((s,n)=>s+weight(n),0);let offset=x;for(const child of node.children){const w=width*weight(child)/total;draw(child,offset,w,depth+1);offset+=w;}
 }draw(selected,0,1100,0);svg.setAttribute('viewBox',`0 0 1100 ${(maxDepth+1)*48}`);document.getElementById('selection').textContent=selected.path;
}
metric.addEventListener('change',render);document.getElementById('up').addEventListener('click',()=>{selected=root;render();});
document.getElementById('filter').addEventListener('input',e=>{const term=e.target.value.toLowerCase();for(const row of document.querySelectorAll('tbody tr'))row.hidden=!row.textContent.toLowerCase().includes(term);});render();
</script></html>
"""
