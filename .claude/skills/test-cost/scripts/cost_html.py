"""Render a Report as one self-contained HTML page.

Inline CSS with light and dark tokens, inline SVG charts and a few lines of
vanilla JavaScript for sorting and filtering; no external assets. Colours and
mark specs follow the dataviz skill's reference palette.
"""

from __future__ import annotations

import html
import math

from cost_data import (
    ACCELERATORS,
    Agg,
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
FAMILY_CLASSES = {
    fam: [c for c in HW_CLASSES if HW_FAMILY[c] == fam] for fam in FAMILIES
}
CHART_W = 960
BAR = 18
PITCH = 28
esc = html.escape

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--dim:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--dim:#52514e;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.top{position:sticky;top:0;z-index:2;background:var(--surface);border-bottom:1px solid var(--border)}
.top nav{max-width:1600px;margin:0 auto;padding:.45rem 1.25rem;display:flex;gap:.25rem;flex-wrap:wrap;align-items:center}
.top b{margin-right:.75rem}
.top a{color:var(--ink2);text-decoration:none;padding:.25rem .6rem;border-radius:6px;font-size:.9rem}
.top a:hover{background:var(--page);color:var(--ink)}
main{max-width:1600px;margin:0 auto;padding:1.5rem 1.25rem 4rem}
h1{font-size:1.75rem;margin:.25rem 0}
h2{font-size:1.3rem;margin:2.5rem 0 .5rem}
h3{font-size:1.02rem;margin:1.25rem 0 .4rem}
p{margin:.35rem 0}
.sub{color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.25rem;margin:.75rem 0}
.hero{font-size:56px;font-weight:600;line-height:1.1;letter-spacing:-.01em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.75rem;margin-top:1.25rem}
.tile{background:var(--page);border-radius:8px;padding:.7rem .9rem}
.tile .v{font-size:1.5rem;font-weight:600}
.tile .l{color:var(--ink2);font-size:.9rem}
.tile .c{color:var(--muted);font-size:.82rem}
figure{margin:0 0 1.5rem}
figure h3{margin:0 0 .1rem}
figure .sub{font-size:.9rem;margin:0 0 .4rem}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.85rem;color:var(--ink2);margin:.3rem 0 .5rem}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:.35rem}
svg text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink2)}
svg .tick{fill:var(--muted);font-variant-numeric:tabular-nums}
svg .val{fill:var(--ink);font-variant-numeric:tabular-nums}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .mark:hover,svg .mark:focus{filter:brightness(1.12);outline:none}
.scroll{overflow-x:auto}
.tall{max-height:75vh;overflow:auto}
.tall th{position:sticky;top:0;z-index:1}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:.3rem .45rem;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600;background:var(--surface);cursor:pointer;user-select:none}
th.t,td.t{text-align:left}
th[data-dir=asc]::after{content:" \\25B2";font-size:.7em}
th[data-dir=desc]::after{content:" \\25BC";font-size:.7em}
tfoot td{font-weight:600;border-top:1px solid var(--axis)}
td.m,span.m{color:var(--muted)}
.tools{display:flex;gap:.75rem;align-items:center;margin:.4rem 0 .6rem}
.tools input{font:inherit;padding:.35rem .5rem;border:1px solid var(--axis);border-radius:6px;background:var(--surface);color:var(--ink);min-width:17rem}
.tools .n{color:var(--muted);font-size:.85rem}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;background:var(--page);padding:.1em .3em;border-radius:4px}
.method li{margin:.4rem 0}
.bad{font-weight:600}
"""

JS = """
(function(){
  function val(td){var v=td.getAttribute('data-v');return v===null?td.textContent.trim().toLowerCase():parseFloat(v);}
  document.querySelectorAll('table.sortable').forEach(function(t){
    var heads=t.tHead.rows[0].cells,body=t.tBodies[0];
    Array.prototype.forEach.call(heads,function(th,i){
      th.addEventListener('click',function(){
        var asc=th.getAttribute('data-dir')==='desc';
        Array.prototype.forEach.call(heads,function(h){h.removeAttribute('data-dir');});
        th.setAttribute('data-dir',asc?'asc':'desc');
        var rows=Array.prototype.slice.call(body.rows);
        rows.sort(function(a,b){var x=val(a.cells[i]),y=val(b.cells[i]);
          if(typeof x==='number'&&typeof y==='number'){return asc?x-y:y-x;}
          return asc?String(x).localeCompare(String(y)):String(y).localeCompare(String(x));});
        rows.forEach(function(r){body.appendChild(r);});
      });
    });
  });
  document.querySelectorAll('input[data-filter]').forEach(function(inp){
    var t=document.getElementById(inp.getAttribute('data-filter'));
    var n=document.getElementById(inp.getAttribute('data-filter')+'-n');
    var rows=Array.prototype.slice.call(t.tBodies[0].rows);
    inp.addEventListener('input',function(){
      var q=inp.value.trim().toLowerCase(),k=0;
      rows.forEach(function(r){var show=!q||r.textContent.toLowerCase().indexOf(q)>=0;r.style.display=show?'':'none';if(show){k++;}});
      n.textContent=k+' of '+rows.length+' rows';
    });
  });
})();
"""


def fmt_k(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e4:
        return f"{value / 1e3:.0f}K"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def fmt_h(hours: float) -> str:
    return f"{hours:,.0f}" if hours >= 100 else f"{hours:,.1f}"


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


def column_path(x: float, y: float, w: float, h: float, r: float) -> str:
    if r <= 0 or w < 2 * r or h < r:
        return f"M{x:.1f},{y:.1f}h{w:.1f}v{h:.1f}h{-w:.1f}z"
    return f"M{x:.1f},{y + r:.1f}a{r},{r} 0 0 1 {r},{-r}h{w - 2 * r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - r:.1f}h{-w:.1f}z"


def hbar_svg(
    rows: list[tuple[str, list[tuple[str, float, str]], str]], gutter: int, aria: str
) -> str:
    """Horizontal bars. rows = (label, [(fill, value, tooltip)], value-label suffix html)."""
    n = len(rows)
    plot_w = CHART_W - gutter - 120
    max_v = max((sum(v for _, v, _ in segs) for _, segs, _ in rows), default=0.0)
    step = nice_step(max_v)
    top = step * math.ceil(max_v / step) if max_v > 0 else step
    scale = plot_w / top
    base_y = 8 + n * PITCH
    out = [
        f'<svg viewBox="0 0 {CHART_W} {base_y + 26}" width="100%" role="img" aria-label="{esc(aria)}">'
    ]
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
        visible = [(fill, v, tip) for fill, v, tip in segs if v * scale >= 1]
        x = float(gutter)
        for k, (fill, v, tip) in enumerate(visible):
            w = v * scale
            last = k == len(visible) - 1
            d = bar_path(x, y, w if last else max(w - 2, 0.5), BAR, 4 if last else 0)
            out.append(
                f'<path class="mark" d="{d}" fill="{fill}" tabindex="0"><title>{esc(tip)}</title></path>'
            )
            x += w
        total = sum(v for _, v, _ in segs)
        out.append(
            f'<text class="val" x="{gutter + total * scale + 6:.1f}" y="{text_y:.1f}">{fmt_k(total)} h{suffix}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def day_columns_svg(report: Report) -> str:
    days = report.days
    left, right, plot_h, top_pad = 64, 16, 220, 10
    slot = (CHART_W - left - right) / max(len(days), 1)
    col_w = min(24.0, slot * 0.6)
    max_v = max((d.wall_h for d in days), default=0.0)
    step = nice_step(max_v)
    top = step * math.ceil(max_v / step) if max_v > 0 else step
    scale = plot_h / top
    base_y = top_pad + plot_h
    aria = "Test-job hours per day, attributed and unattributed"
    out = [
        f'<svg viewBox="0 0 {CHART_W} {base_y + 28}" width="100%" role="img" aria-label="{aria}">'
    ]
    for i in range(int(round(top / step)) + 1):
        y = base_y - i * step * scale
        out.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{CHART_W - right}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt_k(i * step)}</text>'
        )
    out.append(
        f'<line class="axis" x1="{left}" y1="{base_y}" x2="{CHART_W - right}" y2="{base_y}"/>'
    )
    for i, d in enumerate(days):
        x = left + slot * (i + 0.5) - col_w / 2
        unattributed = max(d.wall_h - d.attr_h, 0.0)
        segs = [
            (seg_fill, v)
            for seg_fill, v in (("var(--s1)", d.attr_h), ("var(--dim)", unattributed))
            if v * scale >= 1
        ]
        coverage = ratio(d.attr_h, d.wall_h)
        state = "" if d.settled else " (unsettled, may still change)"
        tip = f"{d.day}: {d.wall_h:,.0f} h test-job hours, {d.attr_h:,.0f} h attributed ({coverage:.1%}){state}"
        out.append(f'<g class="mark" tabindex="0"><title>{esc(tip)}</title>')
        out.append(
            f'<rect x="{x - 2:.1f}" y="{top_pad}" width="{col_w + 4:.1f}" height="{plot_h}" fill="transparent"/>'
        )
        y = float(base_y)
        for k, (fill, v) in enumerate(segs):
            h = v * scale
            last = k == len(segs) - 1
            draw_h = h if last else max(h - 2, 0.5)
            out.append(
                f'<path d="{column_path(x, y - h, col_w, draw_h, 4 if last else 0)}" fill="{fill}"/>'
            )
            y -= h
        out.append("</g>")
        out.append(
            f'<text class="tick" x="{x + col_w / 2:.1f}" y="{base_y + 18}" text-anchor="middle">{d.day.strftime("%m-%d")}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def legend(items: list[tuple[str, str]]) -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i style="background:{fill}"></i>{esc(label)}</span>'
            for fill, label in items
        )
        + "</div>"
    )


def figure(title: str, subtitle: str, body: str) -> str:
    return f'<figure class="card"><h3>{esc(title)}</h3><p class="sub">{esc(subtitle)}</p>{body}</figure>'


def td(text: str, value: float | None = None, cls: str = "") -> str:
    attrs = f' class="{cls}"' if cls else ""
    if value is not None:
        attrs += f' data-v="{value:.4f}"'
    return f"<td{attrs}>{text}</td>"


def num(value: float) -> str:
    return td(fmt_h(value), value)


def count(value: int) -> str:
    return td(f"{value:,}", value)


def pct(value: float) -> str:
    return td(f"{value:.1%}", value)


def text(value: str, cls: str = "t") -> str:
    return td(esc(value), None, cls)


def table(
    tid: str,
    headers: list[tuple[str, bool]],
    rows: list[list[str]],
    foot: list[str] | None = None,
    filterable: bool = False,
) -> str:
    out = []
    if filterable:
        out.append(
            f'<div class="tools"><input data-filter="{tid}" placeholder="Filter rows" aria-label="Filter rows"><span class="n" id="{tid}-n">{len(rows):,} of {len(rows):,} rows</span></div>'
        )
    wrap = "scroll tall" if filterable else "scroll"
    head = "".join(
        f'<th class="t">{esc(h)}</th>' if is_text else f"<th>{esc(h)}</th>"
        for h, is_text in headers
    )
    out.append(
        f'<div class="{wrap}"><table id="{tid}" class="sortable"><thead><tr>{head}</tr></thead><tbody>'
    )
    out.extend("<tr>" + "".join(cells) + "</tr>\n" for cells in rows)
    out.append("</tbody>")
    if foot:
        out.append("<tfoot><tr>" + "".join(foot) + "</tr></tfoot>")
    out.append("</table></div>")
    return "".join(out)


def class_columns(report: Report) -> list[str]:
    return [c for c in HW_CLASSES if report.class_attr_h[c] > 0]


def agg_row(
    agg: Agg, report: Report, classes: list[str], label_cell: str, with_jobs: bool
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
    row += [num(hours), pct(ratio(hours, report.attr_h))]
    if with_jobs:
        row.append(count(agg.jobs))
    row.append(pct(ratio(agg.by_trigger["pr"], hours)))
    row += [
        num(agg.by_class[c]) if agg.by_class[c] > 0 else td("", 0.0, "m")
        for c in classes
    ]
    row.append(num(agg.test_h))
    return row


def summary_section(report: Report) -> str:
    m = report.meta
    w = m.window
    unsettled = sum(1 for d in report.days if not d.settled)
    owners = [o for o in report.owners if o.key not in ("unmapped", "no-header")]
    tiles = [
        (
            f"{report.attr_h:,.0f} h",
            "attributed to test files",
            f"{report.coverage:.1%} of test-job hours; the rest uploaded no per-test results",
        ),
        (
            f"{report.total_jobs:,}",
            "test jobs",
            "completed test (...) jobs, reruns counted separately",
        ),
        (
            f"{report.accelerator_share:.1%}",
            "accelerator share",
            "of test-job hours on NVIDIA, ROCm, XPU or TPU runners",
        ),
        (
            f"{len(report.files):,}",
            "test files with attributed hours",
            f"{len(owners):,} owners",
        ),
        (
            f"{len(report.days)}",
            "UTC days",
            f"{unsettled} unsettled, refetched on every run"
            if unsettled
            else "all days settled",
        ),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{esc(v)}</div><div class="l">{esc(label)}</div><div class="c">{esc(cap)}</div></div>'
        for v, label, cap in tiles
    )
    trigger_rows = [
        [
            text(t),
            num(report.trigger_wall_h[t]),
            pct(ratio(report.trigger_wall_h[t], report.wall_h)),
            num(report.trigger_attr_h[t]),
        ]
        for t in TRIGGERS
        if report.trigger_wall_h[t] > 0
    ]
    return (
        f'<section id="summary"><h1>CI test cost: machine-hours by test file and owner</h1>'
        f'<p class="sub">pytorch/pytorch test jobs, {w.start} to {w.last} UTC ({len(report.days)} days). '
        f"Owners come from the checkout at {esc(m.checkout)}. Hours are machine-hours by hardware class; they are not price-weighted.</p>"
        f'<div class="card"><div class="hero">{report.wall_h:,.0f}</div><div class="sub">machine-hours spent in test jobs</div>'
        f'<div class="tiles">{tiles_html}</div></div>'
        f'<div class="card"><h3>Hours by trigger</h3>'
        + table(
            "trigger-table",
            [
                ("Trigger", True),
                ("Test-job h", False),
                ("Share", False),
                ("Attributed h", False),
            ],
            trigger_rows,
        )
        + "</div></section>"
    )


def charts_section(report: Report) -> str:
    if not report.owners:
        return '<section id="charts"><h2>Charts</h2><p class="sub">No attributed test results in this window.</p></section>'
    fills = [f"var(--s{i + 1})" for i in range(len(FAMILIES))]
    family_fill = dict(zip(FAMILIES, fills))
    top_owner = report.owners[0]
    owner_rows = []
    for owner in report.owners[:15]:
        segs = []
        for fam in FAMILIES:
            hours = sum(owner.by_class[c] for c in FAMILY_CLASSES[fam])
            tip = f"{owner.key} on {fam}: {hours:,.0f} h ({ratio(hours, owner.hours):.1%} of {owner.key})"
            segs.append((family_fill[fam], hours, tip))
        owner_rows.append((owner.key, segs, ""))
    legend_items = [
        (
            family_fill[fam],
            f"{fam} ({', '.join(FAMILY_CLASSES[fam])})"
            if len(FAMILY_CLASSES[fam]) > 1
            else fam,
        )
        for fam in FAMILIES
    ]
    owners_fig = figure(
        f"{top_owner.key} tests use {ratio(top_owner.hours, report.attr_h):.0%} of attributed test-job hours",
        f"Attributed machine-hours by owner and hardware family, top {len(owner_rows)} of {len(report.owners)} owners; full table below",
        legend(legend_items)
        + hbar_svg(
            owner_rows, 180, "Attributed hours per owner, stacked by hardware family"
        ),
    )
    top_file = report.files[0] if report.files else None
    file_rows = [
        (
            f.key.removeprefix("test/"),
            [("var(--s1)", f.hours, f"{f.key}: {f.hours:,.0f} h, owner {f.owner}")],
            f'<tspan class="tick"> {esc(f.owner)}</tspan>',
        )
        for f in report.files[:20]
    ]
    files_fig = figure(
        f"{top_file.key} alone uses {fmt_k(top_file.hours)} h ({ratio(top_file.hours, report.attr_h):.1%} of attributed hours)"
        if top_file
        else "No file could be mapped",
        f"Attributed machine-hours by test file with its owner, top {len(file_rows)} of {len(report.files)} files",
        hbar_svg(file_rows, 370, "Attributed hours per test file"),
    )
    classes = sorted(
        (c for c in HW_CLASSES if report.class_wall_h[c] > 0),
        key=lambda c: (-report.class_wall_h[c], HW_CLASSES.index(c)),
    )
    top_class = classes[0]
    class_rows = [
        (
            c,
            [
                (
                    "var(--s1)",
                    report.class_wall_h[c],
                    f"{c}: {report.class_wall_h[c]:,.0f} h test-job hours, {report.class_attr_h[c]:,.0f} h attributed",
                )
            ],
            "",
        )
        for c in classes
    ]
    classes_fig = figure(
        f"{top_class} runners account for {ratio(report.class_wall_h[top_class], report.wall_h):.0%} of test-job hours",
        "Test-job machine-hours by hardware class, including hours that uploaded no per-test results",
        hbar_svg(class_rows, 110, "Test-job hours per hardware class"),
    )
    days_fig = figure(
        f"Attribution covers {report.coverage:.0%} of test-job hours; the rest is jobs that never upload per-test results",
        "Test-job machine-hours per UTC day: attributed to test files and unattributed",
        legend([("var(--s1)", "Attributed"), ("var(--dim)", "Unattributed")])
        + day_columns_svg(report),
    )
    return f'<section id="charts"><h2>Charts</h2>{owners_fig}{files_fig}{classes_fig}{days_fig}</section>'


def owners_section(report: Report) -> str:
    classes = class_columns(report)
    headers = (
        [
            ("Owner", True),
            ("Files", False),
            ("Hours", False),
            ("Share", False),
            ("PR share", False),
        ]
        + [(c, False) for c in classes]
        + [("Raw test h", False)]
    )
    rows = [
        agg_row(o, report, classes, text(o.key), with_jobs=False) for o in report.owners
    ]
    foot = [
        td("Total", None, "t"),
        count(len(report.files)),
        num(report.attr_h),
        pct(1.0 if report.attr_h else 0.0),
        pct(ratio(report.trigger_attr_h["pr"], report.attr_h)),
    ]
    foot += [num(report.class_attr_h[c]) for c in classes] + [
        num(sum(o.test_h for o in report.owners))
    ]
    return (
        '<section id="owners"><h2>Owners</h2>'
        '<p class="sub">Attributed machine-hours per owner: the first <code># Owner(s)</code> label of each file with the module: / oncall: prefix removed. '
        "<code>unknown</code> is the literal <code>module: unknown</code> label; <code>no-header</code> files have no header; <code>unmapped</code> invoking names have no file in this checkout. "
        'Click a header to sort; type to filter.</p><div class="card">'
        + table("owners-table", headers, rows, foot, filterable=True)
        + "</div></section>"
    )


def files_section(report: Report) -> str:
    classes = class_columns(report)
    headers = (
        [
            ("File", True),
            ("Owner", True),
            ("Hours", False),
            ("Share", False),
            ("Jobs", False),
            ("PR share", False),
        ]
        + [(c, False) for c in classes]
        + [("Raw test h", False)]
    )
    rows = []
    for f in report.files:
        default_name = f.key.removeprefix("test/").removesuffix(".py").replace("/", ".")
        aliases = sorted(f.members - {default_name})
        label = esc(f.key) + (
            f' <span class="m">via {esc(", ".join(aliases))}</span>' if aliases else ""
        )
        rows.append(agg_row(f, report, classes, td(label, None, "t"), with_jobs=True))
    return (
        '<section id="files"><h2>Files</h2>'
        '<p class="sub">Every test file with attributed hours. Raw test hours are the summed per-test durations; they exceed attributed hours when tests run in parallel inside a job. '
        '"via" lists invoking names that differ from the file name.</p><div class="card">'
        + table("files-table", headers, rows, filterable=True)
        + "</div></section>"
    )


def hardware_section(report: Report) -> str:
    label_count = {
        c: sum(1 for lab in report.labels if lab.hw_class == c) for c in HW_CLASSES
    }
    class_rows = []
    for c in HW_CLASSES:
        wall, attr = report.class_wall_h[c], report.class_attr_h[c]
        if wall <= 0 and attr <= 0:
            continue
        jobs = sum(lab.jobs for lab in report.labels if lab.hw_class == c)
        class_rows.append(
            [
                text(c),
                text(HW_FAMILY[c]),
                num(wall),
                pct(ratio(wall, report.wall_h)),
                num(attr),
                pct(ratio(attr, wall)),
                count(jobs),
                count(label_count[c]),
            ]
        )
    label_rows = [
        [
            text(lab.label),
            text(lab.hw_class),
            count(lab.jobs),
            num(lab.wall_h),
            num(lab.attr_h),
            pct(ratio(lab.attr_h, lab.wall_h)),
        ]
        for lab in report.labels
    ]
    accel = ", ".join(c for c in HW_CLASSES if c in ACCELERATORS)
    return (
        '<section id="hardware"><h2>Hardware</h2>'
        f'<p class="sub">Test-job hours by hardware class and by runner label. Accelerator classes: {accel}. '
        "Hours are not price-weighted; an H100 hour costs far more than a CPU hour.</p>"
        '<div class="card"><h3>By hardware class</h3>'
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
        )
        + '<h3>By runner label</h3><p class="sub">The mapping actually used; a label in the wrong class means the classifier needs a new pattern.</p>'
        + table(
            "label-table",
            [
                ("Runner label", True),
                ("Class", True),
                ("Jobs", False),
                ("Test-job h", False),
                ("Attributed h", False),
                ("Coverage", False),
            ],
            label_rows,
            filterable=True,
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
    return (
        '<section id="coverage"><h2>Coverage</h2>'
        '<p class="sub">How much of the test-job time could be attributed to test files, per day and per workflow, and the GitHub cross-check of the job data.</p>'
        '<div class="card"><h3>By day</h3><p class="sub">Unsettled days are younger than the settle period and are refetched on every run because test results can arrive up to two days late; settled days are cached.</p>'
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
        )
        + '<h3>By workflow</h3><p class="sub">Sorted by unattributed hours. Workflows near 0% coverage never upload per-test results (XPU, s390x, TSan, torchtitan, most perf jobs); their hours are structurally unattributable.</p>'
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
        )
        + "<h3>GitHub cross-check</h3>"
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
    return (
        '<section id="unmapped"><h2>Unmapped</h2>'
        f'<p class="sub">{len(report.unmapped)} invoking names ({unmapped_h:,.1f} h) have no test/&lt;name&gt;.py in this checkout: C++ gtest launchers such as test_libtorch, or files that only exist on a PR branch. '
        f"{len(no_header)} files ({sum(f.hours for f in no_header):,.1f} h) have no Owner(s) header.</p>"
        '<div class="card"><h3>Invoking names without a file</h3>'
        + (
            table(
                "unmapped-table",
                [("Invoking file", True), ("Hours", False), ("Jobs", False)],
                unmapped_rows,
            )
            if unmapped_rows
            else '<p class="sub">None.</p>'
        )
        + "<h3>Files without an Owner(s) header</h3>"
        + (
            table(
                "noheader-table",
                [("File", True), ("Hours", False), ("Jobs", False)],
                no_header_rows,
            )
            if no_header_rows
            else '<p class="sub">None.</p>'
        )
        + "</div></section>"
    )


def methodology_section(report: Report) -> str:
    m = report.meta
    items = [
        "Jobs: <code>default.workflow_job</code> rows named <code>... / test (...)</code> for pytorch/pytorch with status completed and a runner assigned, deduplicated by job id and bucketed by <code>completed_at</code> UTC day. Wall seconds are completed_at minus started_at, so queue time is excluded; a rerun is a new job id and counts separately.",
        "Per-test seconds: <code>tests.all_test_runs.time</code> grouped by job and invoking file, taking rows inserted between one day before and two days after the job day. Each source report contributes only its latest ingestion snapshot; repeated testcases within that snapshot and separate rerun reports are preserved. File job counts deduplicate job IDs after resolving invoking-file aliases.",
        "Attribution: hours(file, job) = job wall hours x file test seconds / job test seconds; when every test in a job reports zero seconds the split uses test counts instead. Attributed hours therefore never exceed test-job hours. Cancelled and failed jobs count when they uploaded results, because the machine time was spent.",
        "Trigger: main = pushes to main plus workflow_dispatch runs on trunk/&lt;sha&gt; tags; pr = pull_request events plus ciflow/* tag pushes; scheduled = schedule events; other = everything else.",
        f"Owner: first label of the <code># Owner(s): [...]</code> header in the checkout at {esc(m.checkout)}, with the module: or oncall: prefix removed. <code>unknown</code> is the literal <code>module: unknown</code> label, <code>no-header</code> marks files without the header, <code>unmapped</code> marks invoking names with no file in the checkout. Tests imported by another file are charged to the importing file.",
        "Hardware class: an ordered regex table over the runner label, shown in the Hardware section; accelerator labels that match no known GPU stay <code>unknown</code> rather than counting as CPU. GPU-count suffixes such as -t4-4 share the single-GPU class. Classes are not price-weighted: an H100 hour costs far more than a CPU hour, and donated hardware (ROCm, B200, XPU, TPU, s390x) has no price in the CI cost tables, so hours by class are the honest unit.",
        "Coverage: some workflows never upload per-test results (XPU, s390x, TSan, torchtitan, most perf jobs), so their hours are structurally unattributable; only a dip on the newest day or two is upload lag.",
        "Determinism: days at least three days old are cached per query hash under agent_space/test-cost/cache and never refetched; younger days are refetched on every run, so only settled windows reproduce byte for byte. Owners reflect the checkout at report time.",
        f"GitHub cross-check: {report.verify_requested} jobs chosen by cityHash64(id) over the window are fetched from the GitHub Actions API and compared on runner label and wall seconds."
        if report.verify_requested
        else "GitHub cross-check: disabled for this report.",
    ]
    return (
        '<section id="methodology"><h2>Methodology</h2><div class="card"><ul class="method">'
        + "".join(f"<li>{item}</li>" for item in items)
        + f'</ul><p class="sub">Reproduce: <code>{esc(m.command)}</code>. Query hash {esc(m.query_hash)}, checkout {esc(m.checkout)}.</p></div></section>'
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
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>CI test cost {w.start} to {w.last}</title><style>{CSS}</style></head><body>"
        f'<header class="top"><nav><b>test-cost</b>{nav}</nav></header><main>'
        + "\n".join(sections)
        + f"</main><script>{JS}</script></body></html>\n"
    )
