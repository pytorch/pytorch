import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, render_template, request
from jinja2 import DictLoader

from torch.distributed.debug._store import get_world_size, tcpstore_client


def fetch_all(
    endpoint: str, args: str = ""
) -> tuple[list[str], Iterator[requests.Response]]:
    store = tcpstore_client()
    keys = [f"rank{r}" for r in range(get_world_size())]
    addrs = store.multi_get(keys)
    addrs = [f"{addr.decode()}/handler/{endpoint}?{args}" for addr in addrs]

    with ThreadPoolExecutor(max_workers=10) as executor:
        resps = executor.map(requests.post, addrs)

    return addrs, resps


def format_json(blob: str):
    parsed = json.loads(blob)
    return json.dumps(parsed, indent=2)


templates = {
    "base.html": """
<!doctype html>
<head>
    <title>{% block title %}{% endblock %} - PyTorch Distributed</title>
    <link rel="shortcut icon" type="image/x-icon" href="https://pytorch.org/favicon.ico?">

    <style>
        body {
            margin: 0;
            font-family:
                -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
                "Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji",
                "Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.5;
            color: #212529;
            text-align: left;
            background-color: #fff;
        }
        h1, h2, h2, h4, h5, h6, .h1, .h2, .h2, .h4, .h5, .h6 {
            margin-bottom: .5rem;
            font-weight: 500;
            line-height: 1.2;
        }
        nav {
            background-color: rgba(0, 0, 0, 0.17);
            padding: 10px;
            display: flex;
            align-items: center;
            padding: 16px;
            justify-content: flex-start;
        }
        nav h1 {
            display: inline-block;
            margin: 0;
        }
        nav a {
           margin: 0 8px;
        }
        section {
            max-width: 1280px;
            padding: 16px;
            margin: 0 auto;
        }
        pre {
            white-space: pre-wrap;
            max-width: 100%;
        }
    </style>
</head>

<nav>
    <h1>Torch Distributed Debug Server</h1>

    <a href="/">Home</a> <!--@lint-ignore-->
    <a href="/stacks">Python Stack Traces</a> <!--@lint-ignore-->
    <a href="/fr_trace">FlightRecorder</a> <!--@lint-ignore-->
    <a href="/fr_trace_nccl">FlightRecorder NCCL</a> <!--@lint-ignore-->
    <a href="/profile">torch profiler</a> <!--@lint-ignore-->
</nav>

<section class="content">
  {% block header %}{% endblock %}
  {% for message in get_flashed_messages() %}
    <div class="flash">{{ message }}</div>
  {% endfor %}
  {% block content %}{% endblock %}
</section>
    """,
    "index.html": """
{% extends "base.html" %}
{% block header %}
  <h1>{% block title %}Index{% endblock %}</h1>
{% endblock %}
{% block content %}
Hi
{% endblock %}
    """,
    "raw_resp.html": """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{title}}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ resp.text }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
    """,
    "json_resp.html": """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{ title }}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ format_json(resp.text) }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
    """,
    "profile.html": """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}torch.profiler{% endblock %}</h1>
{% endblock %}

{% block content %}
    <form action="/profile" method="get">
        <label for="duration">Duration (seconds):</label>
        <input type="number" id="duration" name="duration" value="{{ duration }}" min="1" max="60">
        <input type="submit" value="Submit">
    </form>

    <script>
    function stringToArrayBuffer(str) {
        const encoder = new TextEncoder();
        return encoder.encode(str).buffer;
    }
    async function openPerfetto(data) {
        const ui = window.open('https://ui.perfetto.dev/#!/');
        if (!ui) { alert('Popup blocked. Allow popups for this page and click again.'); return; }

        // Perfetto readiness handshake: PING until we receive PONG
        await new Promise((resolve, reject) => {
        const onMsg = (e) => {
            if (e.source === ui && e.data === 'PONG') {
            window.removeEventListener('message', onMsg);
            clearInterval(pinger);
            resolve();
            }
        };
        window.addEventListener('message', onMsg);
        const pinger = setInterval(() => { try { ui.postMessage('PING', '*'); } catch (_e) {} }, 250);
        setTimeout(() => { clearInterval(pinger); window.removeEventListener('message', onMsg); reject(); }, 20000);
        }).catch(() => { alert('Perfetto UI did not respond. Try again.'); return; });

        ui.postMessage({
        perfetto: {
            buffer: stringToArrayBuffer(JSON.stringify(data)),
            title: "torch profiler",
            fileName: "trace.json",
        }
        }, '*');
    }
    </script>

    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <script>
            function run{{ i }}() {
                var data = {{ resp.text | safe }};
                openPerfetto(data);
            }
            </script>

            <button onclick="run{{ i }}()">View {{ i }}</button>
        {% endif %}
    {% endfor %}
{% endblock %}
    """,
}

app = Flask(__name__)
app.jinja_loader = DictLoader(templates)
app.jinja_env.globals.update(
    zip=zip,
    format_json=format_json,
    enumerate=enumerate,
)


@app.route("/")
def _index_handler():
    return render_template("index.html")


@app.route("/stacks")
def _stacks_handler():
    addrs, resps = fetch_all("dump_traceback")
    return render_template("raw_resp.html", title="Stacks", addrs=addrs, resps=resps)


@app.route("/fr_trace")
def _fr_trace_handler():
    addrs, resps = fetch_all("fr_trace_json")

    return render_template(
        "json_resp.html",
        title="FlightRecorder",
        addrs=addrs,
        resps=resps,
    )


@app.route("/fr_trace_nccl")
def _fr_trace_nccl_handler():
    addrs, resps = fetch_all("dump_nccl_trace_json", "onlyactive=true")

    return render_template(
        "json_resp.html",
        title="FlightRecorder NCCL",
        addrs=addrs,
        resps=resps,
    )


@app.route("/profile")
def _profiler_handler():
    duration = request.args.get("duration", default=1.0, type=float)

    addrs, resps = fetch_all("torch_profile", f"duration={duration}")

    return render_template("profile.html", addrs=addrs, resps=resps)


def main(port: int) -> None:
    app.run(host="::", port=port)
