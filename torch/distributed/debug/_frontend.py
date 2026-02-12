import asyncio
import json
import logging
import socket
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from jinja2 import DictLoader, Environment
from tabulate import tabulate

from torch.distributed.debug._store import get_world_size, tcpstore_client
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.config_manager import JobConfig
from torch.distributed.flight_recorder.components.types import (
    Collective,
    Group,
    Membership,
    NCCLCall,
)


logger: logging.Logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Response:
    status_code: int
    text: str

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self):
        return json.loads(self.text)


def fetch_thread_pool(urls: list[str]) -> Iterable[Response]:
    # late import for optional dependency
    import requests

    max_workers = 20

    def get(url: str) -> Response:
        resp = requests.post(url)
        return Response(resp.status_code, resp.text)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resps = executor.map(get, urls)

    return resps


def fetch_aiohttp(urls: list[str]) -> Iterable[Response]:
    # late import for optional dependency
    # pyrefly: ignore [missing-import]
    import aiohttp

    async def fetch(session: aiohttp.ClientSession, url: str) -> Response:
        async with session.post(url) as resp:
            text = await resp.text()
            return Response(resp.status, text)

    async def gather(urls: list[str]) -> Iterable[Response]:
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(*[fetch(session, url) for url in urls])

    return asyncio.run(gather(urls))


def fetch_all(endpoint: str, args: str = "") -> tuple[list[str], Iterable[Response]]:
    store = tcpstore_client()
    keys = [f"rank{r}" for r in range(get_world_size())]
    addrs = store.multi_get(keys)
    addrs = [f"{addr.decode()}/handler/{endpoint}?{args}" for addr in addrs]

    try:
        resps = fetch_aiohttp(addrs)
    except ImportError:
        resps = fetch_thread_pool(addrs)

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
    <a href="/pyspy_dump">py-spy Stacks</a> <!--@lint-ignore-->
    <a href="/fr_trace">FlightRecorder CPU</a> <!--@lint-ignore-->
    <a href="/fr_trace_json">(JSON)</a> <!--@lint-ignore-->
    <a href="/fr_trace_nccl">FlightRecorder NCCL</a> <!--@lint-ignore-->
    <a href="/fr_trace_nccl_json">(JSON)</a> <!--@lint-ignore-->
    <a href="/profile">torch profiler</a> <!--@lint-ignore-->
    <a href="/wait_counters">Wait Counters</a> <!--@lint-ignore-->
    <a href="/tcpstore">TCPStore</a> <!--@lint-ignore-->
</nav>

<section class="content">
  {% block header %}{% endblock %}
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
    <form action="" method="get">
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
    "tcpstore.html": """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}TCPStore Keys{% endblock %}</h1>
{% endblock %}
{% block content %}
    <pre>
    {% for k, v in zip(keys, values) -%}
{{ k }}: {{ v | truncate(100) }}
    {% endfor %}
    </pre>
{% endblock %}
    """,
    "fr_trace.html": """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{ title }}{% endblock %}</h1>
{% endblock %}
{% block content %}
    <h2>Groups</h2>
    {{ groups | safe }}
    <h2>Memberships</h2>
    {{ memberships | safe }}
    <h2>Collectives</h2>
    {{ collectives | safe }}
    <h2>NCCL Calls</h2>
    {{ ncclcalls | safe }}
{% endblock %}
    """,
    "pyspy_dump.html": """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}py-spy Stack Traces{% endblock %}</h1>
{% endblock %}
{% block content %}
    <form action="" method="get">
        <input type="checkbox" id="native" name="native" value="1"/>
        <label for="native">Native</label>
        <input type="checkbox" id="subprocesses" name="subprocesses" value="1"/>
        <label for="subprocesses">Subprocesses</label>
        <input type="submit" value="Submit">
    </form>

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
}


class _IPv6HTTPServer(ThreadingHTTPServer):
    address_family: socket.AddressFamily = socket.AF_INET6  # pyre-ignore
    request_queue_size: int = 1024


class HTTPRequestHandler(BaseHTTPRequestHandler):
    frontend: "FrontendServer"

    def log_message(self, format, *args):
        logger.info(
            "%s %s",
            self.client_address[0],
            format % args,
        )

    def do_GET(self):
        self.frontend._handle_request(self)

    def get_path(self) -> str:
        return urlparse(self.path).path

    def get_query(self) -> dict[str, list[str]]:
        return parse_qs(self.get_raw_query())

    def get_raw_query(self) -> str:
        return urlparse(self.path).query

    def get_query_arg(
        self, name: str, default: object = None, type: type = str
    ) -> object:
        query = self.get_query()
        if name not in query:
            return default
        return type(query[name][0])


class FrontendServer:
    def __init__(self, port: int):
        # Setup templates
        loader = DictLoader(templates)
        self._jinja_env = Environment(loader=loader, enable_async=True)
        self._jinja_env.globals.update(
            zip=zip,
            format_json=format_json,
            enumerate=enumerate,
        )

        # Create routes
        self._routes = {
            "/": self._handle_index,
            "/stacks": self._handle_stacks,
            "/pyspy_dump": self._handle_pyspy_dump,
            "/fr_trace": self._handle_fr_trace,
            "/fr_trace_json": self._handle_fr_trace_json,
            "/fr_trace_nccl": self._handle_fr_trace_nccl,
            "/fr_trace_nccl_json": self._handle_fr_trace_nccl_json,
            "/profile": self._handle_profiler,
            "/wait_counters": self._handle_wait_counters,
            "/tcpstore": self._handle_tcpstore,
        }

        # Create HTTP server
        RequestHandlerClass = type(
            "HTTPRequestHandler",
            (HTTPRequestHandler,),
            {"frontend": self},
        )

        server_address = ("", port)
        self._server = _IPv6HTTPServer(server_address, RequestHandlerClass)

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
            name="distributed.debug.FrontendServer",
        )
        self._thread.start()

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception:
            logger.exception("got exception in frontend server")

    def join(self) -> None:
        self._thread.join()

    def _handle_request(self, req: HTTPRequestHandler) -> None:
        path = req.get_path()
        if path not in self._routes:
            req.send_error(404, f"Handler not found: {path}")
            return

        handler = self._routes[path]
        try:
            resp = handler(req)
        # Catch SystemExit to not crash when FlightRecorder errors.
        except (Exception, SystemExit) as e:
            logger.exception(
                "Exception in frontend server when handling %s",
                path,
            )
            req.send_error(500, f"Exception: {repr(e)}")
            return

        req.send_response(200)
        req.send_header("Content-type", "text/html")
        req.end_headers()
        req.wfile.write(resp)

    def _render_template(self, template: str, **kwargs: object) -> bytes:
        return self._jinja_env.get_template(template).render(**kwargs).encode()

    def _handle_index(self, req: HTTPRequestHandler) -> bytes:
        return self._render_template("index.html")

    def _handle_stacks(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("dump_traceback")
        return self._render_template(
            "raw_resp.html", title="Stacks", addrs=addrs, resps=resps
        )

    def _handle_pyspy_dump(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("pyspy_dump", req.get_raw_query())
        return self._render_template(
            "pyspy_dump.html",
            addrs=addrs,
            resps=resps,
        )

    def _render_fr_trace(self, addrs: list[str], resps: list[Response]) -> bytes:
        config = JobConfig()

        args = config.parse_args(args=[])
        args.allow_incomplete_ranks = True
        args.verbose = True

        details = {}
        for rank, resp in enumerate(resps):
            resp.raise_for_status()
            dump = {
                "rank": rank,
                "host_name": addrs[rank],
                **resp.json(),
            }
            if "entries" not in dump:
                dump["entries"] = []
            details[f"rank{rank}.json"] = dump

        version = next(iter(details.values()))["version"]

        db = build_db(details, args, version)

        return self._render_template(
            "fr_trace.html",
            title="FlightRecorder",
            groups=tabulate(db.groups, headers=Group._fields, tablefmt="html"),
            memberships=tabulate(
                db.memberships, headers=Membership._fields, tablefmt="html"
            ),
            collectives=tabulate(
                db.collectives, headers=Collective._fields, tablefmt="html"
            ),
            ncclcalls=tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="html"),
        )

    def _handle_fr_trace(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("fr_trace_json")

        return self._render_fr_trace(addrs, list(resps))

    def _handle_fr_trace_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("fr_trace_json")

        return self._render_template(
            "json_resp.html",
            title="FlightRecorder",
            addrs=addrs,
            resps=resps,
        )

    def _handle_fr_trace_nccl(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("dump_nccl_trace_json", "onlyactive=true")

        return self._render_fr_trace(addrs, list(resps))

    def _handle_fr_trace_nccl_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("dump_nccl_trace_json", "onlyactive=true")

        return self._render_template(
            "json_resp.html",
            title="FlightRecorder NCCL",
            addrs=addrs,
            resps=resps,
        )

    def _handle_profiler(self, req: HTTPRequestHandler) -> bytes:
        duration = req.get_query_arg("duration", default=1.0, type=float)

        addrs, resps = fetch_all("torch_profile", f"duration={duration}")

        return self._render_template("profile.html", addrs=addrs, resps=resps)

    def _handle_wait_counters(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("wait_counter_values")
        return self._render_template(
            "json_resp.html", title="Wait Counters", addrs=addrs, resps=resps
        )

    def _handle_tcpstore(self, req: HTTPRequestHandler) -> bytes:
        store = tcpstore_client(prefix="")
        keys = store.list_keys()
        keys.sort()
        values = [repr(v) for v in store.multi_get(keys)]
        return self._render_template("tcpstore.html", keys=keys, values=values)


def main(port: int) -> None:
    logger.setLevel(logging.INFO)

    server = FrontendServer(port=port)
    logger.info("Frontend server started on port %d", server._server.server_port)
    server.join()
