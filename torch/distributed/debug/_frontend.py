import asyncio
import json
import logging
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from jinja2 import DictLoader, Environment

from torch.distributed.debug._store import get_world_size, tcpstore_client


logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Response:
    status_code: int
    text: str

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self):
        return json.loads(self.text)


@dataclass(slots=True)
class NavLink:
    path: str
    label: str


@dataclass(slots=True)
class Route:
    path: str
    handler: Callable[["HTTPRequestHandler"], bytes]


class DebugHandler(ABC):
    @abstractmethod
    def routes(self) -> list[Route]: ...

    @abstractmethod
    def nav_links(self) -> list[NavLink]: ...

    def templates(self) -> dict[str, str]:
        return {}

    def dump(self) -> str | None:
        return None

    def dump_filename(self) -> str:
        return type(self).__name__.lower()


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Template constants
# ---------------------------------------------------------------------------


BASE_TEMPLATE = """
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

    {{ nav_links | safe }}
</nav>

<section class="content">
  {% block header %}{% endblock %}
  {% block content %}{% endblock %}
</section>
    """

RAW_RESP_TEMPLATE = """
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
    """

JSON_RESP_TEMPLATE = """
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
    """


# ---------------------------------------------------------------------------
# PeriodicDumper
# ---------------------------------------------------------------------------


class PeriodicDumper:
    def __init__(
        self,
        handlers: list[DebugHandler],
        output_dir: str,
        interval_seconds: float = 60.0,
    ) -> None:
        self._handlers = handlers
        self._output_dir = output_dir
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        os.makedirs(self._output_dir, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="distributed.debug.PeriodicDumper",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for handler in self._handlers:
                try:
                    content = handler.dump()
                except Exception:
                    logger.exception("Failed to dump %s", handler.dump_filename())
                    continue
                if content is None:
                    continue
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{handler.dump_filename()}_{timestamp}.txt"
                path = os.path.join(self._output_dir, filename)
                try:
                    with open(path, "w") as f:
                        f.write(content)
                except Exception:
                    logger.exception("Failed to write dump to %s", path)
            self._stop_event.wait(self._interval_seconds)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


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
    def __init__(
        self,
        port: int,
        handlers: list[DebugHandler] | None = None,
    ):
        if handlers is None:
            from torch.distributed.debug._debug_handlers import default_handlers

            handlers = default_handlers()

        # Build nav HTML from handlers
        nav_html = "\n".join(
            f'    <a href="{link.path}">{link.label}</a> <!--@lint-ignore-->'
            for handler in handlers
            for link in handler.nav_links()
        )

        # Merge all handler templates + shared templates
        all_templates: dict[str, str] = {
            "base.html": BASE_TEMPLATE,
            "raw_resp.html": RAW_RESP_TEMPLATE,
            "json_resp.html": JSON_RESP_TEMPLATE,
        }
        for handler in handlers:
            all_templates.update(handler.templates())

        loader = DictLoader(all_templates)
        self._jinja_env = Environment(loader=loader, enable_async=True)
        self._jinja_env.globals.update(
            zip=zip,
            format_json=format_json,
            enumerate=enumerate,
            nav_links=nav_html,
        )

        # Build route table from handlers
        self._routes: dict[str, Callable[[HTTPRequestHandler], bytes]] = {}
        for handler in handlers:
            for route in handler.routes():
                self._routes[route.path] = route.handler

        self._handlers = handlers

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

    def render_template(self, template: str, **kwargs: object) -> bytes:
        return self._jinja_env.get_template(template).render(**kwargs).encode()


def main(
    port: int,
    dump_dir: str | None,
    dump_interval: float,
    handlers: list[DebugHandler],
    enabled_dumps: set[str],
) -> None:
    logger.setLevel(logging.INFO)

    server = FrontendServer(port=port, handlers=handlers)
    logger.info("Frontend server started on port %d", server._server.server_port)

    dumper: PeriodicDumper | None = None
    if dump_dir is not None:
        dumper = PeriodicDumper(
            [
                handler
                for handler in handlers
                if handler.dump_filename() in enabled_dumps
            ],
            dump_dir,
            dump_interval,
        )
        dumper.start()
        logger.info(
            "Periodic dumper started, writing to %s every %.0fs",
            dump_dir,
            dump_interval,
        )

    try:
        server.join()
    finally:
        if dumper is not None:
            dumper.stop()
