from __future__ import annotations

from typing import TYPE_CHECKING

from tabulate import tabulate

from torch.distributed.debug._frontend import (
    DebugHandler,
    fetch_all,
    format_json,
    NavLink,
    Response,
    Route,
)
from torch.distributed.debug._store import tcpstore_client
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.config_manager import JobConfig
from torch.distributed.flight_recorder.components.types import (
    Collective,
    Database,
    Group,
    Membership,
    NCCLCall,
)


if TYPE_CHECKING:
    from torch.distributed.debug._frontend import FrontendServer, HTTPRequestHandler


# ---------------------------------------------------------------------------
# Handler-specific templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
  <h1>{% block title %}Index{% endblock %}</h1>
{% endblock %}
{% block content %}
Hi
{% endblock %}
    """

PYSPY_DUMP_TEMPLATE = """
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
    """

FR_TRACE_TEMPLATE = """
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
    """

PROFILE_TEMPLATE = """
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
    """

TCPSTORE_TEMPLATE = """
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
    """


# ---------------------------------------------------------------------------
# Handler classes
# ---------------------------------------------------------------------------


class IndexHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/", "Home")]

    def templates(self) -> dict[str, str]:
        return {"index.html": INDEX_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        return req.frontend.render_template("index.html")


class StacksHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/stacks", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/stacks", "Python Stack Traces")]

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("dump_traceback")
        return req.frontend.render_template(
            "raw_resp.html", title="Stacks", addrs=addrs, resps=resps
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("dump_traceback")
        parts: list[str] = []
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "stacks"


class PySpyHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/pyspy_dump", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/pyspy_dump", "py-spy Stacks")]

    def templates(self) -> dict[str, str]:
        return {"pyspy_dump.html": PYSPY_DUMP_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("pyspy_dump", req.get_raw_query())
        return req.frontend.render_template(
            "pyspy_dump.html",
            addrs=addrs,
            resps=resps,
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("pyspy_dump")
        parts: list[str] = []
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "pyspy_dump"


class FlightRecorderHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [
            Route("/fr_trace", self._handle_fr_trace),
            Route("/fr_trace_json", self._handle_fr_trace_json),
            Route("/fr_trace_nccl", self._handle_fr_trace_nccl),
            Route("/fr_trace_nccl_json", self._handle_fr_trace_nccl_json),
        ]

    def nav_links(self) -> list[NavLink]:
        return [
            NavLink("/fr_trace", "FlightRecorder CPU"),
            NavLink("/fr_trace_json", "(JSON)"),
            NavLink("/fr_trace_nccl", "FlightRecorder NCCL"),
            NavLink("/fr_trace_nccl_json", "(JSON)"),
        ]

    def templates(self) -> dict[str, str]:
        return {"fr_trace.html": FR_TRACE_TEMPLATE}

    @staticmethod
    def _build_db(addrs: list[str], resps: list[Response]) -> Database:
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
        # pyrefly: ignore [bad-argument-type]
        return build_db(details, args, version)

    def _render_tables(
        self, server: FrontendServer, addrs: list[str], resps: list[Response]
    ) -> bytes:
        db = self._build_db(addrs, resps)
        return server.render_template(
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
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_fr_trace_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("fr_trace_json")
        return req.frontend.render_template(
            "json_resp.html",
            title="FlightRecorder",
            addrs=addrs,
            resps=resps,
        )

    def _handle_fr_trace_nccl(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("dump_nccl_trace_json", "onlyactive=true")
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_fr_trace_nccl_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("dump_nccl_trace_json", "onlyactive=true")
        return req.frontend.render_template(
            "json_resp.html",
            title="FlightRecorder NCCL",
            addrs=addrs,
            resps=resps,
        )

    def dump(self) -> str | None:
        parts = []

        addrs, resps = fetch_all("fr_trace_json")
        db = self._build_db(addrs, list(resps))
        parts.extend(
            [
                "=== FR Trace ===",
                "--- Groups ---",
                tabulate(db.groups, headers=Group._fields, tablefmt="plain"),
                "--- Memberships ---",
                tabulate(db.memberships, headers=Membership._fields, tablefmt="plain"),
                "--- Collectives ---",
                tabulate(db.collectives, headers=Collective._fields, tablefmt="plain"),
                "--- NCCL Calls ---",
                tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="plain"),
            ]
        )

        try:
            nccl_addrs, nccl_resps = fetch_all(
                "dump_nccl_trace_json", "onlyactive=true"
            )
            nccl_db = self._build_db(nccl_addrs, list(nccl_resps))
            parts.extend(
                [
                    "",
                    "=== FR Trace NCCL ===",
                    "--- Groups ---",
                    tabulate(nccl_db.groups, headers=Group._fields, tablefmt="plain"),
                    "--- Memberships ---",
                    tabulate(
                        nccl_db.memberships,
                        headers=Membership._fields,
                        tablefmt="plain",
                    ),
                    "--- Collectives ---",
                    tabulate(
                        nccl_db.collectives,
                        headers=Collective._fields,
                        tablefmt="plain",
                    ),
                    "--- NCCL Calls ---",
                    tabulate(
                        nccl_db.ncclcalls,
                        headers=NCCLCall._fields,
                        tablefmt="plain",
                    ),
                ]
            )
        except Exception:
            parts.append("\n=== FR Trace NCCL ===\nFailed to fetch NCCL trace")

        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "fr_trace"


class ProfilerHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/profile", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/profile", "torch profiler")]

    def templates(self) -> dict[str, str]:
        return {"profile.html": PROFILE_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        duration = req.get_query_arg("duration", default=1.0, type=float)
        addrs, resps = fetch_all("torch_profile", f"duration={duration}")
        return req.frontend.render_template("profile.html", addrs=addrs, resps=resps)


class WaitCountersHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/wait_counters", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/wait_counters", "Wait Counters")]

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("wait_counter_values")
        return req.frontend.render_template(
            "json_resp.html", title="Wait Counters", addrs=addrs, resps=resps
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("wait_counter_values")
        parts: list[str] = []
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            if resp.status_code == 200:
                parts.append(format_json(resp.text))
            else:
                parts.append(f"Error: {resp.status_code}")
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "wait_counters"


class TCPStoreHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/tcpstore", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/tcpstore", "TCPStore")]

    def templates(self) -> dict[str, str]:
        return {"tcpstore.html": TCPSTORE_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        store = tcpstore_client(prefix="")
        keys = store.list_keys()
        keys.sort()
        values = [repr(v) for v in store.multi_get(keys)]
        return req.frontend.render_template("tcpstore.html", keys=keys, values=values)

    def dump(self) -> str | None:
        store = tcpstore_client(prefix="")
        keys = store.list_keys()
        keys.sort()
        values = [repr(v) for v in store.multi_get(keys)]
        parts = [f"{k}: {v}" for k, v in zip(keys, values)]
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "tcpstore"


class TorchCommsFlightRecorderHandler(DebugHandler):
    """Handler for TorchComms FlightRecorder trace data."""

    def routes(self) -> list[Route]:
        return [
            Route("/torchcomms_fr_trace", self._handle_torchcomms_fr_trace),
            Route("/torchcomms_fr_trace_json", self._handle_torchcomms_fr_trace_json),
        ]

    def nav_links(self) -> list[NavLink]:
        return [
            NavLink("/torchcomms_fr_trace", "TorchComms FR"),
            NavLink("/torchcomms_fr_trace_json", "(JSON)"),
        ]

    def templates(self) -> dict[str, str]:
        return {"fr_trace.html": FR_TRACE_TEMPLATE}

    def _handle_torchcomms_fr_trace(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("torchcomms_fr_trace_json", "onlyactive=true")
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_torchcomms_fr_trace_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("torchcomms_fr_trace_json", "onlyactive=true")
        return req.frontend.render_template(
            "json_resp.html",
            title="TorchComms FlightRecorder",
            addrs=addrs,
            resps=resps,
        )

    def _render_tables(
        self, server: FrontendServer, addrs: list[str], resps: list[Response]
    ) -> bytes:
        db = FlightRecorderHandler._build_db(addrs, resps)
        return server.render_template(
            "fr_trace.html",
            title="TorchComms FlightRecorder",
            groups=tabulate(db.groups, headers=Group._fields, tablefmt="html"),
            memberships=tabulate(
                db.memberships, headers=Membership._fields, tablefmt="html"
            ),
            collectives=tabulate(
                db.collectives, headers=Collective._fields, tablefmt="html"
            ),
            ncclcalls=tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="html"),
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("torchcomms_fr_trace_json")
        db = FlightRecorderHandler._build_db(addrs, list(resps))
        parts = [
            "=== TorchComms FR Trace ===",
            "--- Groups ---",
            tabulate(db.groups, headers=Group._fields, tablefmt="plain"),
            "--- Memberships ---",
            tabulate(db.memberships, headers=Membership._fields, tablefmt="plain"),
            "--- Collectives ---",
            tabulate(db.collectives, headers=Collective._fields, tablefmt="plain"),
            "--- NCCL Calls ---",
            tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="plain"),
        ]
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "torchcomms_fr_trace"


def default_handlers() -> list[DebugHandler]:
    return [
        IndexHandler(),
        StacksHandler(),
        PySpyHandler(),
        FlightRecorderHandler(),
        TorchCommsFlightRecorderHandler(),
        ProfilerHandler(),
        WaitCountersHandler(),
        TCPStoreHandler(),
    ]
