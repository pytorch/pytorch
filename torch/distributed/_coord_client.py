"""
Synchronous client for the torchmux coordinator server.

Exposes :class:`CoordClient` with ``register``, ``wait_for_turn``,
``prepare``, ``release_gpu``, ``release_baton``, ``acquire_baton``,
and ``done``.

Threading model: an asyncio event loop runs in a daemon thread.
Foreground (torch) code calls synchronous methods which dispatch to
the loop via ``asyncio.run_coroutine_threadsafe`` and block.
"""

import asyncio
import importlib.util
import os
import threading
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "coord_client",
    os.path.join(os.path.dirname(__file__), "../../checkpoint/coord_client.py"),
)
_base = importlib.util.module_from_spec(_spec)

# The base module does `from protocol import ...` which needs protocol.py
# on sys.path. Temporarily add checkpoint/ so its bare imports resolve.
import sys

_ckpt_dir = os.path.join(os.path.dirname(__file__), "../../checkpoint")
sys.path.insert(0, _ckpt_dir)
try:
    _spec.loader.exec_module(_base)
finally:
    sys.path.remove(_ckpt_dir)

CoordClientError = _base.CoordClientError
PeerGone = _base.PeerGone
NoPeers = _base.NoPeers

from torch.distributed._coord_protocol import (
    ERR_NO_PEERS,
    ERR_PEER_GONE,
    OP_ACQUIRE_BATON,
    OP_DONE,
    OP_PREPARE,
    OP_REGISTER,
    OP_RELEASE_BATON,
    OP_RELEASE_GPU,
    OP_WAIT_FOR_TURN,
    read_message,
    write_message,
)


# ---- Tensor serialization ----


def _serialize_tensor(tensor) -> tuple[dict, bytes]:
    import ctypes

    import torch

    cpu_contig = tensor.detach().contiguous().cpu()
    nbytes = cpu_contig.numel() * cpu_contig.element_size()
    ptr = cpu_contig.untyped_storage().data_ptr()
    payload = bytes((ctypes.c_char * nbytes).from_address(ptr))
    header = {
        "shape": list(cpu_contig.shape),
        "dtype": str(cpu_contig.dtype).removeprefix("torch."),
        "nbytes": nbytes,
    }
    return header, payload


def _deserialize_tensor(header: dict, payload: bytes):
    import torch

    dtype = getattr(torch, header["dtype"])
    shape = tuple(header["shape"])
    numel = 1
    for d in shape:
        numel *= d
    flat = torch.frombuffer(bytearray(payload), dtype=dtype, count=numel).clone()
    return flat.reshape(shape)


# ---- Async engine ----


class _Engine:
    def __init__(self, addr: str):
        self.addr = addr
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        buf_limit = 16 * 1024 * 1024
        if self.addr.startswith("uds:"):
            path = self.addr[4:]
            self.reader, self.writer = await asyncio.open_unix_connection(
                path, limit=buf_limit
            )
        elif self.addr.startswith("tcp:"):
            host, _, port = self.addr[4:].rpartition(":")
            self.reader, self.writer = await asyncio.open_connection(
                host, int(port), limit=buf_limit
            )
        else:
            raise ValueError(f"unrecognized addr {self.addr!r}")

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None

    async def rpc(
        self, header: dict, payload: bytes = b""
    ) -> tuple[dict, bytes]:
        async with self._lock:
            if self.writer is None:
                raise CoordClientError("not connected")
            await write_message(self.writer, header, payload)
            try:
                resp_hdr, resp_payload = await read_message(self.reader)
            except asyncio.IncompleteReadError as e:
                raise PeerGone("coordinator closed the connection") from e
            if not resp_hdr.get("ok"):
                err = resp_hdr.get("error") or "request failed"
                detail = resp_hdr.get("detail")
                if err == ERR_NO_PEERS:
                    raise NoPeers(err)
                if err == ERR_PEER_GONE:
                    raise PeerGone(err)
                raise CoordClientError(err)
            return resp_hdr, resp_payload


# ---- CoordClient ----


class CoordClient:
    def __init__(self, addr: str | None = None):
        self.addr = addr or os.environ.get("COORD_ADDR")
        if not self.addr:
            raise CoordClientError(
                "addr not provided and COORD_ADDR env not set"
            )
        self._engine = _Engine(self.addr)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._rank: int | None = None
        self._start_loop()
        self._run_coro(self._engine.connect())

    def _start_loop(self) -> None:
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()

        self._thread = threading.Thread(
            target=run_loop, name="CoordClient", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    def _run_coro(self, coro, *, timeout: float | None = None):
        if self._loop is None:
            raise CoordClientError("loop not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except KeyboardInterrupt:
            self._loop.call_soon_threadsafe(fut.cancel)
            raise

    # ---- Public API ----

    def register(self, rank: int, gpu_id: int = 0) -> None:
        self._run_coro(
            self._engine.rpc(
                {"op": OP_REGISTER, "rank": rank, "gpu_id": gpu_id}
            )
        )
        self._rank = rank

    def wait_for_turn(self) -> None:
        self._run_coro(self._engine.rpc({"op": OP_WAIT_FOR_TURN}))

    def prepare(
        self,
        send: dict[tuple[int, ...], Any],
        recv: tuple[int, ...] | list[int],
    ) -> dict[int, Any] | None:
        """Declare this rank's collective step.

        Returns dict[src_rank -> tensor | None] on fast path, or None
        if the caller must follow up with :meth:`release_gpu`.
        """
        import torch

        send_entries = []
        payloads: list[bytes] = []
        for dsts, tensor in send.items():
            if not isinstance(dsts, tuple):
                raise TypeError("send keys must be tuples of destination ranks")
            if tensor is None:
                send_entries.append({"dsts": list(dsts), "tensor": None})
            elif isinstance(tensor, torch.Tensor):
                hdr, body = _serialize_tensor(tensor)
                send_entries.append({"dsts": list(dsts), "tensor": hdr})
                payloads.append(body)
            else:
                raise TypeError(
                    f"send values must be Tensor or None, "
                    f"got {type(tensor).__name__}"
                )

        req = {"op": OP_PREPARE, "send": send_entries, "recv": list(recv)}
        resp_hdr, resp_payload = self._run_coro(
            self._engine.rpc(req, b"".join(payloads))
        )
        if resp_hdr.get("block"):
            return None
        return self._decode_recv(resp_hdr.get("recv") or [], resp_payload)

    def release_gpu(self) -> dict[int, Any]:
        """Send release_gpu to coordinator. Caller is responsible for
        checkpoint/restore around this call.

        Blocks until the coordinator schedules this rank back and
        returns the recv data from the pending prepare.
        """
        resp_hdr, resp_payload = self._run_coro(
            self._engine.rpc({"op": OP_RELEASE_GPU})
        )
        return self._decode_recv(resp_hdr.get("recv") or [], resp_payload)

    def release_baton(self) -> None:
        """Release this rank's GPU baton without a pending prepare.
        Caller is responsible for checkpoint before calling."""
        self._run_coro(self._engine.rpc({"op": OP_RELEASE_BATON}))

    def acquire_baton(self) -> None:
        """Block until the coordinator grants this rank's GPU baton.
        Caller is responsible for restore after this returns."""
        self._run_coro(self._engine.rpc({"op": OP_ACQUIRE_BATON}))

    def done(self) -> None:
        try:
            self._run_coro(self._engine.rpc({"op": OP_DONE}))
        finally:
            self.close()

    def close(self) -> None:
        if self._loop is None:
            return
        try:
            self._run_coro(self._engine.close(), timeout=2.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None

    # ---- Helpers ----

    def _decode_recv(
        self, entries: list[dict], payload: bytes
    ) -> dict[int, Any]:
        out: dict[int, Any] = {}
        offset = 0
        for entry in entries:
            src = entry["src"]
            hdr = entry.get("tensor")
            if hdr is None:
                out[src] = None
            else:
                nbytes = hdr["nbytes"]
                body = payload[offset : offset + nbytes]
                offset += nbytes
                out[src] = _deserialize_tensor(hdr, body)
        return out
