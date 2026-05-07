"""Synchronous client for the coordinator server.

Exposes:

* :class:`CoordClient` — primary API. ``register(rank)``,
  ``prepare(send, recv)``, ``release_gpu()``, ``done()``.

Addressing: pass ``addr`` as ``"uds:/path/to/sock"`` or ``"tcp:host:port"``.
Also reads the ``COORD_ADDR`` env var as the default.

Threading model: the client runs an asyncio event loop in a daemon thread.
Foreground (torch) code calls synchronous methods which dispatch to the
loop via ``asyncio.run_coroutine_threadsafe`` and block on ``.result()``.
No ``asyncio.Queue`` across threads.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    import torch

from .protocol import (
    ERR_MISMATCH,
    ERR_PEER_GONE,
    JsonDict,
    OP_DONE,
    OP_PREPARE,
    OP_REGISTER,
    OP_RELEASE_GPU,
    read_message,
    RecvEntry,
    SendEntry,
    TensorHeader,
    write_message,
)


class CoordClientError(RuntimeError):
    pass


class PeerGone(CoordClientError):
    """Raised if a peer we were waiting on disappears."""


class CollectiveMismatch(CoordClientError):
    """Raised when this rank's prepare is inconsistent with a peer's."""


# ---- Tensor serialization (raw-bytes fast path) ----


def _serialize_tensor(tensor: object) -> tuple[TensorHeader, bytes]:
    """Encode a torch.Tensor as (header, raw_bytes). Caller must have live
    CUDA; tensor is materialized to CPU contiguous. Uses untyped storage
    bytes so bfloat16 and torch-only dtypes roundtrip cleanly."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")
    cpu_contig = tensor.detach().contiguous().cpu()
    nbytes = cpu_contig.numel() * cpu_contig.element_size()
    storage = cpu_contig.untyped_storage()
    payload = bytes(storage[:nbytes])
    header = {
        "shape": list(cpu_contig.shape),
        "dtype": str(cpu_contig.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "nbytes": nbytes,
    }
    return header, payload


def _deserialize_tensor(header: TensorHeader, payload: bytes) -> torch.Tensor:
    import torch

    dtype = getattr(torch, header["dtype"])
    shape = tuple(header["shape"])
    buf = bytearray(payload)
    numel = 1
    for d in shape:
        numel *= d
    flat = torch.frombuffer(buf, dtype=dtype, count=numel).clone()
    t = flat.reshape(shape)
    target = header.get("device", "cpu")
    if target != "cpu":
        t = t.to(target)
    return t


def _recv_entries(header: JsonDict) -> list[RecvEntry]:
    return cast(list[RecvEntry], header.get("recv") or [])


# ---- Async engine (runs in background thread) ----


class _Engine:
    """Owns the event loop and the server connection. Request-response is
    strictly sequential (one in flight at a time)."""

    def __init__(self, addr: str) -> None:
        self.addr = addr
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self.addr.startswith("uds:"):
            path = self.addr[4:]
            self.reader, self.writer = await asyncio.open_unix_connection(path)
        elif self.addr.startswith("tcp:"):
            host, _, port = self.addr[4:].rpartition(":")
            self.reader, self.writer = await asyncio.open_connection(host, int(port))
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
        self, header: JsonDict, payload: bytes = b""
    ) -> tuple[JsonDict, bytes]:
        async with self._lock:
            if self.writer is None or self.reader is None:
                raise CoordClientError("not connected")
            await write_message(self.writer, header, payload)
            try:
                resp_hdr, resp_payload = await read_message(self.reader)
            except asyncio.IncompleteReadError as e:
                raise PeerGone("coordinator closed the connection") from e
            if not resp_hdr.get("ok"):
                err = str(resp_hdr.get("error") or "request failed")
                detail = resp_hdr.get("detail")
                if err == ERR_PEER_GONE:
                    raise PeerGone(err)
                if err == ERR_MISMATCH:
                    raise CollectiveMismatch(str(detail or err))
                raise CoordClientError(err)
            return resp_hdr, resp_payload


# ---- CoordClient ----


class CoordClient:
    def __init__(self, addr: str | None = None) -> None:
        self.addr = addr or os.environ.get("COORD_ADDR")
        if not self.addr:
            raise CoordClientError("addr not provided and COORD_ADDR env not set")
        self._engine = _Engine(self.addr)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._rank: int | None = None
        self._start_loop()
        self._run_coro(self._engine.connect())

    # ---- Loop plumbing ----

    def _start_loop(self) -> None:
        def run_loop() -> None:
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

    def _run_coro(self, coro: Any, *, timeout: float | None = None) -> Any:
        """
        We expect that most calls will run without a timeout - they can wait
        a REALLY long time for the coordinator to get back to them so any
        timeout should be global.
        """
        if self._loop is None:
            coro.close()
            raise CoordClientError("loop not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except KeyboardInterrupt:
            self._loop.call_soon_threadsafe(fut.cancel)
            raise

    # ---- Public API ----

    def register(self, rank: int) -> None:
        """Register with the coordinator.

        Rank 0 returns immediately once registered.  Later ranks block until
        their rank-ordered turn and the active rank yields.
        """
        if not isinstance(rank, int):
            raise TypeError(f"rank must be int, got {type(rank).__name__}")
        msg: JsonDict = {"op": OP_REGISTER, "rank": rank}
        self._run_coro(self._engine.rpc(msg))
        self._rank = rank

    def prepare(
        self,
        send: dict[tuple[int, ...], torch.Tensor | None],
        recv: tuple[int, ...] | list[int],
    ) -> dict[int, torch.Tensor | None] | None:
        """Declare this rank's collective step.

        Args:
            send: mapping from tuple-of-destination-ranks to tensor OR None.
                  A ``None`` value is a notification-only deposit.
            recv: iterable of source ranks this rank expects to receive from.

        Returns:
            ``dict[src_rank -> tensor | None]`` if the fast path hit (all
            recv sources have already deposited and this rank can keep running).
            ``None`` if the caller must follow up with :meth:`release_gpu`.

        Raises:
            :class:`CollectiveMismatch` if any peer's pending prepare is
            inconsistent with this call's send/recv spec.

        Collective mapping
        ------------------
        Every PyTorch collective maps to a single ``prepare`` call. Below,
        ``self`` is this rank's id, ``others`` is the tuple of peer ranks
        participating in the group. ``None`` as a ``send`` value is a
        notification-only deposit — the matching ``recv`` returns ``None``.

        =============================  =================================================  =================================  =============================
        Collective                     ``send``                                           ``recv``                           post-local work
        =============================  =================================================  =================================  =============================
        ``send(dst, t)``               ``{(dst,): t}``                                    ``()``                             —
        ``recv(src)``                  ``{}``                                             ``(src,)``                         —
        ``broadcast(t, src)`` at src   ``{tuple(others): t}``                             ``()``                             —
        ``broadcast`` at non-src       ``{}``                                             ``(src,)``                         —
        ``reduce(t, dst)`` non-dst     ``{(dst,): t}``                                    ``()``                             —
        ``reduce`` at dst              ``{}``                                             ``tuple(others)``                  local reduce
        ``scatter(list, src)`` src     ``{(i,): list[i] for i != self}``                  ``()``                             use ``list[self]``
        ``scatter`` non-src            ``{}``                                             ``(src,)``                         —
        ``gather(t, dst)`` non-dst     ``{(dst,): t}``                                    ``()``                             —
        ``gather`` at dst              ``{}``                                             ``tuple(others)``                  assemble list
        ``all_gather(t)``              ``{tuple(others): t}``                             ``tuple(others)``                  concat
        ``all_reduce(t)``              ``{tuple(others): t}``                             ``tuple(others)``                  local reduce
        ``reduce_scatter(chunks)``     ``{(i,): chunks[i] for i != self}``                ``tuple(others)``                  reduce with ``chunks[self]``
        ``all_to_all(chunks)``         ``{(j,): chunks[j] for j != self}``                ``tuple(others)``                  —
        ``barrier()``                  ``{tuple(others): None}``                          ``tuple(others)``                  —
        =============================  =================================================  =================================  =============================
        """
        import torch

        send_entries: list[SendEntry] = []
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
                    f"send values must be Tensor or None, got {type(tensor).__name__}"
                )

        req = cast(
            JsonDict, {"op": OP_PREPARE, "send": send_entries, "recv": list(recv)}
        )
        resp_hdr, resp_payload = self._run_coro(
            self._engine.rpc(req, b"".join(payloads))
        )
        if resp_hdr.get("block"):
            return None
        return self._decode_recv(_recv_entries(resp_hdr), resp_payload)

    def release_gpu(self) -> dict[int, torch.Tensor | None]:
        """Checkpoint CUDA, send release_gpu to the coordinator, wait for
        the recv data, then restore before decoding any returned tensors.
        Must follow a :meth:`prepare` that returned ``None``.
        """
        import torch

        from .cuda_checkpoint import checkpoint_self, restore_self

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        checkpoint_self()
        try:
            resp_hdr, resp_payload = self._wait_for_recv_raw()
        finally:
            restore_self()
        return self._decode_recv(_recv_entries(resp_hdr), resp_payload)

    def wait_for_recv(self) -> dict[int, torch.Tensor | None]:
        """Send release_gpu to the coordinator and block until the recv
        data is delivered.  Like :meth:`release_gpu` but without CUDA
        checkpoint/restore — the caller manages that if needed.
        """
        resp_hdr, resp_payload = self._wait_for_recv_raw()
        return self._decode_recv(_recv_entries(resp_hdr), resp_payload)

    def _wait_for_recv_raw(self) -> tuple[JsonDict, bytes]:
        """Send release_gpu and return the raw framed response."""
        resp_hdr, resp_payload = self._run_coro(
            self._engine.rpc({"op": OP_RELEASE_GPU})
        )
        return resp_hdr, resp_payload

    def done(self) -> None:
        if self._loop is None:
            return
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
        self, entries: list[RecvEntry], payload: bytes
    ) -> dict[int, torch.Tensor | None]:
        out: dict[int, torch.Tensor | None] = {}
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
