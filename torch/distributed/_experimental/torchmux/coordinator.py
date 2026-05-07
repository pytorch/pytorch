"""Coordinator server for torch-process orchestration.

Clients register by integer rank.  Data exchange goes through
:func:`prepare` (see ``protocol.py``).  Wire framing and op contracts
are defined there.

Run as::

    python coordinator.py                          # UDS at default path
    python coordinator.py --socket /tmp/foo.sock   # explicit UDS path
    python coordinator.py --tcp-port 0             # TCP; 0 picks a free port

On startup, prints the bound address on ONE line of stdout so launchers
can capture it::

    ADDR uds:/tmp/foo.sock
    ADDR tcp:127.0.0.1:54321
"""

import argparse
import asyncio
import collections
import os
import signal
import sys
from typing import cast, TypeAlias, TypedDict

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
    TensorHeader,
    write_message,
)


_ReceivedItem: TypeAlias = tuple[TensorHeader | None, bytes]
_ReceivedMap: TypeAlias = dict[int, _ReceivedItem]


# ---- Internal signals ----


class _PeerGone(Exception):
    """Internal signal: a peer we were waiting on disappeared."""


# ---- Pending state ----


class _SendEntry(TypedDict):
    dsts: list[int]
    tensor: TensorHeader | None
    payload: bytes


class _Prepare:
    """A rank's outstanding prepare() call.  Holds the parsed send/recv specs
    plus the future the coordinator completes when the recv is satisfied."""

    def __init__(self, send: list[_SendEntry], recv: list[int]) -> None:
        self.send = send
        self.recv: list[int] = list(recv)
        # Populated as recv sources deposit:
        # dict[src_rank] -> (tensor_header|None, payload_bytes)
        self.received: _ReceivedMap = {}
        # Set when the client calls release_gpu.
        # Resolved once all recvs are satisfied.
        self.release_future: asyncio.Future[_ReceivedMap] | None = None
        # Set by _cleanup_client when a peer disappears before release_future
        # exists.  handle_release_gpu checks this to report PeerGone.
        self.failed: _PeerGone | None = None


# ---- Coordinator ----


class Coordinator:
    def __init__(self) -> None:
        # Registered clients: rank -> writer.
        self.clients: dict[int, asyncio.StreamWriter] = {}
        self.writer_to_rank: dict[asyncio.StreamWriter, int] = {}

        # Startup scheduling: rank 0 starts immediately once it registers.
        # Other ranks block in register until their rank-ordered turn and the
        # active rank yields.
        self.next_register_rank = 0
        self.register_futures: list[tuple[int, asyncio.Future[None]]] = []
        self.active_rank: int | None = None

        # Active prepare per rank (one at a time).
        self.pending: dict[int, _Prepare] = {}

        # Mailboxes: (src, dst) -> FIFO of deposits made by src for dst.
        # Each deposit is (tensor_header_or_None, payload_bytes).
        self.mailboxes: dict[tuple[int, int], collections.deque[_ReceivedItem]] = {}

    def log(self, *args: object) -> None:
        print("[coord]", *args, file=sys.stderr, flush=True)

    # ---- Scheduling ----

    def _try_complete_pending(self) -> None:
        """Resolve pending prepares whose recv set is now fully satisfied."""
        self._try_schedule_next()

    def _recv_available(self, rank: int, recv_spec: list[int]) -> bool:
        return all(self.mailboxes.get((src, rank)) for src in recv_spec)

    def _collect_available(self, rank: int, prep: _Prepare) -> None:
        for src in prep.recv:
            if src in prep.received:
                continue
            mbox = self.mailboxes.get((src, rank))
            if mbox:
                prep.received[src] = mbox.popleft()
                if not mbox:
                    self.mailboxes.pop((src, rank), None)

    def _take_available(self, rank: int, recv_spec: list[int]) -> _ReceivedMap:
        received: _ReceivedMap = {}
        for src in recv_spec:
            mbox = self.mailboxes[(src, rank)]
            received[src] = mbox.popleft()
            if not mbox:
                self.mailboxes.pop((src, rank), None)
        return received

    def _pop_register_future(self, rank: int) -> asyncio.Future[None] | None:
        candidates = [
            (i, r, fut)
            for i, (r, fut) in enumerate(self.register_futures)
            if not fut.done()
        ]
        for i, r, fut in candidates:
            if r == rank:
                self.register_futures.pop(i)
                return fut
        return None

    def _try_schedule_next(self) -> None:
        if self.active_rank is not None:
            return

        fut = self._pop_register_future(self.next_register_rank)
        if fut is not None:
            self.active_rank = self.next_register_rank
            self.next_register_rank += 1
            self.log(f"unblocking rank {self.active_rank} from register")
            fut.set_result(None)
            return

        for rank, prep in list(self.pending.items()):
            if prep.release_future is None or prep.release_future.done():
                continue
            self._collect_available(rank, prep)
            if len(prep.received) == len(prep.recv):
                self.active_rank = rank
                self.log(f"scheduling rank {rank}")
                prep.release_future.set_result(prep.received)
                self.pending.pop(rank)
                return

    # ---- Op: register ----

    async def handle_register(
        self, hdr: JsonDict, writer: asyncio.StreamWriter
    ) -> None:
        rank = hdr["rank"]
        if not isinstance(rank, int):
            await write_message(writer, {"ok": False, "error": "rank must be int"})
            return
        if rank in self.clients:
            await write_message(
                writer, {"ok": False, "error": f"rank {rank} already registered"}
            )
            return
        self.clients[rank] = writer
        self.writer_to_rank[writer] = rank

        fut = asyncio.get_running_loop().create_future()
        self.register_futures.append((rank, fut))
        self.log(f"registered rank {rank}")
        self._try_schedule_next()
        await fut
        await write_message(writer, {"ok": True, "error": None})

    # ---- Op: prepare ----

    async def handle_prepare(
        self, hdr: JsonDict, payload: bytes, writer: asyncio.StreamWriter
    ) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        if rank in self.pending:
            await write_message(
                writer, {"ok": False, "error": "already has pending prepare"}
            )
            return

        send_spec = cast(list[JsonDict], hdr.get("send") or [])
        recv_spec = cast(list[int], hdr.get("recv") or [])

        # Split the concatenated payload into per-send-entry slices.
        offset = 0
        send_entries: list[_SendEntry] = []
        for entry in send_spec:
            tensor_hdr = cast(TensorHeader | None, entry.get("tensor"))
            dsts = cast(list[int], entry["dsts"])
            if tensor_hdr is None:
                send_entries.append({"dsts": dsts, "tensor": None, "payload": b""})
            else:
                nbytes = tensor_hdr["nbytes"]
                body = payload[offset : offset + nbytes]
                offset += nbytes
                send_entries.append(
                    {"dsts": dsts, "tensor": tensor_hdr, "payload": body}
                )

        # Mismatch detection.
        mismatch = self._check_mismatch(rank, send_entries, recv_spec)
        if mismatch is not None:
            await write_message(
                writer, {"ok": False, "error": ERR_MISMATCH, "detail": mismatch}
            )
            return

        # Deposit sends.
        for entry in send_entries:
            for dst in entry["dsts"]:
                self.mailboxes.setdefault((rank, dst), collections.deque()).append(
                    (entry["tensor"], entry["payload"])
                )

        can_recv_now = self._recv_available(rank, recv_spec)
        if can_recv_now:
            # Fast path — respond with data immediately.
            # We could yield the active rank here. For now, keep it running to
            # avoid expensive context swaps, but yielding when the mailboxes get
            # very large would probably be a good improvement.
            received = self._take_available(rank, recv_spec)
            await write_message(
                writer,
                *self._build_recv_response(recv_spec, received),
            )
        else:
            # Slow path — client must call release_gpu next.
            prep = _Prepare(send_entries, recv_spec)
            self._collect_available(rank, prep)
            self.pending[rank] = prep
            await write_message(writer, {"ok": True, "block": True, "error": None})

        # This rank's deposits may have satisfied other pending prepares.
        self._try_complete_pending()

    def _check_mismatch(
        self, rank: int, send_entries: list[_SendEntry], recv_spec: list[int]
    ) -> str | None:
        targets = set()
        for entry in send_entries:
            targets.update(entry["dsts"])
            for dst in entry["dsts"]:
                pd = self.pending.get(dst)
                if pd is not None and rank not in pd.recv:
                    return f"rank {rank} sent to {dst}; {dst} does not recv from {rank}"
        for waiting_rank, prep in self.pending.items():
            if rank in prep.recv and rank not in prep.received:
                if waiting_rank in targets:
                    continue
                if self.mailboxes.get((rank, waiting_rank)):
                    continue
                return (
                    f"rank {waiting_rank} recv from {rank}; "
                    f"{rank} does not send to {waiting_rank}"
                )
        for src in recv_spec:
            ps = self.pending.get(src)
            if ps is None:
                continue
            peer_targets = set()
            for entry in ps.send:
                peer_targets.update(entry["dsts"])
            if rank in peer_targets:
                continue
            if self.mailboxes.get((src, rank)):
                continue
            return f"rank {rank} recv from {src}; {src} does not send to {rank}"
        return None

    def _build_recv_response(
        self,
        recv_spec: list[int],
        received: _ReceivedMap,
    ) -> tuple[JsonDict, bytes]:
        entries: list[RecvEntry] = []
        bodies: list[bytes] = []
        for src in recv_spec:
            hdr, body = received[src]
            entries.append({"src": src, "tensor": hdr})
            if body:
                bodies.append(body)
        return (
            cast(
                JsonDict,
                {"ok": True, "block": False, "error": None, "recv": entries},
            ),
            b"".join(bodies),
        )

    # ---- Op: release_gpu ----

    async def handle_release_gpu(
        self, hdr: JsonDict, writer: asyncio.StreamWriter
    ) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        prep = self.pending.get(rank)
        if prep is None:
            await write_message(
                writer, {"ok": False, "error": "no pending prepare to release"}
            )
            return

        if prep.failed is not None:
            self.pending.pop(rank, None)
            if self.active_rank == rank:
                self.active_rank = None
            self._try_schedule_next()
            await write_message(writer, {"ok": False, "error": ERR_PEER_GONE})
            return

        fut = asyncio.get_running_loop().create_future()
        prep.release_future = fut
        if self.active_rank == rank:
            self.active_rank = None

        # Check if already satisfied.
        self._collect_available(rank, prep)
        if len(prep.received) == len(prep.recv):
            self.log(f"rank {rank} recv already satisfied")
        self._try_schedule_next()

        try:
            received = await fut
        except _PeerGone:
            await write_message(writer, {"ok": False, "error": ERR_PEER_GONE})
            return
        except asyncio.CancelledError:
            raise

        await write_message(writer, *self._build_recv_response(prep.recv, received))

    # ---- Op: done ----

    async def handle_done(self, hdr: JsonDict, writer: asyncio.StreamWriter) -> None:
        await self._cleanup_client(writer)
        await write_message(writer, {"ok": True, "error": None})

    # ---- Connection cleanup ----

    async def _cleanup_client(self, writer: asyncio.StreamWriter) -> None:
        rank = self.writer_to_rank.pop(writer, None)
        if rank is None:
            return
        self.clients.pop(rank, None)

        # Cancel any deferred register future for this rank.
        new_reg = []
        skipped_register_rank = False
        for r, fut in self.register_futures:
            if r == rank:
                skipped_register_rank = r == self.next_register_rank
                if not fut.done():
                    fut.cancel()
            else:
                new_reg.append((r, fut))
        self.register_futures = new_reg
        if skipped_register_rank:
            self.next_register_rank += 1

        prep = self.pending.pop(rank, None)
        if prep is not None and prep.release_future is not None:
            if not prep.release_future.done():
                prep.release_future.cancel()
        if self.active_rank == rank:
            self.active_rank = None

        # Drop mailboxes whose dst is this rank.
        for k in [k for k in self.mailboxes if k[1] == rank]:
            self.mailboxes.pop(k, None)

        # Fail other ranks' pending prepares that expected from this rank.
        for other_rank, other_prep in list(self.pending.items()):
            self._collect_available(other_rank, other_prep)
            if rank in other_prep.recv and rank not in other_prep.received:
                if (
                    other_prep.release_future is not None
                    and not other_prep.release_future.done()
                ):
                    other_prep.release_future.set_exception(_PeerGone())
                    self.pending.pop(other_rank, None)
                else:
                    other_prep.failed = _PeerGone()

        # A departing rank may have left deposits that satisfy others.
        self._try_complete_pending()

        # If no peers remain, fail any remaining register waiters.
        if len(self.clients) == 0:
            for _, fut in self.register_futures:
                if not fut.done():
                    fut.cancel()
            self.register_futures.clear()
            self.next_register_rank = 0

    async def handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername") or writer.get_extra_info("sockname")
        self.log(f"connection from {peer}")
        try:
            while True:
                try:
                    header, payload = await read_message(reader)
                except asyncio.IncompleteReadError:
                    break
                op = header.get("op")
                try:
                    if op == OP_REGISTER:
                        await self.handle_register(header, writer)
                    elif op == OP_PREPARE:
                        await self.handle_prepare(header, payload, writer)
                    elif op == OP_RELEASE_GPU:
                        await self.handle_release_gpu(header, writer)
                    elif op == OP_DONE:
                        await self.handle_done(header, writer)
                        break
                    else:
                        await write_message(
                            writer, {"ok": False, "error": f"unknown op {op!r}"}
                        )
                except Exception as e:
                    self.log(f"op {op!r} raised: {e!r}")
                    try:
                        await write_message(writer, {"ok": False, "error": repr(e)})
                    except Exception:
                        break
        finally:
            await self._cleanup_client(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ---- Entry point ----


async def serve(
    coord: Coordinator, socket_path: str | None, tcp_port: int | None
) -> None:
    if tcp_port is not None:
        server = await asyncio.start_server(
            coord.handle_connection, host="127.0.0.1", port=tcp_port
        )
        bound = server.sockets[0].getsockname()
        print(f"ADDR tcp:{bound[0]}:{bound[1]}", flush=True)
    else:
        if not socket_path:
            raise ValueError("socket_path is required when tcp_port is None")
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server = await asyncio.start_unix_server(
            coord.handle_connection, path=socket_path
        )
        print(f"ADDR uds:{socket_path}", flush=True)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async with server:
        stop_task = asyncio.create_task(stop_event.wait())
        serve_task = asyncio.create_task(server.serve_forever())
        _done, pending = await asyncio.wait(
            {stop_task, serve_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        server.close()
        await server.wait_closed()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--socket",
        help="UDS path (default: $XDG_RUNTIME_DIR/coord.sock or /tmp/coord.sock)",
    )
    g.add_argument(
        "--tcp-port", type=int, help="Bind TCP on 127.0.0.1; 0 picks a free port"
    )
    args = ap.parse_args()

    coord = Coordinator()
    if args.tcp_port is None:
        socket_path = args.socket or os.path.join(
            os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "coord.sock"
        )
        asyncio.run(serve(coord, socket_path=socket_path, tcp_port=None))
    else:
        asyncio.run(serve(coord, socket_path=None, tcp_port=args.tcp_port))


if __name__ == "__main__":
    main()
