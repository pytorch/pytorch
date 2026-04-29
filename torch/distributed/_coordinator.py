"""
Coordinator server for torchmux GPU-sharing orchestration.

Clients register by integer rank and GPU id. Each physical GPU has its own
baton — exactly one rank per GPU holds at a time. Ranks on different GPUs
run concurrently. Data exchange goes through :func:`prepare` (see
``_coord_protocol.py``).

Run standalone::

    python -m torch.distributed._coordinator --tcp-port 0

Or started as a subprocess by ``torchmux.main()``.

On startup, prints the bound address on ONE line to stdout::

    ADDR tcp:127.0.0.1:54321
"""

import argparse
import asyncio
import collections
import json
import os
import signal
import sys

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


class _NoPeers(Exception):
    pass


class _PeerGone(Exception):
    pass


class _Prepare:
    def __init__(self, send: list, recv: list[int]):
        self.send = send
        self.recv: list[int] = list(recv)
        self.received: dict[int, tuple[dict | None, bytes]] = {}
        self.release_future: asyncio.Future | None = None


class Coordinator:
    def __init__(self, initial_holders: dict[int, int] | None = None):
        self.clients: dict[int, asyncio.StreamWriter] = {}
        self.writer_to_rank: dict[asyncio.StreamWriter, int] = {}
        self.rank_gpu: dict[int, int] = {}

        # Per-GPU baton: gpu_id -> rank currently holding, or None.
        self.current_holders: dict[int, int | None] = dict(initial_holders or {})

        self.waiters: collections.deque[tuple[int, asyncio.Future]] = (
            collections.deque()
        )
        self.pending: dict[int, _Prepare] = {}
        self.mailboxes: dict[
            tuple[int, int], collections.deque[tuple[dict | None, bytes]]
        ] = {}
        self.eligible: collections.deque[int] = collections.deque()

    def log(self, *args):
        print("[coord]", *args, file=sys.stderr, flush=True)

    # ---- Baton helpers ----

    async def _grant_baton(self, rank: int) -> None:
        gpu_id = self.rank_gpu[rank]
        assert self.current_holders.get(gpu_id) is None
        self.current_holders[gpu_id] = rank

    async def _try_schedule_next(self, gpu_id: int) -> None:
        if self.current_holders.get(gpu_id) is not None:
            return

        # Priority 1: eligible ranks (data-ready) on this GPU.
        for rank in list(self.eligible):
            if self.rank_gpu.get(rank) != gpu_id:
                continue
            prep = self.pending.get(rank)
            if prep is None or prep.release_future is None:
                self.eligible = collections.deque(
                    r for r in self.eligible if r != rank
                )
                continue
            self.eligible = collections.deque(
                r for r in self.eligible if r != rank
            )
            await self._grant_baton(rank)
            if not prep.release_future.done():
                prep.release_future.set_result(prep.received)
            self.pending.pop(rank, None)
            return

        # Priority 2: bootstrap / yield waiters on this GPU.
        for rank, fut in list(self.waiters):
            if self.rank_gpu.get(rank) != gpu_id:
                continue
            if rank not in self.clients:
                continue
            self.waiters = collections.deque(
                (r, f) for r, f in self.waiters if r != rank
            )
            await self._grant_baton(rank)
            if not fut.done():
                fut.set_result(None)
            return

    # ---- Op: register ----

    async def handle_register(self, hdr, writer) -> None:
        rank = hdr["rank"]
        gpu_id = hdr.get("gpu_id", 0)
        if rank in self.clients:
            await write_message(
                writer, {"ok": False, "error": f"rank {rank} already registered"}
            )
            return
        self.clients[rank] = writer
        self.writer_to_rank[writer] = rank
        self.rank_gpu[rank] = gpu_id
        self.log(f"registered rank {rank} on gpu {gpu_id}")
        await write_message(writer, {"ok": True, "error": None})

    # ---- Op: wait_for_turn (bootstrap) ----

    async def handle_wait_for_turn(self, hdr, writer) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        gpu_id = self.rank_gpu[rank]
        if self.current_holders.get(gpu_id) is None:
            await self._grant_baton(rank)
            await write_message(writer, {"ok": True, "error": None})
            return
        if self.current_holders.get(gpu_id) == rank:
            await write_message(writer, {"ok": True, "error": None})
            return
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append((rank, fut))
        try:
            await fut
        except _NoPeers:
            await write_message(writer, {"ok": False, "error": ERR_NO_PEERS})
            return
        except asyncio.CancelledError:
            self.waiters = collections.deque(
                (r, f) for r, f in self.waiters if f is not fut
            )
            raise
        await write_message(writer, {"ok": True, "error": None})

    # ---- Op: prepare ----

    async def handle_prepare(self, hdr, payload: bytes, writer) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        if rank in self.pending:
            await write_message(
                writer, {"ok": False, "error": "already has pending prepare"}
            )
            return

        send_spec = hdr.get("send") or []
        recv_spec = hdr.get("recv") or []

        offset = 0
        send_entries = []
        for entry in send_spec:
            tensor_hdr = entry.get("tensor")
            dsts = entry["dsts"]
            if tensor_hdr is None:
                send_entries.append({"dsts": dsts, "tensor": None, "payload": b""})
            else:
                nbytes = tensor_hdr["nbytes"]
                body = payload[offset : offset + nbytes]
                offset += nbytes
                send_entries.append(
                    {"dsts": dsts, "tensor": tensor_hdr, "payload": body}
                )

        for entry in send_entries:
            for dst in entry["dsts"]:
                self.mailboxes.setdefault((rank, dst), collections.deque()).append(
                    (entry["tensor"], entry["payload"])
                )

        all_ready = all(
            self.mailboxes.get((src, rank)) for src in recv_spec
        )

        if all_ready and len(recv_spec) > 0:
            received: dict[int, tuple[dict | None, bytes]] = {}
            for src in recv_spec:
                mbox = self.mailboxes[(src, rank)]
                received[src] = mbox.popleft()
                if not mbox:
                    self.mailboxes.pop((src, rank), None)
            await write_message(
                writer,
                *self._build_recv_response(recv_spec, received),
            )
        elif len(recv_spec) == 0:
            await write_message(
                writer,
                *self._build_recv_response(recv_spec, {}),
            )
        else:
            prep = _Prepare(send_entries, recv_spec)
            for src in recv_spec:
                mbox = self.mailboxes.get((src, rank))
                if mbox:
                    prep.received[src] = mbox.popleft()
                    if not mbox:
                        self.mailboxes.pop((src, rank), None)
            self.pending[rank] = prep
            await write_message(writer, {"ok": True, "block": True, "error": None})

        await self._sweep_eligible_after_send(rank)

    def _build_recv_response(self, recv_spec, received):
        entries = []
        bodies: list[bytes] = []
        for src in recv_spec:
            hdr, body = received[src]
            entries.append({"src": src, "tensor": hdr})
            if body:
                bodies.append(body)
        return (
            {"ok": True, "block": False, "error": None, "recv": entries},
            b"".join(bodies),
        )

    async def _sweep_eligible_after_send(self, just_sent_rank: int) -> None:
        newly_eligible_gpus: set[int] = set()
        for rank, prep in list(self.pending.items()):
            if prep.release_future is None:
                continue
            for src in prep.recv:
                if src in prep.received:
                    continue
                mbox = self.mailboxes.get((src, rank))
                if mbox:
                    prep.received[src] = mbox.popleft()
                    if not mbox:
                        self.mailboxes.pop((src, rank), None)
            if len(prep.received) == len(prep.recv) and rank not in self.eligible:
                self.eligible.append(rank)
                gpu_id = self.rank_gpu.get(rank)
                if gpu_id is not None:
                    newly_eligible_gpus.add(gpu_id)

        for gpu_id in newly_eligible_gpus:
            await self._try_schedule_next(gpu_id)

    # ---- Op: release_gpu ----

    async def handle_release_gpu(self, hdr, writer) -> None:
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

        gpu_id = self.rank_gpu[rank]
        if self.current_holders.get(gpu_id) == rank:
            self.current_holders[gpu_id] = None

        fut = asyncio.get_running_loop().create_future()
        prep.release_future = fut

        for src in prep.recv:
            if src in prep.received:
                continue
            mbox = self.mailboxes.get((src, rank))
            if mbox:
                prep.received[src] = mbox.popleft()
                if not mbox:
                    self.mailboxes.pop((src, rank), None)
        if len(prep.received) == len(prep.recv) and rank not in self.eligible:
            self.eligible.append(rank)

        await self._try_schedule_next(gpu_id)

        try:
            received = await fut
        except _PeerGone:
            await write_message(writer, {"ok": False, "error": ERR_PEER_GONE})
            return
        except _NoPeers:
            await write_message(writer, {"ok": False, "error": ERR_NO_PEERS})
            return

        await write_message(
            writer, *self._build_recv_response(prep.recv, received)
        )

    # ---- Op: release_baton / acquire_baton ----

    async def handle_release_baton(self, hdr, writer) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        gpu_id = self.rank_gpu[rank]
        if self.current_holders.get(gpu_id) == rank:
            self.current_holders[gpu_id] = None
        await write_message(writer, {"ok": True, "error": None})
        await self._try_schedule_next(gpu_id)

    async def handle_acquire_baton(self, hdr, writer) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        gpu_id = self.rank_gpu[rank]
        if self.current_holders.get(gpu_id) == rank:
            await write_message(writer, {"ok": True, "error": None})
            return
        if self.current_holders.get(gpu_id) is None:
            await self._grant_baton(rank)
            await write_message(writer, {"ok": True, "error": None})
            return
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append((rank, fut))
        try:
            await fut
        except _NoPeers:
            await write_message(writer, {"ok": False, "error": ERR_NO_PEERS})
            return
        await write_message(writer, {"ok": True, "error": None})

    # ---- Op: done ----

    async def handle_done(self, hdr, writer) -> None:
        await self._cleanup_client(writer)
        await write_message(writer, {"ok": True, "error": None})

    # ---- Connection cleanup ----

    async def _cleanup_client(self, writer: asyncio.StreamWriter) -> None:
        rank = self.writer_to_rank.pop(writer, None)
        if rank is None:
            return
        self.clients.pop(rank, None)
        gpu_id = self.rank_gpu.pop(rank, None)

        self.waiters = collections.deque(
            (r, f) for r, f in self.waiters if r != rank
        )
        self.eligible = collections.deque(r for r in self.eligible if r != rank)

        prep = self.pending.pop(rank, None)
        if prep is not None and prep.release_future is not None:
            if not prep.release_future.done():
                prep.release_future.cancel()

        for k in [k for k in self.mailboxes if k[1] == rank]:
            self.mailboxes.pop(k, None)

        for other_rank, other_prep in list(self.pending.items()):
            if rank in other_prep.recv and rank not in other_prep.received:
                if (
                    other_prep.release_future is not None
                    and not other_prep.release_future.done()
                ):
                    other_prep.release_future.set_exception(_PeerGone())
                self.pending.pop(other_rank, None)
                self.eligible = collections.deque(
                    r for r in self.eligible if r != other_rank
                )

        if gpu_id is not None and self.current_holders.get(gpu_id) == rank:
            self.current_holders[gpu_id] = None
            await self._try_schedule_next(gpu_id)

        if len(self.clients) == 0:
            for _r, fut in list(self.waiters):
                if not fut.done():
                    fut.set_exception(_NoPeers())
            self.waiters.clear()

    async def handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
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
                    elif op == OP_WAIT_FOR_TURN:
                        await self.handle_wait_for_turn(header, writer)
                    elif op == OP_PREPARE:
                        await self.handle_prepare(header, payload, writer)
                    elif op == OP_RELEASE_GPU:
                        await self.handle_release_gpu(header, writer)
                    elif op == OP_RELEASE_BATON:
                        await self.handle_release_baton(header, writer)
                    elif op == OP_ACQUIRE_BATON:
                        await self.handle_acquire_baton(header, writer)
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
                        await write_message(
                            writer, {"ok": False, "error": repr(e)}
                        )
                    except Exception:
                        break
        finally:
            await self._cleanup_client(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ---- Server entry point ----


async def serve(
    coord: Coordinator,
    socket_path: str | None,
    tcp_port: int | None,
):
    buf_limit = 16 * 1024 * 1024
    if tcp_port is not None:
        server = await asyncio.start_server(
            coord.handle_connection,
            host="127.0.0.1",
            port=tcp_port,
            limit=buf_limit,
        )
        bound = server.sockets[0].getsockname()
        print(f"ADDR tcp:{bound[0]}:{bound[1]}", flush=True)
    else:
        assert socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server = await asyncio.start_unix_server(
            coord.handle_connection, path=socket_path, limit=buf_limit
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


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--socket", help="UDS path")
    g.add_argument(
        "--tcp-port", type=int, help="TCP port on 127.0.0.1; 0 picks a free port"
    )
    ap.add_argument(
        "--initial-holders",
        help='JSON dict mapping gpu_id to initial holder rank, e.g. \'{"0":0,"1":1}\'',
    )
    args = ap.parse_args()

    initial_holders: dict[int, int] = {}
    if args.initial_holders:
        initial_holders = {
            int(k): int(v) for k, v in json.loads(args.initial_holders).items()
        }

    coord = Coordinator(initial_holders=initial_holders)
    if args.tcp_port is None:
        socket_path = args.socket or os.path.join(
            os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "torchmux_coord.sock"
        )
        asyncio.run(serve(coord, socket_path=socket_path, tcp_port=None))
    else:
        asyncio.run(serve(coord, socket_path=None, tcp_port=args.tcp_port))


if __name__ == "__main__":
    main()
