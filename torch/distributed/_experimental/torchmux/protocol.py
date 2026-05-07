"""Wire protocol for the checkpoint coordinator.

Single source of truth for the op names, error codes, framing, and header
shapes shared between ``coordinator.py`` and ``coord_client.py``.

High-level model
----------------

N torch processes cooperate on one or more GPUs, coordinated by a central
coordinator.  Each process registers under an integer ``rank``.

Data exchange uses a single primitive, :func:`prepare`, that expresses any
point-to-point or collective pattern.  A ``prepare`` call declares:

  * ``send``:  what tensors (or notifications) this rank is depositing for
    specific peer ranks.
  * ``recv``:  which peer ranks this rank expects tensors (or notifications)
    from.

If all ``recv`` sources have already deposited and the scheduler can keep this
rank running, the coordinator returns the data synchronously (the "fast path").
Otherwise the coordinator replies ``block``; the caller must call
:func:`release_gpu` next, which checkpoints the process and waits for the
coordinator to return the data once all recv sources have deposited and this
rank is scheduled again.

Per-pair FIFO ordering is guaranteed: rank A's nth send to B pairs with
rank B's nth recv from A, matching PyTorch's "user ensures collective call
order" contract.

Wire format per message
-----------------------
::

    [ 4 bytes big-endian u32 : header_len ]
    [ header_len bytes       : UTF-8 JSON header ]
    [ 8 bytes big-endian u64 : payload_len ]    # 0 if no payload
    [ payload_len bytes      : concatenated tensor bodies ]

Every response header includes ``{"ok": bool, "error": str | None, ...}``.
When ``ok`` is false, ``error`` is a machine-readable string — see the
``ERR_*`` constants below.

Ops
---
``OP_REGISTER``
    Request:  ``{"op": "register", "rank": int}``
    Response: ``{"ok": true}``
    Registers this connection under ``rank``.  Error if already registered.
    Rank 0 returns immediately once registered. Later ranks remain blocked in
    register until their rank-ordered turn and the active rank yields.

``OP_PREPARE``
    Request::

        {
            "op": "prepare",
            "send": [{"dsts": [int, ...], "tensor": TensorHeader | null}, ...],
            "recv": [int, ...],  # source ranks
        }

    followed by concatenated tensor bodies (only for non-null send entries,
    in ``send`` order).

    Response — fast path (all recvs already deposited)::

        {
            "ok": true,
            "block": false,
            "recv": [{"src": int, "tensor": TensorHeader | null}, ...],
        }

    followed by concatenated tensor bodies (only non-null, in ``recv`` order).

    Response — slow path or scheduler yield, caller must call release_gpu next::

        {"ok": true, "block": true}

    Errors: ``"mismatch"`` if a peer's pending prepare is inconsistent
    (sent to a peer that didn't declare the recv, or recv declared from a
    peer that isn't sending).

``OP_RELEASE_GPU``
    Request:  ``{"op": "release_gpu"}``
    Response: ``{"ok": true, "recv": [...]}`` + payloads, same shape as
    the fast-path branch of ``prepare``.

    Valid only after a ``prepare`` that returned ``block: true``.  Blocks
    server-side until this rank's recv set is fully satisfied and the rank is
    scheduled again.  Returns the recv data.  Client is expected to have
    checkpointed before sending this request and restore immediately on
    receiving the response.

``OP_DONE``
    Request:  ``{"op": "done"}``
    Response: ``{"ok": true}``
    Client is exiting; coordinator cleans up its state, drops any stale
    mailboxes, and fails any pending prepares on peers with ``peer_gone``.

(See ``CoordClient.prepare`` in ``coord_client.py`` for how to express each
PyTorch collective using this primitive.)

Error codes
-----------
``ERR_PEER_GONE``  — a peer we were waiting on (recv src) disappeared.
``ERR_MISMATCH``   — a peer's pending prepare is inconsistent with ours.
"""

import asyncio
import json
import struct
from typing import TypeAlias, TypedDict


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonDict: TypeAlias = dict[str, JsonValue]


# ---- Op names ----

OP_REGISTER = "register"
OP_PREPARE = "prepare"
OP_RELEASE_GPU = "release_gpu"
OP_DONE = "done"


# ---- Error codes ----

ERR_PEER_GONE = "peer_gone"
ERR_MISMATCH = "mismatch"


# ---- Header shapes ----


class TensorHeader(TypedDict):
    shape: list[int]
    dtype: str  # e.g. "float32", "bfloat16"
    device: str  # e.g. "cuda:0", "cpu"
    nbytes: int  # payload size for this tensor


class SendEntry(TypedDict):
    dsts: list[int]
    tensor: TensorHeader | None


class RecvEntry(TypedDict):
    src: int
    tensor: TensorHeader | None


# ---- Framing ----


async def read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly n bytes or raise IncompleteReadError."""
    return await reader.readexactly(n)


async def read_message(reader: asyncio.StreamReader) -> tuple[JsonDict, bytes]:
    """Read one framed message. Returns ``(header_dict, payload_bytes)``."""
    (hdr_len,) = struct.unpack(">I", await read_exact(reader, 4))
    header: JsonDict = json.loads((await read_exact(reader, hdr_len)).decode("utf-8"))
    (payload_len,) = struct.unpack(">Q", await read_exact(reader, 8))
    payload: bytes = await read_exact(reader, payload_len) if payload_len else b""
    return header, payload


async def write_message(
    writer: asyncio.StreamWriter, header: JsonDict, payload: bytes = b""
) -> None:
    """Write one framed message and drain."""
    hdr_bytes = json.dumps(header).encode("utf-8")
    writer.write(struct.pack(">I", len(hdr_bytes)))
    writer.write(hdr_bytes)
    writer.write(struct.pack(">Q", len(payload)))
    if payload:
        writer.write(payload)
    await writer.drain()
