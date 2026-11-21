# mypy: allow-untyped-defs
"""FlyDSL kernels for symm_mem::fused_all_gather_matmul and fused_matmul_reduce_scatter on ROCm gfx950 (AsyncTP).

Opt-in: the ops' wrappers (__init__.py) call all_gather_matmul / matmul_reduce_scatter only with
TORCH_SYMM_MEM_ENABLE_FLYDSL_ASYNC_TP set, and only for calls ag_supported() / rs_supported() admit (bf16, the
gather / scatter dim first in memory, "sum", the kernels' tile multiples, TP 2/4/8); anything else takes the
wrapper's next branch. This module imports only torch: FlyDSL, triton and the kernels (_fused_flydsl_kernels.py,
built on torch's vendored FlyDSL GEMM) are imported on the first admitted call.

Every rank of a group must take the same branch: the kernels rendezvous on a state's first call, and a rank that took
another branch would leave the others waiting (check_group_agrees() turns that into an error). The predicates read
only shapes, strides, dtypes, devices and the group size, never pointers, so they agree as long as the environment
(the env vars, the FlyDSL and triton installs) is the same on every rank.

All-gather + matmul (_AGState): one state per (K, N, TP, B layout, rows per rank, group, device), running one kernel
(agmm_gfx950): the first KC blocks
(rounded up to a multiple of 8) only pull the peers' shards into the gathered A chunk by chunk and then exit; every
other block computes one tile, peer tiles waiting on their chunk flag. One call, all on the current stream: bump the
device epoch, stage the shard into half (epoch & 1) of the symmetric buffer and into this rank's rows of the gathered
A, flush (one L2 write-back per XCD, which the peers' reads of the staged half need), the symmetric-memory barrier
(every rank's staging is in, and every peer is done reading this half from two calls ago), then the fused kernel.

A state is built at its first call: a collective symmetric-memory rendezvous and a FlyDSL compile (which also runs the
kernel once, with no copy blocks and the flags far past any target), so every rank must make its calls in the same
order, and the first call of a state must not be under CUDA graph capture (later calls may be: all state lives on the
device). The gathered A is written on every call, return_A or not. A chunk flag still missing after wait_ms() fails
the launch with a [FATAL] line instead of hanging; the barrier times out the same way from a state's second call on.
States are built without a lock: make a state's first call from one thread.

Matmul + reduce-scatter (_RSState): one state per (K, N, TP, B layout, rows, group, device), running one kernel
(rsmm_gfx950): the first R blocks (rounded up to a multiple of 8) only reduce; every other block computes one tile
of this rank's partial into its symmetric staging buffer in reduce-scatter order and signals the chunk's owner. The
order is 2D: G row groups x C column slabs make G * C steps, a step covering one row group and one column slab of EVERY
chunk (every owner's link is busy from the first step, and a step's rows of A stay in the Infinity Cache for the slabs
that share them); the owner's reducers sum a step's block of every source's staging once that step's tiles from every
source are in. One call, all on the current stream: bump the device epoch, the symmetric-memory barrier (every peer is
done reading this rank's staging from the previous call), the kernel. The staging buffer is M x N bf16 per state. Its
first call compiles and runs the kernel once with no reducers and every counter add into local scratch; the rest is as
for the all-gather, a step counter still short after wait_ms() failing the launch.

Tiles: both kernels run the 256x256 tile, except a narrow all-gather at small M (see TILE_ENV and _ag_tile), which runs
128x256: a different state (the tile is in its key), the same kernel and tile order in units of tiles.
"""

from __future__ import annotations

import collections
import functools
import os

import torch
import torch.distributed.distributed_c10d as c10d


WORLDS = (2, 4, 8)
# Above this N per rank the stock path is faster (its copy hides behind its own matmul).
AG_MAX_N = 20480
TIMEOUT_ENV = "TORCH_SYMM_MEM_FLYDSL_TIMEOUT_MS"
_AGREE = "TORCH_SYMM_MEM_ENABLE_FLYDSL_ASYNC_TP, TORCH_DISABLE_NATIVE_JIT, the FlyDSL and triton installs"
CHUNK_ROWS = 256  # rows per flagged chunk of the gathered A: one tile row
SLOT = 32  # int32 stride between flags, so each flag has its own 128-byte line
STAGE_BLOCK = 4096
# Workgroups go round-robin over the XCDs (8 per XCD on gfx950), so every XCD's L2 gets a flush; a CU mask that left
# an XCD out would leave its staged lines unflushed.
FLUSH_PROGRAMS = 64
# (K, N / TP, TP) -> copy blocks where the best measured at M 65536 (Llama 3.1 8B / 70B) is not KC_DEFAULT.
KC = {
    (8192, 2560, 4): 15,
    (8192, 2048, 4): 15,
    (8192, 14336, 2): 6,
    (4096, 14336, 2): 6,
}
KC_DEFAULT = {8: 28, 4: 12, 2: 8}
# A reduce-scatter chunk (rows / TP) must be a multiple of this many rows (the shapes the kernel is validated on).
RS_UNIT_ROWS = {8: 256, 4: 512, 2: 1024}
# The reduce-scatter's step grid: RS_G row groups x RS_C column slabs (RS_C_WIDE for an N of RS_WIDE_N or more), fitted
# down to divisors of the call's tile rows / columns that make every step a multiple of 8 tiles. Measured at TP 8, M 65536.
RS_G = 8
RS_C = 4
RS_C_WIDE = 16
RS_WIDE_N = 16384
# (K, N, TP) -> reducer blocks where the best measured at M 65536 (Llama 3.1) is not R_DEFAULT: the calls whose GEMM is
# at least as long as their reduction (down / gate-up, K large for their N) want few reducers, as every reducer is a CU
# the GEMM does not have; the others (o / qkv, K small) are bound by the reduction and want R_DEFAULT.
R = {
    (3584, 8192, 8): 16,
    (7168, 8192, 8): 16,
    (6656, 16384, 8): 16,
    (13312, 16384, 8): 16,
    (3584, 4096, 8): 16,
    (2048, 8192, 4): 16,
    (7168, 8192, 4): 8,
    (2560, 8192, 4): 16,
    (14336, 8192, 4): 16,
    (4096, 8192, 2): 16,
    (5120, 8192, 2): 16,
}
R_DEFAULT = {8: 28, 4: 12, 2: 8}

# Tile (block_m x block_n) of the all-gather kernel's GEMM (512 threads at both). At small M a narrow-N all-gather is transfer
# bound, and the 256 rows of the last arriving chunk are a whole 256x256 tile's time of exposed tail: the half-height tile
# halves it (measured at TP 8, M 8192 / 16384, N <= 1792: 3-10% faster; a wider N is GEMM bound, where it is 2-47% slower).
# The reduce-scatter gains nothing from it, so it always runs the default tile.
# TORCH_SYMM_MEM_FLYDSL_TILE: "auto" (default), or one of TILES to force it for the all-gather (a tile that doesn't fit the
# call falls back to the default). Read on every call; the tile is part of the state's key.
TILE_ENV = "TORCH_SYMM_MEM_FLYDSL_TILE"
TILES = ("256x256", "128x256")  # the first is the default
# "auto" takes AUTO_TILE for an all-gather of at most AUTO_MAX_LM rows per rank and a global N (N per rank x TP) of at most
# AUTO_MAX_N.
AUTO_TILE = "128x256"
AUTO_MAX_LM = 2048
AUTO_MAX_N = 14336

# Calls per op ("ag", "rs") that took the FlyDSL path; read by tests.
_calls: collections.Counter = collections.Counter()
_states: dict = {}
_states_built: collections.Counter = collections.Counter()


@functools.cache
def _available(device_index: int) -> bool:
    from torch._native import flydsl_utils as fu
    from torch.utils._triton import has_triton

    return (
        fu.runtime_available()
        and fu._version_is_ok()
        and has_triton()
        and fu._is_supported_arch(device_index, ("gfx950",))
    )


def _flat(t, dim):
    """t with `dim` moved first, as a 2D [rows, last dim] view; None for the last dim, an empty t, or a layout that
    would need a copy (the micro-pipeline pass restrides its operands so that it doesn't)."""
    if t.dim() < 2 or not 0 <= dim < t.dim() - 1 or 0 in t.shape:
        return None
    x = t.movedim(dim, 0)
    return x.view(-1, x.shape[-1]) if x.is_contiguous() else None


def _b_transposed(B):
    """True for W.t() (column-major), False for row-major, None for anything else."""
    if B.stride() == (1, B.shape[0]):
        return True
    if B.stride() == (B.shape[1], 1):
        return False
    return None


def _operands_ok(a, B):
    """bf16, 2D, on one device, K matching, N > 0 (a is _flat's view, so 2D and contiguous). Pointers are not read:
    they can differ between ranks (the ops align their operands instead, see _aligned())."""
    return (
        a.dtype == torch.bfloat16
        and B.dtype == torch.bfloat16
        and B.device == a.device
        and B.dim() == 2
        and B.shape[0] == a.shape[1]
        and B.shape[1] > 0
    )


def _fits(rows, cols):
    from torch._native import flydsl_utils as fu

    return fu._fits_int32_buffer_span(rows, cols, 2)


def _aligned(t):
    """t, or a copy with the same strides at a 16-byte aligned address (the kernels' vector loads)."""
    return t if t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.preserve_format)


def _tile_dims(tile):
    bm, bn = tile.split("x")
    return int(bm), int(bn)


def _ag_fits(lm, k, n, world, tile=TILES[0]):
    """Whether the kernel takes a [lm, K] shard per rank times a [K, N] B at `tile`."""
    # Tile rows: the XCD-interleaved order needs whole groups of 32 and tiles in 8s. The flagged chunks (CHUNK_ROWS
    # rows) hold whole tile rows.
    bm, bn = _tile_dims(tile)
    tm = lm * world // bm
    if (
        lm % CHUNK_ROWS
        or CHUNK_ROWS % bm
        or n % bn
        or k % 128
        or tm % 32
        or tm * (n // bn) % 8
    ):
        return False
    # The gathered A and the output are addressed per tile; B's buffer offsets (one descriptor over K x N) and
    # _stage3's int32 offsets into the shard are still 32-bit.
    return _fits(k, n) and _fits(lm, k)


def _ag_tile(lm, k, n, world):
    """The tile of an all-gather the default tile admits. TILE_ENV forces one (the default if it doesn't fit); "auto"
    takes AUTO_TILE for the calls described at AUTO_*, else the default."""
    want = os.environ.get(TILE_ENV, "auto")
    if want != "auto":
        if want not in TILES:
            raise ValueError(f"{TILE_ENV}={want!r}: expected auto or one of {TILES}")
    elif lm <= AUTO_MAX_LM and n * world <= AUTO_MAX_N:
        want = AUTO_TILE
    else:
        want = TILES[0]
    return want if want == TILES[0] or _ag_fits(lm, k, n, world, want) else TILES[0]


def _ag_meta(A_shard, Bs, gather_dim, world):
    """(K, N, B transposed, i) of the kernel's GEMM if it takes this call, else None. With several Bs (the
    micro-pipeline pass sends q / k / v and w1 / w3 together) the kernel multiplies the widest one it admits, Bs[i],
    and torch.mm the gathered A by the others. Reads only metadata."""
    a = _flat(A_shard, gather_dim)
    if a is None or not Bs or world not in WORLDS:
        return None
    if not all(_operands_ok(a, B) for B in Bs):
        return None
    # AG_MAX_N was measured per call against the stock op, which hides its copy behind every matmul of the call, so
    # it bounds the call's total N (2026-11-17, TP 2 gate-up 14336 x 2 at M 16384: stock 4796 us, FlyDSL on one B +
    # torch.mm 5815 us).
    if sum(B.shape[1] for B in Bs) > AG_MAX_N:
        return None
    (lm, k), lays = a.shape, [_b_transposed(B) for B in Bs]
    ok = [
        i
        for i, B in enumerate(Bs)
        if lays[i] is not None and _ag_fits(lm, k, B.shape[1], world)
    ]
    if not ok:
        return None
    i = max(ok, key=lambda i: Bs[i].shape[1])
    return k, Bs[i].shape[1], lays[i], i


def ag_supported(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    gather_dim: int,
    group_name: c10d.GroupName,
) -> bool:
    """Whether all_gather_matmul takes this call (the same answer on every rank, see the module doc)."""
    if A_shard.device.type != "cuda" or not _available(A_shard.device.index):
        return False
    world = c10d._resolve_process_group(group_name).size()
    return _ag_meta(A_shard, Bs, gather_dim, world) is not None


def _rs_config(k, n, world):
    """(G, C, R) of a reduce-scatter state: the step grid and the reducer blocks."""
    return (
        RS_G,
        RS_C_WIDE if n >= RS_WIDE_N else RS_C,
        R.get((k, n, world), R_DEFAULT[world]),
    )


def _rs_meta(A, B, reduce_op, scatter_dim, world):
    """(K, N, B transposed) if the fused matmul + reduce-scatter kernel takes this call, else None. Reads only
    metadata."""
    a = _flat(A, scatter_dim)
    if a is None or reduce_op != "sum" or world not in WORLDS or not _operands_ok(a, B):
        return None
    b_t = _b_transposed(B)
    (m, k), n = a.shape, B.shape[1]
    br = RS_UNIT_ROWS[world]
    # Tiles per unit of a chunk: the XCD-interleaved order takes them in groups of 8 g.
    g = br // 256 * (n // 256)
    # The flattened rows split into the ranks' chunks only where the scatter dim does.
    if b_t is None or A.shape[scatter_dim] % world or (m // world) % br:
        return None
    if n % 256 or k % 128 or (m // 256) * (n // 256) % (8 * g):
        return None
    if br * n % (512 * 8 * 2):  # whole reducer items per unit
        return None
    # A and the staging are addressed per tile; B's buffer offsets (one descriptor over K x N) and the TP 8 reducers'
    # (one descriptor per chunk, M / TP x N) are still 32-bit.
    if not (_fits(k, n) and _fits(m // world, n)):
        return None
    if world == 8:
        # The TP 8 shapes the kernel is validated on: a unit's tile rows x N % 1024 (an XCD takes 4 tile rows x 1/4 of
        # the tile columns of an 8-row step); others stay on the stock path.
        rows = world * (br // 256)
        if rows % 4 or 8 % (rows // 4) or world * g % 32:
            return None
    return k, n, b_t


def rs_supported(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: c10d.GroupName,
) -> bool:
    """Whether matmul_reduce_scatter takes this call (the same answer on every rank, see the module doc)."""
    if A.device.type != "cuda" or not _available(A.device.index):
        return False
    world = c10d._resolve_process_group(group_name).size()
    return _rs_meta(A, B, reduce_op, scatter_dim, world) is not None


def wait_ms() -> int:
    """TORCH_SYMM_MEM_FLYDSL_TIMEOUT_MS (default 60000; 0 = wait forever), read once per state before its rendezvous:
    how long a fused kernel's tile / reducer waits for a peer before it prints a [FATAL] line and traps, and the
    symmetric-memory barrier's timeout from the state's second call on (the first call's peers may still be
    compiling). The trap fails the launch and torch raises at the next sync; some HIP runtimes abort the process
    instead."""
    raw = os.environ.get(TIMEOUT_ENV, "60000")
    try:
        ms = int(raw)
    except ValueError:
        ms = -1
    if not 0 <= ms < 2**31:
        raise ValueError(
            f"{TIMEOUT_ENV}={raw!r}: expected milliseconds in [0, 2^31), 0 for no timeout"
        )
    return ms


def check_group_agrees(op, group, what):
    """Before a state's rendezvous: wait (the group store's timeout) until every rank of `group` is building the same
    state. A rank that took another branch never arrives, and the rendezvous would otherwise leave this one waiting
    with no hint why."""
    store = c10d._get_process_group_store(group)
    n = _states_built[group.group_name]
    _states_built[group.group_name] += 1
    keys = [f"symm_mem_flydsl/{n}/{r}" for r in range(group.size())]
    store.set(keys[group.rank()], what)
    try:
        store.wait(keys)
    except RuntimeError as e:  # DistStoreError on timeout
        raise RuntimeError(
            f"{op} (FlyDSL): not every rank of group {group.group_name} reached the FlyDSL path for {what}. "
            f"The FlyDSL branch must be enabled alike on every rank ({_AGREE})"
        ) from e
    seen = {store.get(k).decode() for k in keys}
    if seen != {what}:
        raise RuntimeError(
            f"{op} (FlyDSL): the ranks of group {group.group_name} are building different states: {sorted(seen)}"
        )


def _control_block(ms):
    """The fused kernels' int32 control block: [epoch, timeout reporters, timeout ms (2^31 - 1 for none), unused]."""
    return torch.tensor(
        [0, 0, ms if ms > 0 else 2**31 - 1, 0], dtype=torch.int32, device="cuda"
    )


def _order_streams(st):
    """Order this call on state `st` (current device) after the state's previous call: a state's epoch and staging
    assume in-order calls (one state on two unordered streams hangs the AG and corrupts the RS); different states are
    independent. Under CUDA graph capture nothing is waited on: torch.cuda.graph synchronizes before capture, and a
    capturing stream may not wait on uncaptured work, so within one capture keep a state's calls on one stream. Nor
    is a replay ordered against eager calls: don't replay a graph while eager calls (or another replay) on the same
    state may run."""
    cur = torch.cuda.current_stream()
    if (
        st.stream is not None
        and st.stream != cur
        and not torch.cuda.is_current_stream_capturing()
    ):
        cur.wait_stream(st.stream)
    st.stream = cur


def _rendezvous(buf, group_name):
    """symm_mem.rendezvous(buf), checked to cover the whole allocation: the kernels read the peers' buffers at
    buffer_ptrs_dev and ignore the handle's offset."""
    import torch.distributed._symmetric_memory as symm_mem

    hdl = symm_mem.rendezvous(buf, group_name)
    if hdl.offset != 0:
        raise RuntimeError(
            "FlyDSL fused AsyncTP kernels read peers via buffer_ptrs_dev and require a whole-allocation "
            f"symmetric-memory handle (offset 0), got offset {hdl.offset}"
        )
    return hdl


def _group_m(tpr):
    # The tile order starts XCD x on order positions [x * GROUP_M, (x + 1) * GROUP_M): this rank's own rows, which
    # wait for no chunk, only if 8 * GROUP_M <= tpr (tile rows per rank); else whole XCDs start on peer rows and stall
    # on the copy. Smaller groups reuse B less, so take the largest that fits.
    return next((g for g in (4, 2, 1) if 8 * g <= tpr), 1)


class _AGState:
    def __init__(self, m, k, n, b_t, kc, group, b, tile=TILES[0]):
        import flydsl.compiler as flyc

        import torch.distributed._symmetric_memory as symm_mem
        from torch.distributed._symmetric_memory import _fused_flydsl_kernels as K

        rank, world = group.rank(), group.size()
        self.K, self.m, self.k, self.n, self.kc = K, m, k, n, kc
        self.tile = tile
        bm, bn = _tile_dims(tile)
        self.rank, self.world, self.group = rank, world, group
        self.ms = wait_ms()
        lm = m // world
        self.shard, self.nch = lm * k // 2, lm // CHUNK_ROWS
        self.shard_b, self.ch_b = lm * k * 2, CHUNK_ROWS * k * 2
        if lm % CHUNK_ROWS or kc % (world - 1) or self.ch_b % (512 * 16 * 8):
            raise RuntimeError(
                f"fused_all_gather_matmul (FlyDSL): {lm} rows per rank, K {k} and "
                f"{kc} copy blocks don't fit the chunking"
            )
        if not _ag_fits(lm, k, n, world, tile):
            raise RuntimeError(
                f"fused_all_gather_matmul (FlyDSL): M {m}, N {n} don't fit the "
                f"XCD-interleaved tile order of {tile} tiles"
            )
        check_group_agrees(
            "fused_all_gather_matmul",
            group,
            f"fused_all_gather_matmul M {m} K {k} N {n} B transposed {b_t} tile {tile}",
        )
        self.buf = symm_mem.empty(
            2 * self.shard, dtype=torch.int32, device=torch.device("cuda")
        )
        self.hdl = _rendezvous(self.buf, group.group_name)
        self.flags = torch.zeros(
            world * self.nch * SLOT, dtype=torch.int32, device="cuda"
        )
        self.ep = _control_block(self.ms)
        self.bar_ms = 0
        self.sink = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.mult = kc // (world - 1)
        self.prm = K.gemm_param(m, n, k, b_t, _group_m(lm // bm), (bm, bn))
        self.calls, self.stream = 0, None
        # Compiling also runs the kernel once: with kc 0 (no copy blocks) on flags far past any target.
        never = torch.full_like(self.flags, 1 << 30)
        a0 = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
        c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        s = torch.cuda.current_stream().cuda_stream
        self.gemm = flyc.compile(
            K.agmm_gfx950,
            K.arg(c),
            K.arg(a0),
            K.arg(b),
            never.data_ptr(),
            self.ep.data_ptr(),
            self.mult,
            SLOT * 4,
            rank,
            world,
            self.nch,
            self.hdl.buffer_ptrs_dev,
            a0.data_ptr(),
            0,
            self.shard_b,
            self.ch_b,
            self.prm,
            s,
        )
        torch.cuda.synchronize()

    def __call__(self, a_shard, b):
        # Everything that can raise comes before _bump: the epoch it advances is never rolled back, and every wait
        # target is a multiple of it, so a call that bumps and then raises leaves this state out of step for good.
        K = self.K
        gat = torch.empty(self.m, self.k, dtype=torch.bfloat16, device="cuda")
        c = torch.empty(self.m, self.n, dtype=torch.bfloat16, device="cuda")
        src = a_shard.view(torch.int32)
        own = gat.view(torch.int32).view(-1)[self.rank * self.shard :]
        K._bump[(1,)](self.ep, num_warps=1)
        K._stage3[(-(-self.shard // STAGE_BLOCK),)](
            src,
            self.buf,
            own,
            self.ep,
            self.shard,
            BUFS=2,
            BLOCK=STAGE_BLOCK,
        )
        K._flush[(FLUSH_PROGRAMS,)](self.sink, num_warps=1)
        self.hdl.barrier(channel=0, timeout_ms=self.bar_ms)
        self.bar_ms = self.ms
        self.gemm(
            c,
            gat,
            b,
            self.flags.data_ptr(),
            self.ep.data_ptr(),
            self.mult,
            SLOT * 4,
            self.rank,
            self.world,
            self.nch,
            self.hdl.buffer_ptrs_dev,
            gat.data_ptr(),
            self.kc,
            self.shard_b,
            self.ch_b,
            self.prm,
            torch.cuda.current_stream().cuda_stream,
        )
        return gat, c


def _ag_2d(a, b, k, n, b_t, group):
    """The gathered A [M, K] and A @ b [M, N] for a contiguous, aligned 2D shard a."""
    world, lm = group.size(), a.shape[0]
    tile = _ag_tile(lm, k, n, world)
    skey = ("ag", k, n, world, b_t, lm, group.group_name, a.device.index, tile)
    with torch.cuda.device(a.device):
        st = _states.get(skey)
        # A group destroyed and re-created under the same name: the state's handles belong to the old one.
        if st is None or st.group is not group:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused_all_gather_matmul (FlyDSL): the first call at (K {k}, N {n}, TP {world}, rows {lm}) is "
                    "under CUDA graph capture; it compiles and rendezvouses, so make one eager call first"
                )
            kc = KC.get((k, n, world), KC_DEFAULT[world])
            st = _states[skey] = _AGState(lm * world, k, n, b_t, kc, group, b, tile)
        _order_streams(st)
        gat, c = st(a, b)
    st.calls += 1
    return gat, c


def all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    gather_dim: int,
    group_name: c10d.GroupName,
    *,
    return_A: bool = True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """symm_mem::fused_all_gather_matmul for a call ag_supported() admits, with the Meta kernel's output strides."""
    group = c10d._resolve_process_group(group_name)
    k, n, b_t, i = _ag_meta(A_shard, Bs, gather_dim, group.size())
    # As _fused_all_gather_matmul_impl: gather dim first, flattened to 2D; unflatten reverts it on the gathered rows.
    lead = [group.size(), *A_shard.movedim(gather_dim, 0).shape[:-1]]

    def unflatten(t):
        return t.view(*lead, -1).flatten(0, 1).movedim(0, gather_dim)

    a = _aligned(_flat(A_shard, gather_dim))
    gat, c = _ag_2d(a, _aligned(Bs[i]), k, n, b_t, group)
    outs = [c if j == i else torch.mm(gat, b) for j, b in enumerate(Bs)]
    _calls["ag"] += 1
    return (unflatten(gat) if return_A else None), [unflatten(o) for o in outs]


class _RSState:
    def __init__(self, m, k, n, b_t, r, grid, group, a0, b):
        import flydsl.compiler as flyc

        import torch.distributed._symmetric_memory as symm_mem
        from torch.distributed._symmetric_memory import _fused_flydsl_kernels as K

        rank, world = group.rank(), group.size()
        self.K, self.m, self.k, self.n, self.r = K, m, k, n, r
        self.rank, self.world, self.group, self.ms = rank, world, group, wait_ms()
        self.lm = m // world
        nm, npn = self.lm // 256, n // 256

        # The requested (G, C) fitted down to divisors of the tile rows per chunk and the tile columns that make a step
        # (world * tile rows per group * tile columns per slab tiles) a multiple of 8. The slab counts to try are the
        # divisors of the tile columns in 2..C, widest first (up to 16 if there is none); C = 1 only for one tile column.
        def _cols(npn, cap):
            cs = [c for c in range(min(cap, npn), 1, -1) if npn % c == 0]
            return cs or [c for c in range(min(16, npn), 1, -1) if npn % c == 0] or [1]

        fit = [
            (g_, c_)
            for g_ in range(min(grid[0], nm), 0, -1)
            for c_ in _cols(npn, grid[1])
            if nm % g_ == 0
            and npn % c_ == 0
            and world * (nm // g_) * (npn // c_) % 8 == 0
        ]
        if n % 256 or self.lm % 256 or not fit:
            raise RuntimeError(
                f"fused_matmul_reduce_scatter (FlyDSL): M {m}, N {n} admit no step grid at TP {world}"
            )
        # nb = G * C steps (step t = g * C + c), rpb = tile columns per column slab.
        self.nb, self.rpb = fit[0][0] * fit[0][1], npn // fit[0][1]
        check_group_agrees(
            "fused_matmul_reduce_scatter",
            group,
            f"fused_matmul_reduce_scatter M {m} K {k} N {n} B transposed {b_t}",
        )
        self.buf = symm_mem.empty(
            m * n + self.nb * 32 * 2,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        self.hdl = _rendezvous(self.buf, group.group_name)
        self.stage = self.buf[: m * n].view(m, n)
        self.cnt = self.buf[m * n :].view(torch.int32)
        self.cnt.zero_()  # peers add only after this call's barrier, which this rank reaches after the zeroing
        self.cnt_off = m * n * 2
        self.ep = _control_block(self.ms)
        self.bar_ms = 0
        self.prm = K.gemm_param(m, n, k, b_t)
        self.calls, self.stream = 0, None
        # Compiling also runs the kernel once with no reducers and every peer entry on this rank's own staging, counter
        # offset 0: counter adds land on staging words every call rewrites.
        table = torch.full(
            (world,), self.stage.data_ptr(), dtype=torch.int64, device="cuda"
        )
        s = torch.cuda.current_stream().cuda_stream
        self.gemm = flyc.compile(
            K.rsmm_gfx950,
            K.arg(self.stage),
            K.arg(a0),
            K.arg(b),
            table.data_ptr(),
            0,
            self.ep.data_ptr(),
            self.stage.data_ptr(),
            rank,
            world,
            0,
            self.nb,
            self.rpb,
            self.lm,
            self.prm,
            s,
        )
        torch.cuda.synchronize()

    def __call__(self, a, b):
        # Everything that can raise comes before _bump: the epoch it advances is never rolled back, and every wait
        # target is a multiple of it, so a call that bumps and then raises leaves this state out of step for good.
        out = torch.empty(self.lm, self.n, dtype=torch.bfloat16, device="cuda")
        self.K._bump[(1,)](self.ep, num_warps=1)
        self.hdl.barrier(channel=0, timeout_ms=self.bar_ms)
        self.bar_ms = self.ms
        self.gemm(
            self.stage,
            a,
            b,
            self.hdl.buffer_ptrs_dev,
            self.cnt_off,
            self.ep.data_ptr(),
            out.data_ptr(),
            self.rank,
            self.world,
            self.r,
            self.nb,
            self.rpb,
            self.lm,
            self.prm,
            torch.cuda.current_stream().cuda_stream,
        )
        return out


def _rs_2d(a, b, k, n, b_t, group):
    """This rank's chunk [M / TP, N] of the sum over ranks of a @ b, for a contiguous, aligned 2D a."""
    world, m = group.size(), a.shape[0]
    g, c, r = _rs_config(k, n, world)
    skey = ("rs", k, n, world, b_t, m // world, group.group_name, a.device.index)
    with torch.cuda.device(a.device):
        st = _states.get(skey)
        # A group destroyed and re-created under the same name: the state's handles belong to the old one.
        if st is None or st.group is not group:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused_matmul_reduce_scatter (FlyDSL): the first call at (K {k}, N {n}, TP {world}, rows "
                    f"{m // world}) is under CUDA graph capture; it compiles and rendezvouses, so make one eager "
                    "call first"
                )
            st = _RSState(m, k, n, b_t, r, (g, c), group, a, b)
            _states[skey] = st
        _order_streams(st)
        out = st(a, b)
    st.calls += 1
    return out


def matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    """symm_mem::fused_matmul_reduce_scatter for a call rs_supported() admits, with the Meta kernel's output strides."""
    group = c10d._resolve_process_group(group_name)
    k, n, b_t = _rs_meta(A, B, reduce_op, scatter_dim, group.size())
    # As _fused_matmul_reduce_scatter_impl: scatter dim first, flattened to 2D; this rank's rows are its share of it.
    lead = list(A.movedim(scatter_dim, 0).shape[:-1])
    lead[0] //= group.size()
    out = _rs_2d(_aligned(_flat(A, scatter_dim)), _aligned(B), k, n, b_t, group)
    _calls["rs"] += 1
    # Contiguous like the Meta kernel's output: a copy only for a batch scattered on a dim after the first.
    return out.view(*lead, -1).movedim(0, scatter_dim).contiguous()
