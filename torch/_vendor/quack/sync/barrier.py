# Copyright (c) 2026, Tri Dao.

"""Global-memory synchronization helpers for CuTe DSL kernels.

These mirror the counter-semaphore pattern used by CUTLASS C++
(`cutlass/barrier.h` / `cutlass/semaphore.h`): one elected thread spins on an
acquire load of a global flag, and one elected thread publishes progress with
a gpu-scope release write.  Two publish flavors exist, differing only in how
the flag is updated:

- ``release_store`` writes an absolute value.  Valid only when the caller is
  the flag's exclusive publisher at that point in the protocol (e.g. inside a
  turnstile, after ``wait_eq`` granted ownership) — this is the deterministic,
  serially-ordered flavor.
- ``release_add`` release-increments.  It commutes, so arrivers may publish in
  any order — the arrival-counter flavor.

Scope is deliberately narrow: flag operations plus an optional group-sync
pairing on :class:`Semaphore`.  Everything that depends on what the caller did
stays with the caller: draining its own async-proxy work (e.g. TMA stores)
before releasing, and any async-proxy fence needed when post-wait consumers
read the protected data through the TMA proxy rather than generic loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr


@cute.jit
def _wait_eq_spin(flag_ptr: cute.Pointer, thread_idx: Int32, val: Int32) -> None:
    """Elected-thread acquire spin. Always issues at least one acquire load, so
    the caller gets an ordering edge even when the flag already equals ``val``."""
    if thread_idx == 0:
        read_val = cute.arch.load(flag_ptr, Int32, sem="acquire", scope="gpu")
        while read_val != val:
            read_val = cute.arch.load(flag_ptr, Int32, sem="acquire", scope="gpu")


@cute.jit
def wait_eq(
    lock_ptr: cute.Pointer,
    thread_idx: Int32,
    flag_offset: Int32 | Int64,
    val: Int32,
    skip_zero: bool = False,
    sync: Literal["none", "warp", "cta"] = "none",
) -> None:
    """Wait until ``lock_ptr[flag_offset] == val`` using ``thread_idx == 0``.

    ``skip_zero`` statically wraps the whole wait (including ``sync``) in
    ``if val != 0`` for protocols where 0 means "no predecessor" — note the
    acquire load is skipped too, so there is no ordering edge in that case.
    """
    if const_expr(not skip_zero) or val != 0:
        _wait_eq_spin(lock_ptr + flag_offset, thread_idx, val)
        if const_expr(sync == "warp"):
            cute.arch.sync_warp()
        elif const_expr(sync == "cta"):
            cute.arch.sync_threads()


@cute.jit
def release_store(
    lock_ptr: cute.Pointer,
    thread_idx: Int32,
    flag_offset: Int32 | Int64,
    val: Int32,
) -> None:
    """gpu-scope release store of ``val`` to ``lock_ptr[flag_offset]`` by
    ``thread_idx == 0``.

    Precondition: the caller is the flag's sole publisher at this point in the
    protocol (turnstile ownership).  The caller orders the group's prior writes
    before this call (warp sync / named barrier) — the release store only
    publishes the electing thread's happens-before chain.
    """
    if thread_idx == 0:
        cute.arch.store(lock_ptr + flag_offset, val, sem="release", scope="gpu")


@cute.jit
def release_add(
    lock_ptr: cute.Pointer,
    thread_idx: Int32,
    flag_offset: Int32 | Int64,
    val: Int32 = 1,
) -> None:
    """Release-increment ``lock_ptr[flag_offset]`` by ``val`` using
    ``thread_idx == 0``.  Commutes across arrivers; see module docstring."""
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        cute.arch.red(flag_ptr, Int32(val), op="add", dtype="s32", sem="release", scope="gpu")


@dataclass(frozen=True)
class Semaphore:
    """Global-memory counter semaphore with an injected group-sync policy.

    Bundles the flag pointer, the electing thread index, and the sync policy so
    call sites read like the protocol (CUTLASS ``GenericBarrier<Sync>``):

    .. code-block:: python

        sem = Semaphore(lock_ptr, tidx, sync=epilogue_barrier)
        sem.wait_eq(split_idx)          # spin acquire, then broadcast to group
        ...commit partials...
        sem.release_store(split_idx + 1)  # collect group, then publish

    The pairing is fixed by semantics: sync AFTER ``wait_eq`` (broadcast the
    acquired edge to the group), sync BEFORE ``release_store``/``release_add``
    (order the group's writes before the flag write).  ``sync`` may be
    ``"none"``, ``"warp"``, ``"cta"``, or any object with ``arrive_and_wait()``
    (e.g. a ``NamedBarrier`` over the participating group — then every group
    thread must call the method, and ``thread_idx`` is group-relative).

    The semaphore knows nothing about fences or async-proxy draining: complete
    your own TMA work before releasing, and fence the async proxy after
    waiting if (and only if) protected data is then read through it.
    """

    lock_ptr: cute.Pointer
    thread_idx: int | Int32
    sync: Union[Literal["none", "warp", "cta"], object] = "none"

    def _sync(self) -> None:
        if isinstance(self.sync, str):
            if self.sync == "warp":
                cute.arch.sync_warp()
            elif self.sync == "cta":
                cute.arch.sync_threads()
        else:
            self.sync.arrive_and_wait()

    def wait_eq(
        self,
        val: int | Int32,
        flag_offset: Optional[int | Int32 | Int64] = None,
        skip_zero: bool = False,
    ) -> None:
        flag_offset = 0 if flag_offset is None else flag_offset
        wait_eq(self.lock_ptr, self.thread_idx, flag_offset, val, skip_zero=skip_zero)
        # Unconditional (unlike free-fn skip_zero): with a barrier sync policy
        # every group thread must reach the sync regardless of the flag value.
        self._sync()

    def release_store(
        self, val: int | Int32, flag_offset: Optional[int | Int32 | Int64] = None
    ) -> None:
        flag_offset = 0 if flag_offset is None else flag_offset
        self._sync()
        release_store(self.lock_ptr, self.thread_idx, flag_offset, val)

    def release_add(
        self, val: int | Int32 = 1, flag_offset: Optional[int | Int32 | Int64] = None
    ) -> None:
        flag_offset = 0 if flag_offset is None else flag_offset
        self._sync()
        release_add(self.lock_ptr, self.thread_idx, flag_offset, val)


# More explicit alias for call sites where plain ``Semaphore`` is ambiguous.
GlobalSemaphore = Semaphore


__all__ = [
    "Semaphore",
    "GlobalSemaphore",
    "wait_eq",
    "release_store",
    "release_add",
]
