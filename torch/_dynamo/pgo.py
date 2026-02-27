"""
Profile Guided Optimization (PGO) implementation for Dynamo.

This module provides functionality for caching and managing code state profiles
that guide optimization decisions in Dynamo. It implements both local and remote
caching mechanisms for storing profile information across runs, handles profile
merging across distributed ranks, and manages the lifecycle of profile data
during compilation. The profiles track dynamic vs static properties of tensors
and help Dynamo make better specialization decisions.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import enum
import functools
import logging
import os
import pickle
import re
import zlib
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import override, Self

import torch._dynamo.config
import torch._utils_internal
import torch.compiler.config
import torch.distributed as dist
from torch._dynamo.utils import (
    CompileEventLogger,
    dynamo_timed,
    set_feature_use,
    warn_once,
)
from torch._environment import is_fbcode
from torch._logging._internal import trace_structured_artifact
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    import types

    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._inductor.remote_cache import JsonDataTy, RemoteCache


class ReservedWorkflowIdUserError(ValueError):
    pass


log = logging.getLogger(__name__)

LOCK_TIMEOUT = 10

# How does in memory representation work?  Concretely, this module is
# responsible for holding GLOBAL state representing the state it holds, no
# other copies permitted.  So we retire frame_state entirely and store it
# here.  This should be reset when Dynamo is reset.  We never GC information
# (similar to how the filesystem doesn't get cleaned up except by tmp
# cleaner), so the expectation is the information is relatively cheap and we
# don't mind leaking it.


# How exactly did we design the cache key?  Here are some of the questions:
#
# - JOB_ID: Do we have a unique identifier for the "training run"  (such that
#   it stays the same if we're running the same code, and changes if we're
#   running something different).
#
# - RANK: Are we sharing the cache across ranks, or does each rank get
#   an individual cache?
#
# We choose to require job_id for PGO cache.  This is to prevent
# situations where unrelated invocations of PyTorch unpredictably cause
# changes to each other's behavior.  With a job_id, at least you know there
# is some "state" associated with it.  (State dict might be another way to
# tell if a run is related or not.)  You can opt-in to YOLO everything
# aliases everything by passing a shared job_id for all your invocations.
#
# We choose to NOT share PGO cache across ranks.  With no RANK_SHARING, there
# is never contention between runs, so we can leisurely update a bundle with
# information we need.  Because we are grouped by job_id, we can have a single
# consolidated bundle for everything (or not; maybe worry about O(n^2) IO if
# we updated every compile--let's just instrument this.)  Can even take a
# filelock for extra safety (expect no contention); expect 50ns overhead from
# uncontended filelock.
#
# If we did share ranks, everyone is storming to modify the same cache files.
# We can do this by having folks atomic write to a CAS-store and then having
# readers do on-the-fly merging (this can be implemented in remote using
# prefix iteration).  As an optional optimization, one rank can be elected to
# handling bundling post facto (ideally, this is done async, after quiescence,
# without compiler collective need to wait for everyone to finish writing
# their bits.) Not sure how you can avoid a listdir because if some rank shows
# up with some new entries we need to pull them in ASAP (unless you want to
# delay bundling).
#
# But compiler collectives fill a similar niche:  compilers chat with each
# other so rank 0 has collected everything.  So elect rank 0 only to write the
# bundle.  Don't even need CAS-store atomic write; just one rank writing an
# updating bundles.  The point is that use compiler collectives to share
# profiles across ranks, but use the PGO cache to persist profiles per rank
# across attempts.  No need to have one mechanism to do everything.


@functools.cache
def _hash_containing_file(filepath: str) -> str:
    # if the file does not exists we consider filepath to be the hash.
    if not os.path.exists(filepath):
        return filepath

    with open(filepath, "rb") as file:
        content = file.read()
        crc32_value = zlib.crc32(content)
        hash = format(crc32_value & 0xFFFFFFFF, "08x")
        return hash


@dataclasses.dataclass(frozen=True)
class CodeId:
    filename: str
    firstlineno: int
    name: str
    # When a job restart, the code can be copied to a different path than the previous attempt. In that case
    # self.filename will have a different value,  we do not want to consider those differences. Instead we
    # hash the content of the file and use it as an identifier of the file.
    #
    # self.filename is kept in the object to give readable information/pointer to the actual file, in a local
    # code state it will refer to the first seen file path.
    file_hash: str

    # Exclude file name.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodeId):
            return False
        return (
            self.file_hash == other.file_hash
            and self.firstlineno == other.firstlineno
            and self.name == other.name
        )

    # Ensure if two CodeIds are the same, then they have the same hash by excluding filename.
    def __hash__(self) -> int:
        return hash((self.file_hash, self.name, self.firstlineno))

    def __str__(self) -> str:
        return f"hash({self.file_hash}){self.filename}:{self.firstlineno}:{self.name}"

    @staticmethod
    def make(code: types.CodeType) -> CodeId:
        return CodeId(
            code.co_filename,
            code.co_firstlineno,
            code.co_name,
            _hash_containing_file(code.co_filename),
        )


@dataclasses.dataclass
class CodeState:
    automatic_dynamic: defaultdict[str, FrameStateSizeEntry] = dataclasses.field(
        default_factory=lambda: defaultdict(FrameStateSizeEntry)
    )


_INIT_CODE_STATE: defaultdict[CodeId, CodeState] | None = None
_CODE_STATE: defaultdict[CodeId, CodeState] | None = None
_LOGGED_DYNAMIC_ALLOWLIST: bool = False
_KNOWN_DYNAMIC_SOURCES: set[str] = set()


@dataclasses.dataclass(frozen=True)
class InferStride:
    """
    Denotes the quantity stride[dim] * size[dim], which is what the stride would
    be for the next physical dimension that results in a contiguous layout.

    For example, given size = [2, 3], stride = [3, 1], we can replace this with
    stride = [InferStride(1), 1], because InferStride(1) = stride[1] * size[1] = 1 * 3 = 3

    Indirecting the representation in this way is important for the join operation
    on strides as if we join [2, 3][3, 1] and [2, 4][4, 1],
    we don't want [2, None][None, 1] which would get eventually symbolized into
    [2, s0][s1, 1] (notice that the relationship between s0 and s1 is broken).
    If we instead rewrite the expressions as InferStride so we have [2, 3][InferStride(1), 1]
    and [2, 4][InferStride(1), 1] we now join to [2, None][InferStride(1), 1] will
    result in [2, s0][s0, 1], as desired.
    """

    dim: int


_T = TypeVar("_T")


class AutoUnset(enum.Enum):
    """
    The identity element of our semilattice, a generic "don't know" element that
    is always subsumed when we get more information.
    """

    token = 0


auto_unset = AutoUnset.token


class AutoDynamic(enum.Enum):
    """
    The top element of our (bounded) semilattice, whenever you merge this with
    any other element you always get it again
    """

    token = 0


auto_dynamic = AutoDynamic.token


@dataclasses.dataclass
class FrameStateSizeEntry:
    scalar: int | AutoDynamic | AutoUnset = dataclasses.field(default=auto_unset)
    # NB: We don't have cases where we have a known dimensionality but
    # we know NOTHING about the individual sizes
    size: AutoDynamic | AutoUnset | tuple[int | AutoDynamic, ...] = dataclasses.field(
        default=auto_unset
    )
    stride: AutoDynamic | AutoUnset | tuple[int | AutoDynamic | InferStride, ...] = (
        dataclasses.field(default=auto_unset)
    )

    def render(self) -> str:
        # Special cases
        def render_single(s: int | AutoDynamic | AutoUnset | InferStride) -> str:
            if s is auto_dynamic:
                return "?"
            elif s is auto_unset:
                # This basically shouldn't happen, this is for debugging
                return "auto unset"
            elif isinstance(s, InferStride):
                return f"S({s.dim})"
            else:
                return str(s)

        def render_tuple(ss: tuple[int | AutoDynamic | InferStride, ...]) -> str:
            return "[" + ", ".join(render_single(s) for s in ss) + "]"

        # Common cases
        if self.size is auto_dynamic and self.stride is auto_dynamic:
            if self.scalar is auto_dynamic:
                return "fully dynamic scalar or tensor"
            else:
                return f"scalar {self.scalar}"
        elif self.scalar is auto_dynamic:
            if isinstance(self.size, tuple) and isinstance(self.stride, tuple):
                return f"tensor size={render_tuple(self.size)} stride={render_tuple(self.stride)}"

        # Fallback
        return f"unusual {repr(self)}"

    def __post_init__(self) -> None:
        assert not isinstance(self.scalar, torch.SymInt), self.scalar
        if isinstance(self.size, tuple):
            for s in self.size:
                assert not isinstance(s, torch.SymInt), s
        if isinstance(self.stride, tuple):
            for s1 in self.stride:
                assert not isinstance(s1, torch.SymInt), s1

    def is_size_dynamic(self, dim: int) -> bool:
        if self.size is auto_dynamic:
            return True
        if self.size is auto_unset:
            return False
        return self.size[dim] is auto_dynamic

    def is_stride_dynamic(self, dim: int) -> bool:
        # At the moment, dynamic strides is a bit buggy.  Good test case
        # here is `PYTORCH_TEST_WITH_DYNAMO=1 python test/test_autograd.py
        # TestAutograd.test_gradcheck_jacobian_mismatch`
        #
        # This if statement preserves historical behavior, which is that we
        # ONLY make strides dynamic if the size is exactly static everywhere.
        # We could potentially relax this but in general we should be very
        # careful about when to infer dynamic strides.
        #
        # Actually, the existing algorithm is already somewhat problematic.
        # Suppose a tensor that is sometimes:
        # f32[2, 3, 5][15, 5, 1] and other times
        # f32[2, 3, 5][5, 10, 1] (specifically, dim 0 and 1 are physically transposed).
        # If we infer strides should be (DYNAMIC, DYNAMIC, 1).  But this is
        # silly: we really should have just guarded on dim order.
        if not (
            isinstance(self.size, tuple) and all(type(s) is int for s in self.size)
        ):
            return False
        if self.stride is auto_dynamic:
            return True
        if self.stride is auto_unset:
            return False
        return self.stride[dim] is auto_dynamic

    @staticmethod
    def _munge_symint(xs: tuple[int, ...]) -> tuple[AutoDynamic | int, ...]:
        return tuple(auto_dynamic if isinstance(x, torch.SymInt) else x for x in xs)

    @classmethod
    def make_scalar(cls, x: int) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(scalar=x, size=auto_dynamic, stride=auto_dynamic)

    @classmethod
    def make_tensor(
        cls, size: tuple[int, ...], stride: tuple[int, ...]
    ) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(
            scalar=auto_dynamic,
            size=cls._munge_symint(size),
            stride=cls._munge_symint(stride),
        )

    @classmethod
    def make_size(cls, size: tuple[int, ...]) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(
            scalar=auto_unset,
            size=cls._munge_symint(size),
            stride=auto_unset,
        )

    @staticmethod
    def _merge_atom(x: _T, y: _T) -> AutoDynamic | _T:
        if x is auto_unset:
            return y
        if y is auto_unset:
            return x
        if x is auto_dynamic or y is auto_dynamic or x != y:
            return auto_dynamic
        return x

    @classmethod
    def _merge_atom_tup(
        cls,
        xs: AutoDynamic | AutoUnset | tuple[_T, ...],
        ys: AutoDynamic | AutoUnset | tuple[_T, ...],
    ) -> AutoDynamic | AutoUnset | tuple[AutoDynamic | _T, ...]:
        if xs is auto_unset:
            return ys
        if ys is auto_unset:
            return xs
        if xs is auto_dynamic or ys is auto_dynamic:
            return auto_dynamic
        if len(xs) != len(ys):
            return auto_dynamic
        return tuple(cls._merge_atom(x, y) for x, y in zip(xs, ys))

    def __ior__(self, other: Self) -> Self:
        self.scalar = self._merge_atom(self.scalar, other.scalar)
        self.size = self._merge_atom_tup(self.size, other.size)
        self.stride = self._merge_atom_tup(self.stride, other.stride)
        return self


def update_automatic_dynamic(
    tx: InstructionTranslator,
    name: str,
    entry: FrameStateSizeEntry,
    *,
    is_unspecialized_nn_module: bool = False,
) -> FrameStateSizeEntry:
    code_id = CodeId.make(tx.f_code)
    frame_state = get_code_state()[code_id]
    if torch._dynamo.config.automatic_dynamic_shapes:
        is_update = name in frame_state.automatic_dynamic
        mut_entry = frame_state.automatic_dynamic[name]
        old_entry = copy.copy(mut_entry)
        mut_entry |= entry

        # Do some logs (damn, I spend more code logging than I do actually doing
        # the updates lol)
        if is_update and old_entry.scalar != mut_entry.scalar:
            log.debug(
                "automatic dynamic int %s val %s != %s",
                name,
                entry.scalar,
                old_entry.scalar,
            )
            CompileEventLogger.instant(
                "automatic_dynamic",
                {
                    "name": name,
                    "dim_changed": "scalar",
                    "reason": "scalar change",
                    "cached": str(old_entry.scalar),
                    "new": str(entry.scalar),
                },
            )
            if is_unspecialized_nn_module:
                log.info(
                    "%s is converted to a symbolic integer. It is an attribute of a "
                    "user defined nn module class. If you wish to keep it static, you can "
                    "mark the nn module class as `torch._dynamo.mark_static`.",
                    name,
                )

        def log_tup(
            tup_name: str, short_reason: str, long_reason: str, i: int | None = None
        ) -> None:
            entry_tup = (
                getattr(entry, tup_name) if i is None else getattr(entry, tup_name)[i]
            )
            old_entry_tup = (
                getattr(old_entry, tup_name)
                if i is None
                else getattr(old_entry, tup_name)[i]
            )
            log.debug(
                "automatic dynamic %s %s %s %s != %s",
                tup_name,
                name,
                short_reason,
                # NB: We used to only report len(...) here for dim mismatch
                entry_tup,
                old_entry_tup,
            )
            CompileEventLogger.instant(
                "automatic_dynamic",
                {
                    "name": name,
                    "dim_changed": "all" if i is None else i,
                    "reason": long_reason,
                    "cached": str(old_entry_tup),
                    "new": str(entry_tup),
                },
            )

        if is_update and old_entry.size != mut_entry.size:
            if isinstance(old_entry.size, tuple) and isinstance(entry.size, tuple):
                if len(old_entry.size) != len(entry.size):
                    log_tup("size", "dim", "dimensionality change")
                else:
                    for i in range(len(entry.size)):
                        if old_entry.size[i] != entry.size[i]:
                            log_tup("size", f"size({i})", "size change", i)
            else:
                log_tup("size", "other", "other")

        if is_update and old_entry.stride != mut_entry.stride:
            if isinstance(old_entry.stride, tuple) and isinstance(entry.stride, tuple):
                if len(old_entry.stride) != len(entry.stride):
                    log_tup("stride", "dim", "dimensionality change")
                else:
                    for i in range(len(entry.stride)):
                        if old_entry.stride[i] != entry.stride[i]:
                            log_tup("stride", f"stride({i})", "stride change", i)
            else:
                log_tup("stride", "other", "other")
    else:
        old_entry = frame_state.automatic_dynamic[name]
        log.debug(
            "automatic dynamic is off, overwriting int %s val %s -> %s",
            name,
            old_entry.scalar,
            entry.scalar,
        )
        frame_state.automatic_dynamic[name] = entry
        mut_entry = entry

    return mut_entry


def process_automatic_dynamic(
    tx: InstructionTranslator,
    name: str,
    entry: FrameStateSizeEntry,
    *,
    is_unspecialized_nn_module: bool = False,
) -> FrameStateSizeEntry:
    if (st := tx.distributed_state) is None:
        return update_automatic_dynamic(
            tx,
            name,
            entry,
            is_unspecialized_nn_module=is_unspecialized_nn_module,
        )
    elif st.all_states is None:
        # Preflight, always pretend as if it's static.  The point here
        # is we want to get through the preflight quickly, and static
        # will run faster.  The preexisting frame state will get
        # applied anyway after we do compiler collectives.
        # TODO: I'm not sure if we should just bong the entire pgo
        # state here, it kind of depends if we're going to have other
        # things that talk in compiler collective.  Also, the PGO
        # state, if we've already inferred something is automatic
        # dynamic, will have lost the actual input sizes, which might
        # be useful for debugging purposes (e.g., observing 0/1
        # specialization).  Bonging the entire PGO state here would
        # let us delete this logic here; the compiler collective
        # would just directly update_automatic_dynamic
        st.local_state.automatic_dynamic[name] = entry
        return entry
    else:
        # Apply the updates.  NB: all_states includes the local state
        # too.
        res = None
        for sub_state in st.all_states:
            if name in sub_state.automatic_dynamic:
                res = update_automatic_dynamic(
                    tx,
                    name,
                    sub_state.automatic_dynamic[name],
                    is_unspecialized_nn_module=is_unspecialized_nn_module,
                )
        assert res is not None
        return res


def format_cache_key(key: str) -> str:
    # NB: We always use global rank for keys, even though they are overkill
    # for local only cache
    rank = None
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    tag = torch.compiler.config.cache_key_tag
    return f"{key}:{rank}:{tag}"


def get_cache_key() -> str | None:
    # TODO: info versions of these logs that log only once
    if torch.compiler.config.force_disable_caches:
        warn_once(
            "dynamo_pgo force disabled by torch.compiler.config.force_disable_caches"
        )
        return None

    # NB: We namespace the cache keys so that only user-specified job id
    # can alias with each other.
    if (r := torch.compiler.config.job_id) is not None:
        if r.startswith("mast:"):
            raise ReservedWorkflowIdUserError(
                "torch.compiler.config.job_id with prefix 'mast:' is reserved for "
                "automatically generated job id associated with a specific MAST job "
                "name and version."
            )
        return format_cache_key(r)

    if (name_version := torch._utils_internal.get_mast_job_name_version()) is not None:
        mast_job_name, mast_job_version = name_version
        return format_cache_key(f"mast:{mast_job_name}:{mast_job_version}")

    return None


def get_extra_cache_key(sticky_key: str) -> str | None:
    if torch.compiler.config.force_disable_caches:
        warn_once(
            "dynamo_pgo force disabled by torch.compiler.config.force_disable_caches"
        )
        return None

    return format_cache_key(sticky_key)


# This solely controls local PGO
def code_state_path(cache_key: str) -> str | None:
    if not torch._dynamo.config.automatic_dynamic_local_pgo:
        log.debug("automatic_dynamic_local_pgo not enabled")
        return None

    from torch._inductor.runtime.runtime_utils import cache_dir

    code_state_key = re.sub(r'[<>:"/\\|?*]', "_", f"code_state_{cache_key}.pkl")
    return os.path.join(cache_dir(), "dynamo", code_state_key)


def should_use_remote_dynamo_pgo_cache() -> bool:
    if torch.compiler.config.force_disable_caches:
        return False

    if (r := torch._dynamo.config.automatic_dynamic_remote_pgo) is not None:
        return r

    if not is_fbcode():
        return False

    if torch._utils_internal.is_fb_unit_test():
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:dynamo_pgo_version"
    )


def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
    from torch._inductor.remote_cache import create_cache

    if not should_use_remote_dynamo_pgo_cache():
        return None

    return create_cache(
        "dynamo-pgo",
        is_fbcode(),
        "FbRemoteDynamoPGOCache",
        "RemoteDynamoPGOCache",
    )


def _collect_dynamic_sources(code_state: CodeState) -> OrderedSet[str]:
    dynamic_sources: OrderedSet[str] = OrderedSet()
    for src, fs in code_state.automatic_dynamic.items():
        dynamic = False
        if isinstance(fs.size, tuple):
            dynamic = auto_dynamic in fs.size  # type: ignore[operator]
        elif fs.scalar == auto_dynamic:
            dynamic = True
        if dynamic:
            dynamic_sources.add(src)
    return dynamic_sources


def _collect_missing_sources(all_sources: OrderedSet[str]) -> OrderedSet[str]:
    from torch._dynamo.variables.builder import is_dynamic_source

    global _KNOWN_DYNAMIC_SOURCES
    missing_sources: OrderedSet[str] = OrderedSet()
    for src in all_sources:
        if src in _KNOWN_DYNAMIC_SOURCES:
            continue
        elif is_dynamic_source(src):
            _KNOWN_DYNAMIC_SOURCES.add(src)
            continue
        missing_sources.add(src)
    return missing_sources


def log_frame_dynamic_whitelist(f_code: types.CodeType) -> None:
    global _KNOWN_DYNAMIC_SOURCES
    code_id = CodeId.make(f_code)
    frame_state = get_code_state()[code_id]
    all_dynamic_sources = _collect_dynamic_sources(frame_state)
    frame_whitelist = ",".join(all_dynamic_sources)
    missing_whitelist = ",".join(_collect_missing_sources(all_dynamic_sources))
    if frame_whitelist:
        with dynamo_timed(name := "pgo.dynamic_whitelist", log_pt2_compile_event=True):
            CompileEventLogger.pt2_compile(
                name,
                recompile_dynamic_whitelist=frame_whitelist,
                missing_dynamic_whitelist=missing_whitelist,
            )


def _log_size_mismatch_recompile() -> None:
    global _LOGGED_DYNAMIC_ALLOWLIST
    if not _LOGGED_DYNAMIC_ALLOWLIST:
        torch._utils_internal.add_mlhub_insight(
            category="dynamic_shapes_analysis",
            insight="Dynamic shape recompilation detected",
            insight_description="PGO detected a recompilation due to dynamic shapes. \
            Please follow the instruction from the action link to reduce \
            recompilation overhead.",
        )
        # add mlhub insight only once per rank
        _LOGGED_DYNAMIC_ALLOWLIST = True


def render_code_state(cs: defaultdict[CodeId, CodeState]) -> str:
    code_state_str = "\n".join(
        f"{k}:\n"
        + "\n".join(
            f"  {src}: {fs.render()}" for src, fs in v.automatic_dynamic.items()
        )
        for k, v in cs.items()
    )
    dynamic_sources: OrderedSet[str] = OrderedSet()
    for state in cs.values():
        dynamic_sources.update(_collect_dynamic_sources(state))
    if dynamic_sources:
        code_state_str += (
            "\n\nPGO detected a recompilation due to dynamic shapes. "
            "To reduce shape recompilations by compiling dynamically to start, "
            f'set environment variable TORCH_COMPILE_DYNAMIC_SOURCES="{",".join(dynamic_sources)}"'
        )
    return code_state_str


@CacheArtifactFactory.register
class PGOCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        meta = write_local_impl(
            self._rewrite_cache_key_for_mega_cache(self.key), self.content
        )
        assert meta is not None

    @override
    @staticmethod
    def type() -> str:
        return "pgo"

    @staticmethod
    def _rewrite_cache_key_for_mega_cache(original_key: str) -> str:
        """
        The PGO cache artifact key for a MAST job contains the job name and the version.
        When we want to use the cache artifact on a different MAST job, we need to
        update the key to use the new MAST job's name and version.
        """
        if not original_key.startswith("mast:"):
            # if original_key is overridden, then dont change it
            return original_key
        if (new_key := get_cache_key()) is not None:
            return new_key
        return original_key


def hit(key: str, ty: str) -> defaultdict[CodeId, CodeState]:
    global _INIT_CODE_STATE
    assert isinstance(_CODE_STATE, defaultdict)
    log.info("get_code_state %s hit %s, %d entries", key, ty, len(_CODE_STATE))
    trace_structured_artifact(
        f"get_{ty}_code_state",
        "string",
        lambda: render_code_state(_CODE_STATE),  # type: ignore[arg-type]
    )
    set_feature_use("pgo", True)
    _INIT_CODE_STATE = copy.deepcopy(_CODE_STATE)
    return _CODE_STATE


def get_local_code_state(cache_key: str) -> defaultdict[CodeId, CodeState] | None:
    global _CODE_STATE
    path = code_state_path(cache_key)
    if path is not None and os.path.exists(path):
        with dynamo_timed(
            name := "pgo.get_local_code_state", log_pt2_compile_event=True
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            # Read lock not necessary as we always write atomically write to
            # the actual location
            with open(path, "rb") as f:
                try:
                    content = f.read()
                    _CODE_STATE = pickle.loads(content)
                    CompileEventLogger.pt2_compile(name, cache_size_bytes=f.tell())
                except Exception:
                    log.warning(
                        "get_code_state failed while reading %s", path, exc_info=True
                    )
                else:
                    CacheArtifactManager.record_artifact(
                        PGOCacheArtifact.type(), cache_key, content
                    )
                    return hit(path, "local")
    return None


def lookup_remote_cache_entry(
    remote_cache: RemoteCache[JsonDataTy],
    cache_key: str,
    event_name: str | None = None,
) -> defaultdict[CodeId, CodeState] | None:
    code_state = None
    try:
        cache_data = remote_cache.get(cache_key)
    except Exception:
        log.warning("get_code_state failed remote read on %s", cache_key, exc_info=True)
    else:
        if cache_data is not None:
            try:
                assert isinstance(cache_data, dict)
                data = cache_data["data"]
                assert isinstance(data, str)
                payload = base64.b64decode(data)
                if event_name is not None:
                    CompileEventLogger.pt2_compile(
                        event_name, cache_size_bytes=len(payload)
                    )
                code_state = pickle.loads(payload)
            except Exception:
                log.warning(
                    "get_code_state failed parsing remote result on %s",
                    cache_key,
                    exc_info=True,
                )
            else:
                CacheArtifactManager.record_artifact(
                    PGOCacheArtifact.type(), cache_key, payload
                )
        else:
            log.info("get_code_state remote miss on %s", cache_key)
    return code_state


def get_remote_code_state(cache_key: str) -> defaultdict[CodeId, CodeState] | None:
    global _CODE_STATE
    remote_cache = get_remote_cache()
    if remote_cache is not None:
        with dynamo_timed(
            name := "pgo.get_remote_code_state",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="pgo_get_remote_code_state_time_us",
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            code_state = lookup_remote_cache_entry(remote_cache, cache_key, name)
            if code_state is not None:
                _CODE_STATE = code_state
                return hit(cache_key, "remote")
    return None


def get_extra_remote_code_state(cache_key: str) -> None:
    """
    Reads an additional PGO profile from the given cache key, and merges it with the default PGO profile.
    """
    global _CODE_STATE
    assert _CODE_STATE is not None

    remote_cache = get_remote_cache()
    if remote_cache is not None:
        with dynamo_timed(
            name := "pgo.get_extra_remote_code_state",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="pgo_get_remote_code_state_time_us",
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            code_state = lookup_remote_cache_entry(remote_cache, cache_key)
            log.info(
                "get_extra_code_state %s hit, %d entries",
                cache_key,
                len(code_state) if code_state is not None else 0,
            )
            if code_state is not None:
                assert not _CODE_STATE
                _CODE_STATE = code_state
                # log to tlparse
                trace_structured_artifact(
                    "get_extra_remote_code_state",
                    "string",
                    lambda: render_code_state(code_state),
                )


def get_code_state() -> defaultdict[CodeId, CodeState]:
    global _CODE_STATE, _INIT_CODE_STATE
    if _CODE_STATE is not None:
        return _CODE_STATE

    # Initialize it (even if we don't look up profile)
    _CODE_STATE = defaultdict(CodeState)

    cache_key = get_cache_key()
    if cache_key is None:
        return _CODE_STATE

    # Attempt local
    local_code_state = get_local_code_state(cache_key)

    # Attempt remote
    if local_code_state is None:
        get_remote_code_state(cache_key)

    # Attempt additional remote if neither local/default remote succeeded
    if (
        not _CODE_STATE
        and (sticky_read := torch.compiler.config.pgo_extra_read_key) is not None
    ):
        extra_read_key = get_extra_cache_key(sticky_read)
        if extra_read_key is not None:
            get_extra_remote_code_state(extra_read_key)

    log.info("get_code_state using default")

    assert _CODE_STATE is not None
    return _CODE_STATE


def put_code_state() -> None:
    if _CODE_STATE is None:
        log.info("put_code_state: never initialized, will not write")
        return

    if _CODE_STATE == _INIT_CODE_STATE:
        log.info("put_code_state: no change, skipping")
        return

    cache_key = get_cache_key()
    if cache_key is None:
        log.info("put_code_state: no cache key, skipping")
        return

    put_local_code_state(cache_key)
    put_remote_code_state(cache_key)
    if (sticky_write := torch.compiler.config.pgo_extra_write_key) is not None:
        extra_write_key = get_extra_cache_key(sticky_write)
        if extra_write_key is not None:
            put_remote_code_state(extra_write_key)


def write_local_impl(cache_key: str, pickled_code: bytes) -> tuple[str, int] | None:
    path = code_state_path(cache_key)

    if path is None:
        return None

    # If the user isn't misusing our API, we should have exclusive access to
    # this directory.  But it's not too hard

    tmp_path = path + ".tmp"
    lock_path = path + ".lock"
    # We /mostly/ don't need the lock but the tmp file could be clobbered
    # TODO: use a safe tempfile create to eliminate lock
    from torch.utils._filelock import FileLock

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        with open(tmp_path, "wb") as f:
            f.write(pickled_code)
            size = f.tell()
        os.replace(tmp_path, path)
    return path, size


def put_local_code_state(cache_key: str) -> None:
    with dynamo_timed(name := "pgo.put_local_code_state", log_pt2_compile_event=True):
        CompileEventLogger.pt2_compile(name, cache_key=cache_key)
        assert _CODE_STATE is not None

        pickled_code = pickle.dumps(_CODE_STATE)

        CacheArtifactManager.record_artifact(
            PGOCacheArtifact.type(), cache_key, pickled_code
        )

        meta = write_local_impl(cache_key, pickled_code)
        if meta is None:
            log.info("put_code_state: local cache disabled")
            return
        path, size = meta

        CompileEventLogger.pt2_compile(name, cache_size_bytes=size)
        log.info("put_code_state: wrote local %s, %d entries", path, len(_CODE_STATE))
        trace_structured_artifact(
            "put_local_code_state",
            "string",
            lambda: render_code_state(_CODE_STATE),
        )


def put_remote_code_state(cache_key: str, extra_code_state: bool = False) -> None:
    event_name = (
        "put_remote_code_state"
        if not extra_code_state
        else "put_extra_remote_code_state"
    )
    with dynamo_timed(
        name := f"pgo.{event_name}",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="pgo_put_remote_code_state_time_us",
    ):
        CompileEventLogger.pt2_compile(name, cache_key=cache_key)
        assert _CODE_STATE is not None

        remote_cache = get_remote_cache()

        if remote_cache is None:
            log.info("%s: remote cache disabled", event_name)
            return

        content = pickle.dumps(_CODE_STATE)
        CompileEventLogger.pt2_compile(name, cache_size_bytes=len(content))
        cache_data: JsonDataTy = {
            "data": base64.b64encode(content).decode("ascii"),
        }
        remote_cache.put(cache_key, cache_data)
        log.info(
            "%s: wrote remote %s, %d entries", event_name, cache_key, len(_CODE_STATE)
        )
        # TODO: don't log this multiple times
        trace_structured_artifact(
            event_name,
            "string",
            lambda: render_code_state(_CODE_STATE),
        )


# NB: this does NOT reset the cached code state on disk
def reset_code_state() -> None:
    global _CODE_STATE, _INIT_CODE_STATE, _LOGGED_DYNAMIC_ALLOWLIST
    _CODE_STATE = None
    _INIT_CODE_STATE = None
    _LOGGED_DYNAMIC_ALLOWLIST = False
