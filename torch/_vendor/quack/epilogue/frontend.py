# Copyright (c) 2026, Han Guo, Tri Dao.
"""FlexAttention-style epilogue authoring: a plain Python function over the
accumulator, lowered onto the EpiOp machinery. The design in one page:

Layer map
---------
* quack.epilogue.ops        — EpiOps: per-tensor RESOURCE lifecycle (smem, TMA,
                              fragments, flushes: begin/begin_loop/end_loop/
                              end), host schema hooks (host_arg_key/
                              host_fake_arg/host_call_arg), and VALUE PORTS
                              (below).
* quack.gemm_runtime.identity — the LEAF: semantic fingerprints, digest
                              registries, picklable class/mod refs.
* quack.gemm_runtime.host   — generic host layer: one jit-cached compile fn
                              (the kernel CLASS is a key argument — classes
                              pickle by module+qualname, so disk keys are
                              stable and async workers import the right module
                              by unpickling), plan cache/build/run driven
                              entirely by the op schema.
* this module               — the fn contract + kernel-class minting. A mod is
                              ONE Python function called per element at trace
                              time; the minted class is a standard
                              ComposableEpiMixin subclass (its traced loops
                              live in quack.epilogue.visit), so hand-written
                              mixins remain first-class and anything the fn
                              form can't express (e.g. symmetric's scheduler)
                              stays a mixin. Value vocabulary (Pair/F2/pexp):
                              quack.epilogue.math.
* quack.epilogue.library    — the library of ready mods and reusable ops.

The fn contract
---------------
``fn(acc, **operands) -> {"D": ..., <outputs/sinks>...}``, called once per
accumulator element (or pair). Values are Float32 scalars on pre-SM100 and
:class:`F2` packed pairs on SM100 (same scalar-or-f32x2 contract as
quack.activation, whose functions compose directly; F2 arithmetic lowers to
packed f32x2 intrinsics). ``"D"`` is optional — omit it to leave the raw
accumulator. Loop shapes mirror the hand-written mixins exactly; no tracer
guesses about vectorization.

* OPERANDS are the fn's parameter names after ``acc``. Kinds are inferred
  from tensor metadata at plan time: (l, n) row broadcast / (l, m) col
  broadcast ((total_m,) rank-1 under varlen) / (l, m, n) full-tile load /
  python scalar or 1-element tensor -> Scalar. ``c`` is reserved for the GEMM
  C operand. ``ops={name: EpiOp(...)}`` pins a name to an explicit op when
  inference is ambiguous (m == n) or the op is custom.
* PAIRING (gated / dgated) is also inferred — aux buffer at half GEMM-N pairs
  the accumulator over adjacent N columns; 16-bit C at twice GEMM-N packs C
  and D two lanes per f32 element — and expressed in the body via
  ``unpack``/``pack`` (see :class:`Pair`, whose arithmetic is lane-wise).
  ``paired=("acc",)`` declares pairing when tensors give no signal (RoPE:
  full-width D, no aux).
* SINKS: ``outputs=(names,)`` declares aux tile stores (each TileStore owns
  its own dtype/rounding, so multiple mixed-dtype outputs compose); ``reduces={name:
  ColVecReduce/RowVecReduce(name, combine="add"|"max"|"max_abs")}`` declares reduce
  outputs (fn returns the per-element value; buffers are per-CTA-tile
  partials). A reduce of a PRODUCT should use ``scaled=True`` and return the
  two factors — ``{"sqsum": (x, x)}`` — so the fold is one fused
  ``fma(val, scale, acc)``: the product is never rounded on its own (bitwise
  parity with folding the product directly, one FFMA instead of FMUL+FADD).
  ``outs={name: sink_op}`` is the general form for any sink op. A sink declares
  ``sink_arity``; arity greater than one means the fn returns a tuple of
  independently collected value planes for the sink to combine.
* PREPASS: ``prepass=fn2, prepass_outs=(names,)`` runs fn2 over the RAW
  accumulator before any store (driver flag epi_needs_acc_prepass; needs a
  re-readable accumulator — SM90 registers, SM100 tmem with no_release;
  pingpong is fine, its per-warpgroup epilogues are strictly exclusive so the
  stats smem is only temporally shared), feeding prepass sink ops; the same
  op then serves the main fn as a value operand carrying the finalized
  statistic (QK-norm: sq-sums -> dense rstd*w multiplier). Any transform the
  statistics must see is explicit duplicate math in fn2 — by design.
* VARLEN: pass ``cu_seqlens_m`` (and ``A_idx`` for gather); operands/outputs
  are (total_m, ...) shaped, colvecs rank-1. No prepass or TileLoad under
  varlen yet.

Value ports (how new ops join the fn dataflow)
----------------------------------------------
An EpiOp declares ``fn_port`` — the frontend dispatches on it alone, never on
the op's type:
* "row" / "col" / "tile" / "scalar": the built-in fragment kinds (the load
  ops declare these; inference picks the same strings from tensor metadata).
* "value": the fn receives op.name as a value; ``fn_prepare`` turns the op's
  begin_loop state into a dense per-loop-index fragment.
* "apply": the fn receives a CALLABLE — ``y = rope(acc)`` — so the op's math
  slots into the fn dataflow at a user-chosen, review-visible point.
* "sink": the fn returns op.name; the frontend collects a dense fragment and
  hands it to ``fn_sink_flush`` once per subtile (fragment-level, so sinks own
  aliasing/coupled-accumulator numerics and per-subtile rescales).
One attribute makes a custom op compose with everything else here.

Speed-of-light rules (bugs otherwise; all were hit once)
--------------------------------------------------------
* Inside cutlass.range bodies, python-static branches must be
  ``const_expr(...)`` (dict comprehensions dodge the AST rewriting).
* vectorize=True demands plain loop-index addressing (no 2*i — use
  flat_divide pair views) and nonzero strides (densify broadcast fragments
  with an unrolled scalar copy first; see the _dense helpers).
* Sinks fold at fragment level: per-element accumulation into the zero-stride
  aliased accumulator slice double-counts on the SM100 packed path.
* Ragged last tile along the reduce dim: OOB accumulator lanes are ZERO —
  the identity for add and max_abs, but not for max or LSE, so max/LSE mask
  OOB lanes to -inf per element by default; check_oob=False compiles the
  predicate out and the host then requires the reduce dim tile-divisible
  (N for colvec/LSE, M for rowvec — varlen_m rejected).

Caching and identity
--------------------
An EpiMod's ``semantic_key`` deep-fingerprints the fn (source plus every
global/closure it references, recursively — factory patterns and helper
edits change it; formatting does not) together with the prepass fn, outputs,
mode, and each op's ``cache_key()`` (type + name + ``config_key()``). The
fingerprint is FAIL-CLOSED: primitives, containers, enums, modules/classes,
functions (incl. wrapped/partial/builtin), dataclasses, and objects
implementing ``__quack_semantic_key__(self) -> object`` are supported;
anything else raises at decoration time — a capture we cannot fingerprint
must never reach the compile cache, because a too-coarse key silently reuses
the wrong kernel. EpiOps implement the protocol as their ``cache_key()``.
Kernel classes are minted per (semantic digest, operand kinds, SM, modes)
but never cross the jit-cache boundary directly: compiles carry a picklable
:class:`~quack.gemm_runtime.identity.GemmClassRef` recipe that re-mints the class at the
point of use — by importing the module-global EpiMod, or, for EpiMods with
no importable anchor (``__main__``, notebooks), via a digest-validated
cloudpickle payload installed into async workers as a side channel that
never touches the cache key. Same digest -> same disk ``.o``, across
processes and workers.

Why this design (and not an epilogue IR)
----------------------------------------
The goals are speed of light AND low marginal cost per new epilogue, and the
two pressures meet at the op boundary:

* The fn is the COMPOSITION site. Ordering (``rope(acc) * alpha`` vs
  ``rope(acc * alpha)``) is explicit user code, reviewable in place — not
  graph topology (EVT trees) or sequencing lists. CuTe-DSL's tracing already
  inlines the fn; an epilogue-level IR would only re-derive what MLIR below
  us optimizes anyway (shared subexpressions are shared SSA values for
  free), while hiding the packed-intrinsic and register-layout control the
  hand-written mixins prove is worth having.
* EpiOps are the EXTENSION site. Whoever adds an op — a new reduction, a
  quantized store, a table load with its own prefetch pipeline — writes the
  resource lifecycle once, and ONE port method (value / apply / sink) makes
  it usable from every fn, composed with every other op, with host plumbing,
  caching, and launch inherited from the schema. The proof cases:
  RotaryCosSinLoad ran verbatim with a 15-line adapter; OnlineLSEReduce
  (a coupled accumulator no combine= flag could express) is one class that
  every mod can name in ``outs=``. Ops written before this frontend existed
  keep working in hand-written mixins unchanged — the fn form is a shortcut
  onto the same machinery, never a second framework.
* The escape hatches are graded, not cliffs: pin one operand (``ops=``),
  add one port method, or drop to a full mixin — each step keeps everything
  else composing.

Everything here is pinned by tests/test_gemm_epilogue.py: bitwise-or-1-ulp
and <=1% perf vs the hand-written mixins for every expressible epilogue.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from typing import NamedTuple, Optional

import torch
import cutlass.cute as cute
from cutlass import Boolean, Float32, const_expr

from torch._vendor.quack.cute_dsl_utils import get_device_capacity, mlir_namedtuple, torch2cute_dtype_map
from torch._vendor.quack.epilogue.ops import (
    ColVecLoad,
    EpiOp,
    RowVecLoad,
    Scalar,
    TileLoad,
    TileStore,
    VecLoad,
    VecReduce,
    is_floating_dtype,
)
from torch._vendor.quack.epilogue.math import F2, F16Lanes, Pair, pack, unpack  # noqa: F401  (re-exports)
from torch._vendor.quack.epilogue.visit import _EpiModMixinBase
from torch._vendor.quack.grouped_reduce import GroupedFeedMainMixin
from torch._vendor.quack.gemm_runtime.host import (
    GemmEpiPlan,
    build_gemm_epi_plan,
    run_gemm_epi_plan,
)
from torch._vendor.quack.gemm_runtime.identity import (
    GemmClassRef,
    TORCH_OP_EPI_MODS,
    function_semantic_key as _function_semantic_key,
    module_locator,
    register_local_epi_mod,
)
from torch._vendor.quack.gemm_config import (
    GemmConfig,
    SplitKMode,
    blockscaled_default_config,
    canonicalize_config_constraints,
    cta_tile_shape_m,
    default_config,
)
from torch._vendor.quack.gemm_tvm_ffi_utils import tensor_key
from torch._vendor.quack.gemm_sm80 import GemmSm80
from torch._vendor.quack.gemm_sm90 import GemmSm90
from torch._vendor.quack.gemm_sm100 import GemmSm100
from torch._vendor.quack.gemm_sm120 import GemmSm120
from torch._vendor.quack.rounding import RoundingMode

canonicalize_config_constraints = torch._dynamo.assume_constant_result(
    canonicalize_config_constraints
)

_SM_BASE = {8: GemmSm80, 9: GemmSm90, 10: GemmSm100, 11: GemmSm100, 12: GemmSm120}

_EPI_MODES = {"element", "acc_pair", "packed_cd_b16x2"}


_KIND_TO_OP = {
    "row": RowVecLoad,
    "col": ColVecLoad,
    "tile": TileLoad,
    "scalar": Scalar,
}


def _infer_kind(name, value, m, n, varlen_m=False):
    if not hasattr(value, "stride"):  # python number
        return "scalar"
    if value.ndim == 0 or value.numel() == 1:
        return "scalar"
    if value.ndim in (2, 3) and tuple(value.shape[-2:]) == (m, n):
        return "tile"
    inner = value.shape[-1]
    if value.ndim <= 2 and inner in (m, n):
        if value.ndim == 1:
            # Rank-1 vectors are the varlen colvec form: (total_m,), offset per
            # segment via cu_seqlens on the device side.
            if not (varlen_m and inner == m):
                raise ValueError(
                    f"operand '{name}': rank-1 vectors are varlen colvecs (total_m,); "
                    f"dense calls pass (l, dim)"
                )
            return "col"
        if m == n:
            raise ValueError(
                f"operand '{name}': m == n makes row/col inference ambiguous; "
                f"pin it via @gemm_epilogue(ops={{'{name}': RowVecLoad(...) or ColVecLoad(...)}})"
            )
        return "row" if inner == n else "col"
    raise ValueError(
        f"cannot infer epilogue operand kind for '{name}' with shape {tuple(value.shape)}"
    )


def _require_shape(name, tensor, expected):
    if tensor is None:
        return
    actual = tuple(tensor.shape)
    expected = tuple(expected)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")


def _tile_shape(batch, m, n, varlen_m):
    return (m, n) if varlen_m or batch is None else (batch, m, n)


def _validate_packed_tensor(name, tensor):
    import torch

    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"{name} must be float16 or bfloat16 in packed_cd_b16x2 mode")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be N-major in packed_cd_b16x2 mode")
    if tensor.storage_offset() % 2 or any(s % 2 for s in tensor.stride()[:-1]):
        raise ValueError(f"{name} storage offset and outer strides must permit a float32 view")


def _mod_gemm_key(A, B, D, C, epi_args, epi_key_overrides, *tail) -> tuple:
    """Plan-cache key for EpiMod.gemm, built from raw call inputs only (no
    validation, no kind inference): full tensor metadata for the GEMM operands
    and every epi_args entry (tensor metadata subsumes the inferred kinds and
    the shape checks; scalars key by presence — their compile mode arrives via
    epi_key_overrides), plus the config tail. A hit is exactly a replay of a
    previously validated call with different data pointers."""
    epi_meta = tuple(
        (name, tensor_key(v) if hasattr(v, "stride") else v is not None)
        for name, v in sorted(epi_args.items())
    )
    overrides = tuple(sorted((epi_key_overrides or {}).items()))
    return (
        tensor_key(A),
        tensor_key(B),
        tensor_key(D),
        tensor_key(C),
        epi_meta,
        overrides,
        *tail,
    )


class _FragmentEpiModMixin(_EpiModMixinBase):
    """Device adapter for whole-fragment TensorSSA EpiMod callbacks."""

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        fn = self._epi_mod_fn
        ops_by_name = {op.name: op for op in self._epi_ops}
        values = {}
        for name, kind in self._epi_mod_operands:
            if const_expr(kind == "scalar"):
                dtype = ops_by_name[name].dtype
                register_dtype = Float32 if dtype is None or is_floating_dtype(dtype) else dtype
                fragment = cute.make_rmem_tensor(tRS_rD.shape, register_dtype)
                fragment.fill(epi_loop_tensors[name])
                values[name] = fragment.load()
                continue
            if const_expr(kind == "value"):
                fragment = ops_by_name[name].fn_prepare(self, epi_loop_tensors[name], False)
                assert fragment is not None
                values[name] = fragment.load()
                continue
            if const_expr(kind == "apply"):
                op = ops_by_name[name]
                pstate = op.fn_prepare(self, epi_loop_tensors[name], False)
                values[name] = op.fn_apply_fragment(self, pstate)
                continue
            assert kind in ("c", "row", "col", "tile"), (
                "TensorSSA callbacks support c/row/col/tile/scalar/value/apply operands"
            )
            fragment = tRS_rC if const_expr(kind == "c") else epi_loop_tensors[name]
            assert fragment is not None
            if const_expr(
                kind == "c" or (kind == "tile" and is_floating_dtype(fragment.element_type))
            ):
                fragment = fragment.to(self.acc_dtype)
            value = fragment.load()
            if const_expr(kind in ("row", "col")):
                value = value.reshape(tRS_rD.shape)
            if const_expr(kind != "c"):
                dtype = ops_by_name[name].dtype
                if const_expr(dtype is Boolean):
                    value = value != cute.full_like(value, 0)
                elif const_expr(dtype is not None and not is_floating_dtype(dtype)):
                    value = value.to(dtype)
            values[name] = value

        result = fn(tRS_rD.load(), **values)
        if const_expr("D" in result):
            tRS_rD.store(result["D"].to(self.acc_dtype))
        outputs = []
        for name in self._epi_mod_outputs:
            value = result[name]
            output = cute.make_rmem_tensor(value.shape, value.element_type)
            output.store(value)
            outputs.append(output)
        sink_tmps = []
        for name in self._epi_mod_sinks:
            planes = []
            for value in self._value_planes(name, ops_by_name[name].sink_arity, result[name]):
                plane = cute.make_rmem_tensor(value.shape, value.element_type)
                plane.store(value)
                planes.append(plane)
            sink_tmps.append(tuple(planes))
        self._flush_sinks(ops_by_name, epi_loop_tensors, tuple(sink_tmps))
        return tuple(outputs)


class EpiMod:
    """A user epilogue function plus the machinery to mint and launch kernels."""

    def __init__(
        self,
        fn,
        outputs=(),
        ops=None,
        reduces=None,
        mode=None,
        paired=(),
        outs=None,
        prepass=None,
        prepass_outs=(),
        extra_ops=(),
        vectorize=None,
        fragmentwise=False,
    ):
        self.fn = fn
        # ``outputs`` entries are names or TileStore instances (per-op config:
        # rounding, predicate, quantize codec); a bare name gets a default
        # TileStore.
        self.output_ops = {}
        out_names = []
        quant_ops = []
        for out in outputs:
            if isinstance(out, TileStore):
                out_names.append(out.name)
                self.output_ops[out.name] = out
                if out.quant is not None:
                    quant_ops.append(out.quant)
            else:
                out_names.append(out)
        self.outputs = tuple(out_names)
        # ``extra_ops``: ops the driver consumes that the fn never sees (e.g.
        # Scalar("sr_seed") feeding stochastic-rounding stores). Their values
        # travel through epi_args under the op name. An output's quantize
        # codec (TileStore(quant=...)) is exactly such an op — lift it here so
        # its SF tensor arg, params, and begin/end lifecycle ride the standard
        # op plumbing.
        self.extra_ops = tuple(extra_ops) + tuple(quant_ops)
        self.ops = dict(ops or {})  # explicit EpiOp pins: {operand_name: EpiOp instance}
        # Sink-port ops by output name; ``reduces`` is kept as sugar for the
        # common VecReduce case, ``outs`` is the general form (any fn_port ==
        # "sink" op: OnlineLSEReduce, future quant stores, ...).
        self.sinks = {**dict(reduces or {}), **dict(outs or {})}
        # ``paired=('acc',)`` remains a compatibility spelling for mode="acc_pair".
        paired = tuple(paired)
        if paired:
            if set(paired) != {"acc"}:
                raise ValueError("paired= only supports ('acc',)")
            if mode not in (None, "acc_pair"):
                raise ValueError("paired=('acc',) conflicts with the requested mode")
            mode = "acc_pair"
        self.mode = "element" if mode is None else mode
        if self.mode not in _EPI_MODES:
            raise ValueError(f"unsupported epilogue mode {self.mode!r}; choose one of {_EPI_MODES}")
        self.fragmentwise = fragmentwise
        if fragmentwise and self.mode != "element":
            raise ValueError("TensorSSA callbacks support element mode only")
        self.paired = ("acc",) if self.mode == "acc_pair" else ()
        # None = vectorize the fn loop where supported (SM100). False = keep
        # the vectorizer off for this epilogue: escape hatch for the DSL
        # vectorizer's crash bugs (fused sincos, arith values reused across
        # pack lanes); free when the epilogue is mainloop-hidden, ~20pp of
        # kernel time when epilogue-exposed (see gemm_epilogue docstring).
        if vectorize not in (None, False):
            raise ValueError("vectorize= accepts None (auto) or False (escape hatch)")
        self.vectorize = vectorize
        self.prepass = prepass
        self.prepass_outs = tuple(prepass_outs)
        if (prepass is None) != (not self.prepass_outs):
            raise ValueError("prepass= and prepass_outs= come together")
        if prepass is not None:
            psig = list(inspect.signature(prepass).parameters)
            if not psig or psig[0] != "acc":
                raise ValueError("prepass fn must take 'acc' first")
            self.prepass_operand_names = tuple(psig[1:])
        else:
            self.prepass_operand_names = ()
        for name, op in self.ops.items():
            if not isinstance(op, EpiOp) or op.name != name:
                raise ValueError(f"op for {name!r} must be an EpiOp named {name!r}")
        for name, op in self.sinks.items():
            if not isinstance(op, EpiOp) or op.fn_port != "sink" or op.name != name:
                raise ValueError(
                    f"sink op for {name!r} must have fn_port == 'sink' and be named {name!r}"
                )
            if not isinstance(op.sink_arity, int) or op.sink_arity < 1:
                raise ValueError(f"sink {name!r}: sink_arity must be a positive integer")
            if op.sink_arity > 1 and self.mode == "acc_pair":
                raise ValueError(
                    f"sink {name!r}: multi-plane sinks are not supported in acc_pair mode "
                    "(a tuple return already carries the two lanes there)"
                )
        for name in self.prepass_outs:
            op = self.ops.get(name)
            if op is None:
                op = self.sinks.get(name)
            if op is not None and op.sink_arity > 1:
                raise ValueError(
                    f"sink {name!r}: multi-plane sinks are not supported in prepass_outs"
                )
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        if not params or params[0] != "acc":
            raise ValueError("epilogue fn must take 'acc' first")
        self.operand_names = tuple(params[1:])
        reserved = {"acc", "D"}
        all_names = set(self.operand_names) | set(self.outputs) | set(self.sinks)
        if all_names & reserved - {"D"}:
            raise ValueError(f"operand/output names may not use reserved names {reserved}")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("duplicate output names")
        for op in self.extra_ops:
            if not isinstance(op, EpiOp) or op.name in all_names:
                raise ValueError(f"extra op {op!r} must be an EpiOp with a fresh name")
        self.semantic_key = (
            _function_semantic_key(fn),
            _function_semantic_key(prepass) if prepass is not None else None,
            self.outputs,
            self.mode,
            self.prepass_outs,
            tuple(op.cache_key() for _, op in sorted(self.ops.items())),
            tuple(op.cache_key() for _, op in sorted(self.sinks.items())),
            tuple(op.cache_key() for _, op in sorted(self.output_ops.items())),
            tuple(op.cache_key() for op in self.extra_ops),
            # Preserve the default disk-cache identity while distinguishing
            # explicit vectorizer choices and whole-fragment callbacks.
            *(() if self.vectorize is None else (("vectorize", self.vectorize),)),
            *(() if not self.fragmentwise else (("fragmentwise", True),)),
        )
        self.semantic_digest = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()
        self._ident = f"{fn.__name__}_{self.semantic_digest[:16]}"
        self._minted = {}
        self._plan_cache = {}
        self._call_cache = {}  # eager __call__ fast path: metadata -> launch recipe
        TORCH_OP_EPI_MODS[self.semantic_digest] = self  # quack::gemm_epi resolution

    def supports_swap_ab(self) -> bool:
        """Whether every orientation-sensitive op supports swap-at-trace."""
        return (
            self.mode == "element"
            and self.prepass is None
            and all(op.supports_swap_ab for op in self.ops.values())
            and all(op.supports_swap_ab for op in self.sinks.values())
        )

    def __getstate__(self):
        # Shipped by value (cloudpickle) to async-compile workers when there
        # is no importable anchor: the minted classes and launch plans are
        # process-local (compiled fns, registered dynamic classes) — the
        # worker re-mints from the recipe.
        state = self.__dict__.copy()
        state["_minted"] = {}
        state["_plan_cache"] = {}
        state["_call_cache"] = {}
        return state

    def _module_locator(self):
        """(module, global_name) if this EpiMod is reachable by import in a
        fresh process (async workers rebuild it that way), else None: defined
        in __main__ (scripts, notebooks) or never bound to a module global."""
        return module_locator(self, self.fn)

    def _class_ref(self, mint_key):
        locator = self._module_locator()
        if locator is None:
            # No importable anchor: the disk key stays semantically correct
            # (the digest is in the ref), resolution goes through the
            # process-local registry, and pool submission ships this EpiMod
            # by value (GemmClassRef.__quack_pool_payload__).
            register_local_epi_mod(self.semantic_digest, self)
            return GemmClassRef(
                "epi_mod_local",
                "",
                "",
                mint_key=mint_key,
                semantic_digest=self.semantic_digest,
            )
        return GemmClassRef(
            "epi_mod",
            *locator,
            mint_key=mint_key,
            semantic_digest=self.semantic_digest,
        )

    def _mint(
        self, kind_sig, sm, paired_acc, packed_c, prepass_sig, rounding, arg_forms, add_to_output
    ):
        key = (kind_sig, sm, paired_acc, packed_c, prepass_sig, rounding, arg_forms, add_to_output)
        cls = self._minted.get(key)
        if cls is not None:
            return cls
        epi_ops = []
        for name, kind in kind_sig:
            if name in self.ops:
                op = self.ops[name]
                if not isinstance(op, EpiOp) or op.name != name:
                    raise ValueError(f"op for {name!r} must be an EpiOp named {name!r}")
                if type(op) in (ColVecLoad, RowVecLoad) and _pinned_visit_kind(op) != kind:
                    # Swap-at-trace flipped this pin into kernel coords; the
                    # visit-kind entry is authoritative so async-worker
                    # re-mints from the mint_key alone reconstruct the same
                    # class.
                    op = _KIND_TO_OP[kind](name)
                epi_ops.append(op)
            elif kind != "c":
                epi_ops.append(_KIND_TO_OP[kind](name))
        for out_name in self.outputs:
            op = self.output_ops.get(out_name)
            if op is None:
                op = TileStore(out_name, gated=paired_acc)
            elif paired_acc and not (
                op.gated or getattr(op, "paired_output_bytes", None) is not None
            ):
                raise ValueError(f"output op {out_name!r} must support acc_pair mode")
            epi_ops.append(op)
        epi_ops.extend(self.sinks.values())
        epi_ops.extend(self.extra_ops)
        # The DSL's TVM-FFI arg-spec converter reads per-field type hints off
        # the NamedTuple, so mint through typing.NamedTuple with the same
        # annotations the hand-written EpilogueArguments use.
        forms_by_name = dict(arg_forms)
        arg_specs = [
            (op.name, op.arg_spec_type(const=forms_by_name.get(op.name) == "Const"))
            for op in epi_ops
        ]
        # Split-K (SERIAL/PARALLEL) per-tile flag + partials workspace; None (and
        # constexpr'd away) for split_k == 1 compiles. Mirrors GemmDefaultEpilogue.
        arg_specs += [
            ("split_k_semaphore", Optional[cute.Tensor]),
            ("split_k_workspace", Optional[cute.Tensor]),
        ]
        Args = NamedTuple("EpilogueArguments", arg_specs)
        Args.__new__.__defaults__ = (None,) * len(arg_specs)
        Args = mlir_namedtuple(Args)
        # Class attribute, not a field: GemmBase reads epilogue_args.add_to_output
        # (duck-typed) to pick the D TMA atom — "add" (cp.reduce.async.bulk,
        # D += delta with no C load) vs "store". A plain bool on the class is
        # trace-time constexpr and invisible to the FFI field walk.
        Args.add_to_output = add_to_output
        cls_name = (
            f"GemmMod_{self._ident}_{'g' if paired_acc else ''}{'p' if packed_c else ''}_"
            f"{'_'.join(k for _, k in kind_sig) or 'none'}"
            f"{''.join(f'_{n}{f}' for n, f in arg_forms)}_sm{sm}"
            # rounding / add_to_output are call-time knobs on an otherwise
            # identical mint: they must distinguish the class name or remints
            # collide.
            f"{'' if rounding == RoundingMode.RN else f'_r{int(rounding)}'}"
            f"{'_ao' if add_to_output else ''}"
        )
        class_semantic_key = (self.semantic_digest, key)
        existing = getattr(sys.modules[__name__], cls_name, None)
        if existing is not None:
            if getattr(existing, "_epi_mod_class_semantic_key", None) != class_semantic_key:
                raise RuntimeError(f"dynamic epilogue class-name collision for {cls_name}")
            self._minted[key] = existing
            return existing
        # The store path (incl. the gated halved-tile machinery) is TileStore
        # config, so every minted class shares one base pair.
        cls = type(
            cls_name,
            (
                GroupedFeedMainMixin,
                _FragmentEpiModMixin if self.fragmentwise else _EpiModMixinBase,
                _SM_BASE[sm],
            ),
            {
                "_epi_ops": tuple(epi_ops),
                "_epi_mod_fn": staticmethod(self.fn),
                "_epi_mod_operands": kind_sig,
                "_epi_mod_outputs": self.outputs,
                "_epi_mod_sinks": tuple(self.sinks),
                "_epi_mod_group_n": 2 if paired_acc else 1,
                "_epi_mod_packed_cd": packed_c,
                "_epi_mod_prepass_fn": staticmethod(self.prepass) if self.prepass else None,
                "_epi_mod_prepass_operands": prepass_sig,
                "_epi_mod_prepass_outs": self.prepass_outs,
                "_epi_mod_rounding": rounding,
                "_epi_mod_vectorize": self.vectorize,
                "_epi_mod_fragmentwise": self.fragmentwise,
                "_extra_param_fields": (
                    ("split_k_semaphore", Optional[cute.Tensor], None),
                    ("split_k_workspace", Optional[cute.Tensor], None),
                ),
                "_epi_mod_class_semantic_key": class_semantic_key,
                "EpilogueArguments": Args,
                "__module__": __name__,
                "__qualname__": cls_name,
            },
        )
        # Registration is useful for inspection and in-process reuse. The JIT
        # cache receives a GemmClassRef, never this process-local class object.
        setattr(sys.modules[__name__], cls_name, cls)
        self._minted[key] = cls
        return cls

    def gemm_tuned(
        self, A, B, D, C=None, *, epi_args: dict, b_kn: bool = False, config_constraints=None
    ):
        """Autotune GEMM over the optionally constrained native config space."""
        from torch._vendor.quack.gemm_runtime.autotune import tuned_mod_gemm

        return tuned_mod_gemm(
            self,
            A,
            B,
            D,
            C,
            epi_args=epi_args,
            b_kn=b_kn,
            config_constraints=config_constraints,
        )

    def gemm(
        self,
        A,
        B,
        D,
        C=None,
        *,
        epi_args: dict,
        tile_M: int = 128,
        tile_N: int = 256,
        cluster_M: int = 2,
        cluster_N: int = 1,
        tile_K: Optional[int] = None,
        pingpong: bool = False,
        persistent: bool = True,
        is_dynamic_persistent: bool = False,
        max_swizzle_size: int = 8,
        tile_count_semaphore=None,
        cu_seqlens_m=None,
        A_idx=None,
        rounding_mode: int = RoundingMode.RN,
        epi_key_overrides=None,  # {op_name: key} when the caller owns the key rule (scalar modes)
        b_kn: bool = False,  # B passed (k, n) / (l, k, n), relabeled at trace time (dense SM90+)
        use_tma_gather: bool = False,
        concat_layout=None,
        SFA=None,  # blockscaled scale factors, see quack.gemm
        SFB=None,
        # BlockScaledFormat names for A / B (independent; required when SFA/SFB
        # are passed) — the descriptors drive validation and the MMA dtypes.
        bs_format_a=None,
        bs_format_b=None,
        swap_ab=False,  # swap-at-trace: requires b_kn (B given (k, n)); dense element mode
        split_k: int = 1,  # K-dim split factor (SERIAL/PARALLEL only; see quack.gemm)
        split_k_mode: int = SplitKMode.SERIAL,
        add_to_output: bool = False,  # D += result via the TMA reduce-add store atom
        #                               (cp.reduce.async.bulk — no C load, half the
        #                               accumulate traffic of C=D). The fn still
        #                               returns the DELTA; requires C=None.
        ag_args=None,  # AllGather+GEMM flags contract (see quack/distributed/)
        # A-operand transform (SM90 RS / SM120 warp-MMA mainloop). Value
        # transforms leave the
        # call contract unchanged. Layout-owning (packed-weight) transforms
        # flip the slots: caller A = ACTIVATIONS (m, k), caller B = the
        # repacked BLOB from fmt.prepare (+ its SF strip via transform_sf);
        # D and tile-shaped epi operands stay caller-oriented (m, n_full) —
        # kernel m is the weight/output-channel dim, so caller row/col vec
        # labels flip exactly like swap_ab.
        transform_a=None,
        transform_sf=None,
        post_init_attrs=(),  # ((attr, value), ...) setattr'd on the gemm object pre-trace
        #                      (constexpr-like knobs, e.g. ("use_pdl", False))
        split_k_buffers=None,  # (sem, ws) from quack.gemm._split_k_buffers: reuse across
        #                        calls when the kernel leaves them reusable (serial
        #                        split-k self-resets); None = fresh per call
        _launch=True,  # False: resolve/compile only (EpiMod.plan) — no kernel launch
    ) -> GemmEpiPlan:
        varlen_m = cu_seqlens_m is not None
        gather_A = A_idx is not None
        blockscaled = SFA is not None
        concat_key = tuple(sorted(concat_layout)) if concat_layout else ()
        owned_fmt = None
        transform_key = None
        arg_sf = None
        n_over = None
        if transform_a is not None:
            from torch._vendor.quack.operand_transform.host import as_transform_mod

            transform_a = as_transform_mod(transform_a)
            owned_fmt = transform_a.owned_fmt
            transform_key = transform_a.plan_key()
            if varlen_m or gather_A or blockscaled or concat_key or ag_args is not None:
                raise ValueError("transform_a: plain dense GEMM only")
            if swap_ab:
                raise ValueError("transform_a + swap_ab: not supported")
            if owned_fmt is not None:
                if pingpong:
                    raise ValueError("layout-owning transform_a does not support pingpong")
                if self.mode != "element" or self.sinks or self.prepass is not None:
                    raise ValueError(
                        "layout-owning transform_a supports element-mode sink-less epilogues only"
                    )
                if C is not None:
                    raise ValueError("layout-owning transform_a + C operand: not supported yet")
            # Slot/bundle/geometry resolution is the handle's — ONE statement
            # of the call contract (TransformModBase.resolve_operands).
            resolved = transform_a.resolve_operands(A, B, transform_sf, tile_M, tile_K)
            A_slot, B_slot, A = resolved.a_slot, resolved.b_slot, resolved.A
            arg_sf, tile_K, n_over = resolved.arg_sf, resolved.tile_k, resolved.n_gemm
        elif transform_sf is not None:
            raise ValueError("transform_sf without a layout-owning transform_a")
        else:
            A_slot, B_slot = (B, A) if swap_ab else (A, B)
        if tile_count_semaphore is not None and not is_dynamic_persistent:
            raise ValueError("tile_count_semaphore requires is_dynamic_persistent=True")
        if add_to_output:
            if C is not None:
                raise ValueError("add_to_output accumulates onto D (no C operand); pass C=None")
            if D is None:
                raise ValueError("add_to_output requires the D output tensor")
            if varlen_m:
                raise ValueError("add_to_output under varlen_m: not supported")
            if D.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                raise ValueError(
                    "add_to_output needs an f32/f16/bf16 D (cp.reduce.async.bulk add), "
                    f"got {D.dtype}"
                )
        if split_k > 1:
            split_k_mode = SplitKMode(split_k_mode)
            if split_k_mode == SplitKMode.SEPARATE:
                raise ValueError(
                    "epilogue-mod GEMM does not support SplitKMode.SEPARATE (the separate "
                    "reduction kernel cannot run epi mods); use SERIAL or PARALLEL"
                )
            if varlen_m or gather_A or swap_ab or ag_args is not None:
                raise ValueError("split_k requires a dense GEMM (no varlen/gather/swap_ab/ag)")
            if rounding_mode != RoundingMode.RN:
                raise ValueError("split_k does not support stochastic rounding")
            if D is None:
                raise ValueError("split_k requires the D output tensor")
        if swap_ab:
            # Swap-at-trace requires dense element mode, B (k, n), and
            # orientation-aware EpiOps. GroupedLocalReduce owns the
            # transposed physical geometry.
            if not b_kn:
                raise ValueError("swap_ab requires b_kn=True (B passed (k, n))")
            if varlen_m or gather_A or blockscaled or concat_key:
                raise ValueError("swap_ab: dense non-blockscaled only")
            if not self.supports_swap_ab():
                raise ValueError("swap_ab requires element-mode and orientation-aware epilogue ops")
            if ag_args is not None:
                # With swapped slots kernel-A is the caller's B: the AG gate
                # would gate the wrong operand (and the wrong M geometry).
                raise ValueError("swap_ab does not support ag_args (AG shards kernel-A along M)")
        # Warm fast path: probe the plan cache on raw-input metadata before any
        # validation or kind inference — a hit is exactly a replay of a
        # previously validated call (the key subsumes everything validation
        # reads), so only the per-call views and the launch remain. The key
        # matches the cold path's record below.
        key = _mod_gemm_key(
            A,
            B,
            D,
            C,
            epi_args,
            epi_key_overrides,
            tile_M,
            tile_N,
            tile_K,
            cluster_M,
            cluster_N,
            pingpong,
            persistent,
            is_dynamic_persistent,
            max_swizzle_size,
            A.device,
            tensor_key(cu_seqlens_m),
            gather_A,
            rounding_mode,
            b_kn,
            use_tma_gather,
            concat_key,
            tuple(post_init_attrs),
            tensor_key(SFA),
            tensor_key(SFB),
            bs_format_a,
            bs_format_b,
            swap_ab,
            split_k,
            int(split_k_mode),
            add_to_output,
            ag_args is not None,
            transform_key,
            # one slot for the transform's sf-side tensor: the W4 SF strip
            # (owned) or the runtime-operand view (value mods with args)
            tensor_key(transform_sf if arg_sf is None else arg_sf),
            # Per-op host arg keys: dtypes/ranks/static widths that the
            # compiled kernel specializes on (host_arg_key's documented
            # role). Without these, a same-shape call with e.g. a different
            # grouped-stats width would alias a stale plan — the tensor form
            # was only ever saved by the launch-time shape check, and a
            # constexpr-form width has no runtime footprint to check.
            tuple(
                sorted(
                    (name, op.host_arg_key(epi_args.get(name)))
                    for name, op in {
                        **self.ops,
                        **self.sinks,
                        **{o.name: o for o in self.extra_ops},
                    }.items()
                )
            ),
        )
        plan = self._plan_cache.get(key)
        if plan is not None:
            if _launch:
                run_gemm_epi_plan(
                    plan,
                    A_slot,
                    B_slot,
                    D,
                    C,
                    epi_args,
                    ag_args=ag_args,
                    tile_count_semaphore=tile_count_semaphore,
                    cu_seqlens_m=cu_seqlens_m,
                    A_idx=A_idx,
                    SFA=SFA,
                    SFB=SFB,
                    split_k_buffers=split_k_buffers,
                )
            return plan
        if blockscaled:
            if varlen_m or gather_A:
                raise ValueError("blockscaled GEMM does not support varlen/gather yet")
            if concat_key:
                raise ValueError("blockscaled GEMM does not support concat_layout")
            if tile_K is not None:
                raise ValueError("blockscaled GEMM derives tile_K from the MMA instruction")
        if varlen_m:
            if not persistent:
                raise ValueError("varlen_m requires persistent=True")
            num_seqs = cu_seqlens_m.shape[0] - 1
            if B.ndim != 3 or B.shape[0] != num_seqs:
                raise ValueError(
                    f"varlen_m B is per-sequence indexed: expected (num_seqs={num_seqs}, "
                    f"...) got {tuple(B.shape)}; broadcast a shared B zero-copy via "
                    "B.unsqueeze(0).expand(num_seqs, -1, -1)"
                )
            if A.ndim != 2 or A.stride(-1) != 1:
                raise ValueError("varlen_m: A is (total_m, k), k-major")
            if D is not None and (D.ndim != 2 or D.stride(-1) != 1):
                raise ValueError("varlen_m: D is (total_m, n), n-major")
            if self.prepass is not None:
                raise ValueError("acc prepass + varlen: not supported yet")
        if gather_A:
            if not varlen_m:
                raise ValueError("gather_A requires varlen")
            if cluster_N != 1:
                raise ValueError("gather_A requires cluster_N=1")
        n_gemm = n_over if n_over is not None else (B.shape[-1] if b_kn else B.shape[-2])
        # Kernel coords under swap-at-trace (and layout-owning transforms):
        # kernel m = caller n, kernel n = caller m. Shape checks on
        # D/C/outputs stay caller-oriented (the tensors cross natively; the
        # trace transposes); operand-kind inference and vec shape checks use
        # kernel coords.
        paired_acc = self.mode == "acc_pair"
        packed_c = self.mode == "packed_cd_b16x2"
        if paired_acc and (n_gemm % 2 or tile_N % 2):
            raise ValueError("acc_pair mode requires even GEMM N and tile_N")
        post_init_attrs = tuple(post_init_attrs)
        packed_form = None
        if packed_c:
            # Callers pass C (preact pairs) and D (dx/dy out) in their natural
            # 16-bit dtype; the tensors cross the boundary RAW and the trace
            # recasts them to the f32 packed view (GemmBase._recast_packed_cd
            # via cd_packed — no per-call torch views). The original dtype
            # travels to the trace via post_init (implicit_dtype) exactly like
            # the hand-written dgated host path.
            if "c" not in self.operand_names:
                raise ValueError("packed_cd_b16x2 mode requires a 'c' fn parameter")
            if C is None or D is None:
                raise ValueError("packed_cd_b16x2 mode requires both C and D")
            if D.dtype != C.dtype or D.shape != C.shape:
                raise ValueError("packed C requires a matching D of the same dtype and shape")
            if C.dtype not in (torch.float16, torch.bfloat16):
                raise TypeError("C must be float16 or bfloat16 in packed_cd_b16x2 mode")
            post_init_attrs = (*post_init_attrs, ("implicit_dtype", torch2cute_dtype_map[C.dtype]))
        n = n_gemm
        if varlen_m:
            # total_m for operand inference (colvec length); A rows differ
            # under gather_A, so prefer an output's leading extent.
            ref_t = D if D is not None else epi_args.get((self.outputs or (None,))[0])
            if ref_t is None:
                raise ValueError("varlen_m needs D or an aux output")
            m = ref_t.shape[0]
        else:
            m = A.shape[-2]
        # Inference/vec-check dims in kernel coords; base_shape (D/C/outputs)
        # stays caller-oriented (swap-at-trace transposes those at trace).
        m_i, n_i = (n_gemm, m) if (swap_ab or owned_fmt is not None) else (m, n_gemm)
        batch = B.shape[0] if B.ndim == 3 else None
        base_shape = _tile_shape(batch, m, n_gemm, varlen_m)
        if packed_c:
            if C.stride(-1) == 1 or varlen_m:
                packed_shape = _tile_shape(batch, m, 2 * n_gemm, varlen_m)
                _require_shape("C", C, packed_shape)
                _require_shape("D", D, packed_shape)
                _validate_packed_tensor("C", C)
                _validate_packed_tensor("D", D)
                packed_form = "n"
            else:
                # AB-swapped callers pass m-major C/D: the 16-bit pairs pack
                # along the contiguous M dim (the trace halves M instead of N;
                # same layout the hand-written dgated host's .mT views built).
                packed_shape = _tile_shape(batch, 2 * m, n_gemm, varlen_m)
                _require_shape("C", C, packed_shape)
                _require_shape("D", D, packed_shape)
                if C.stride(-2) != 1 or D.stride(-2) != 1:
                    raise ValueError("packed m-major C/D must be contiguous along M")
                if C.storage_offset() % 2 or D.storage_offset() % 2:
                    raise ValueError("packed m-major C/D storage offsets must permit a f32 view")
                if any(s % 2 for s in (*C.stride()[-1:], *D.stride()[-1:])) or (
                    batch is not None and any(s % 2 for s in (C.stride(0), D.stride(0)))
                ):
                    raise ValueError("packed m-major C/D outer strides must permit a f32 view")
                packed_form = "m"
        else:
            _require_shape("C", C, base_shape)
            _require_shape("D", D, base_shape)
        for out_name in self.outputs:
            if out_name not in epi_args:
                raise ValueError(f"missing epilogue output buffer '{out_name}'")
            output_op = self.output_ops.get(out_name)
            aux = epi_args[out_name]
            out_n = (
                output_op.output_n(n_gemm)
                if output_op is not None and hasattr(output_op, "output_n")
                else n_gemm // 2
                if paired_acc
                else n_gemm
            )
            if aux.dtype == torch.float4_e2m1fn_x2:
                out_n //= 2  # fp4 values are stored packed, two per byte
            _require_shape(out_name, aux, _tile_shape(batch, m, out_n, varlen_m))
            if paired_acc:
                expected_bytes = getattr(output_op, "paired_output_bytes", (2,))
                if isinstance(expected_bytes, int):
                    expected_bytes = (expected_bytes,)
                if aux.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                    torch.float4_e2m1fn_x2,
                ):
                    expected_bytes = (*expected_bytes, 1)
                if aux.element_size() not in expected_bytes:
                    raise TypeError(
                        "acc_pair auxiliary output has the wrong storage width: "
                        f"expected one of {expected_bytes} bytes, got {aux.element_size()}"
                    )
                if aux.stride(-1) != 1 or (D is not None and D.stride(-1) != 1):
                    raise ValueError("acc_pair auxiliary output and D must be N-major")
        # Swap-at-trace relabels pinned vec pins into KERNEL coordinates: a
        # caller colvec is the swapped kernel's rowvec (and vice versa), so the
        # pin's class flips for this call. Other orientation-sensitive vec pins
        # (varlen subclasses, reduces) have no swapped form and fail loudly.
        flipped = swap_ab or owned_fmt is not None
        pins = {}
        for name, op in self.ops.items():
            if flipped and type(op) in (ColVecLoad, RowVecLoad):
                pins[name] = (RowVecLoad if type(op) is ColVecLoad else ColVecLoad)(name)
            elif flipped and isinstance(op, (VecLoad, VecReduce)):
                raise ValueError(
                    f"swap_ab/transform_a: pinned vec op {name!r} of type {type(op).__name__} "
                    "has no swapped orientation"
                )
            else:
                pins[name] = op
        epi_values = {}
        kind_sig = []
        for name in self.operand_names:
            if name == "c":
                if C is None:
                    raise ValueError("epilogue fn takes 'c' but no C operand was passed")
                kind_sig.append(("c", "c"))
                continue
            if name not in epi_args:
                raise ValueError(f"missing epilogue operand '{name}'")
            kind = (
                "pinned" if name in pins else _infer_kind(name, epi_args[name], m_i, n_i, varlen_m)
            )
            if varlen_m and kind == "tile":
                raise ValueError(f"operand '{name}': TileLoad does not support varlen_m yet")
            visit_kind = _pinned_visit_kind(pins[name]) if kind == "pinned" else kind
            batch_l = B.shape[0] if B.ndim == 3 else 1
            # Pinned ops own their host schema (host_arg_key validates the
            # value); the built-in shape rules only apply to inferred kinds.
            if kind == "pinned":
                validate = getattr(pins[name], "host_validate", None)
                if validate is not None:
                    validate(
                        epi_args[name],
                        m=m_i,
                        n=n_i,
                        tile_M=tile_M,
                        tile_N=tile_N,
                        batch=batch,
                        varlen_m=varlen_m,
                        epi_args=epi_args,
                        swap_ab=swap_ab,
                    )
            elif visit_kind == "row":
                _require_shape(name, epi_args[name], (batch_l, n_i))
            elif visit_kind == "col":
                expected = (m_i,) if varlen_m else (batch_l, m_i)
                _require_shape(name, epi_args[name], expected)
            elif visit_kind == "tile":
                _require_shape(name, epi_args[name], base_shape)
            kind_sig.append((name, kind if kind != "pinned" else pins[name].__class__.__name__))
            epi_values[name] = epi_args[name]
        for out_name in self.outputs:
            epi_values[out_name] = epi_args[out_name]
        # Sink partials are per-CTA-tile along the reduce dim: under the SM100
        # 2-CTA MMA (even cluster_M) each CTA of the pair owns half the M tile,
        # so M-tile-unit buffers count in cta_tile_M, not tile_M.
        cta_tile_M = cta_tile_shape_m(
            tile_M, cluster_M, get_device_capacity(A.device)[0], blockscaled
        )
        for sink_name in self.sinks:
            if sink_name not in epi_args:
                raise ValueError(f"missing sink output buffer '{sink_name}'")
            op = self.sinks[sink_name]
            validate = getattr(op, "host_validate", None)
            if validate is not None:
                validate(
                    epi_args[sink_name],
                    m=m,
                    n=n_gemm,
                    tile_M=tile_M,
                    tile_N=tile_N,
                    batch=batch,
                    varlen_m=varlen_m,
                    epi_args=epi_args,
                    swap_ab=swap_ab,
                )
            if varlen_m and getattr(op, "dim", 0) == 1 and getattr(op, "combine", "add") != "add":
                # Zero-filled OOB rows (the varlen ragged-load contract) are
                # the identity for add only; a segment-max finalize would also
                # break graph-safety.
                raise ValueError(
                    f"sink '{sink_name}': combine={op.combine!r} M-fold reduces are not "
                    "supported under varlen_m"
                )
            alloc = getattr(op, "sink_alloc_shape", None)
            if alloc is not None:
                lead_k = (m,) if varlen_m or batch is None else (batch, m)
                num_seqs_k = (
                    cu_seqlens_m.shape[0] - 1 if varlen_m and getattr(op, "dim", 0) == 1 else None
                )
                _require_shape(
                    sink_name,
                    epi_args[sink_name],
                    alloc(lead_k, n_gemm, cta_tile_M, tile_N, num_seqs=num_seqs_k),
                )
            if getattr(op, "check_oob", True) is False:
                # The reduce dim must be tile-divisible: dim 0 (colvec) reduces
                # along N, dim 1 (rowvec) along M (varlen_m boundaries are
                # always potentially ragged).
                if getattr(op, "dim", 0) == 0:
                    if n_gemm % tile_N:
                        raise ValueError(
                            f"sink '{sink_name}': check_oob=False requires N divisible by "
                            f"tile_N (N={n_gemm}, tile_N={tile_N})"
                        )
                elif varlen_m or m % cta_tile_M:
                    raise ValueError(
                        f"sink '{sink_name}': check_oob=False requires M divisible by the "
                        f"per-CTA tile and no varlen_m (M={m}, cta_tile_M={cta_tile_M})"
                    )
            epi_values[sink_name] = epi_args[sink_name]
        for op in self.extra_ops:
            if op.name in epi_args:
                validate = getattr(op, "host_validate", None)
                if validate is not None:
                    validate(
                        epi_args[op.name],
                        m=m,
                        n=n_gemm,
                        tile_M=tile_M,
                        tile_N=tile_N,
                        batch=batch,
                        varlen_m=varlen_m,
                        epi_args=epi_args,
                        swap_ab=swap_ab,
                    )
                epi_values[op.name] = epi_args[op.name]
        kind_sig = tuple(kind_sig)

        device_capacity = get_device_capacity(A.device)
        if is_dynamic_persistent and device_capacity[0] == 9 and tile_count_semaphore is None:
            raise ValueError("SM90 dynamic persistent scheduling requires tile_count_semaphore")
        if paired_acc and self.outputs and device_capacity[0] == 9 and tile_N % 32:
            raise ValueError("SM90 acc_pair auxiliary output requires tile_N divisible by 32")
        sf_dtype = sf_vec_size = None
        a_mma_dtype = b_mma_dtype = None
        if blockscaled:
            from torch._vendor.quack.gemm_tvm_ffi_utils import (
                resolve_blockscaled_formats,
                validate_blockscaled_sf,
            )

            fmt_a, fmt_b = resolve_blockscaled_formats(bs_format_a, bs_format_b)
            a_mma_dtype = fmt_a.to_cutlass_dtype()
            b_mma_dtype = fmt_b.to_cutlass_dtype()
            sf_dtype, sf_vec_size = validate_blockscaled_sf(
                A, B, SFA, SFB, device_capacity, b_kn=b_kn, fmt_a=fmt_a, fmt_b=fmt_b
            )
        # Re-map pinned ops' kind for the device loop: explicit pins still
        # need a fragment kind; VecLoads present as their dim.
        visit_sig = tuple(
            (name, _pinned_visit_kind(pins[name]) if name in pins else kind)
            for name, kind in kind_sig
        )
        prepass_sig = ()
        if self.prepass is not None:
            if packed_c:
                raise ValueError("prepass + packed C: not supported")
            unknown = set(self.prepass_operand_names) - {n for n, _ in visit_sig}
            if unknown:
                raise ValueError(f"prepass fn reads undeclared operands {unknown}")
            for out_name in self.prepass_outs:
                if out_name not in {n for n, _ in visit_sig} | set(self.sinks):
                    raise ValueError(f"prepass out '{out_name}' must be a declared op")
            prepass_sig = tuple((n, k) for n, k in visit_sig if n in self.prepass_operand_names)
        # Arg FORMS that change the compiled signature (e.g. a constexpr
        # width vs a tensor descriptor) distinguish the mint; kinds stay pure
        # (they drive device-loop dispatch).
        arg_forms = tuple(
            (name, pins[name].host_arg_form(epi_args[name]))
            for name, _ in kind_sig
            if name in pins and pins[name].host_arg_form(epi_args[name])
        )
        mint_key = (
            visit_sig,
            device_capacity[0],
            paired_acc,
            packed_c,
            prepass_sig,
            rounding_mode,
            arg_forms,
            add_to_output,
        )
        GemmCls = self._mint(*mint_key)
        plan = build_gemm_epi_plan(
            GemmCls,
            device_capacity,
            A_slot,
            B_slot,
            D,
            C,
            epi_values=epi_values,
            epi_key_overrides=epi_key_overrides,
            tile_M=tile_M,
            tile_N=tile_N,
            cluster_M=cluster_M,
            cluster_N=cluster_N,
            tile_K=tile_K,
            pingpong=pingpong,
            persistent=persistent,
            is_dynamic_persistent=is_dynamic_persistent,
            max_swizzle_size=max_swizzle_size,
            varlen_m=varlen_m,
            gather_A=gather_A,
            # slot-A relabels via a_transposed (swap_ab); owned transforms
            # pass B (= caller A activations) natively (n, k)
            b_kn=b_kn and not swap_ab and owned_fmt is None,
            swap_ab=swap_ab,
            transform_a=transform_a,
            use_tma_gather=use_tma_gather,
            concat_layout=concat_key,
            sf_dtype=sf_dtype,
            sf_vec_size=sf_vec_size,
            sf_batched=SFA.ndim == 6 if blockscaled else True,
            a_mma_dtype=a_mma_dtype,
            b_mma_dtype=b_mma_dtype,
            post_init_attrs=post_init_attrs,
            gemm_cls_ref=self._class_ref(mint_key),
            packed_cd=packed_form,
            has_ag=ag_args is not None,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
        self._plan_cache[key] = plan
        if _launch:
            run_gemm_epi_plan(
                plan,
                A_slot,
                B_slot,
                D,
                C,
                epi_values,
                ag_args=ag_args,
                tile_count_semaphore=tile_count_semaphore,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                split_k_buffers=split_k_buffers,
            )
        return plan

    # ── Torch-facing interface (Tier 4 surface) ──────────────────────────
    # One interface parameterized by the epilogue object (see HANDOFF Tier 4):
    # eager ``__call__`` (autotunes via quack.gemm_runtime.autotune, allocates missing
    # outputs), ``plan()`` -> EpiPlan (resolve once; ``run()`` makes zero host
    # decisions). B is (k, n) logical at this surface — the torch convention —
    # with the physical layout free.

    def _lead_shape(self, A, cu_seqlens_m, A_idx):
        if cu_seqlens_m is not None:
            return ((A_idx.shape[0] if A_idx is not None else A.shape[0]),)
        return tuple(A.shape[:-1])

    def _default_config(self, A, B, transform_a):
        """Per-arch default config; the transform handle may override (the
        measured W4 coverage rule). split_k is not threaded through the
        eager/plan surface yet — pass it explicitly via mod.gemm for the
        starved-grid decode-shape win."""
        if transform_a is not None:
            cfg = transform_a.default_config(A, B)
            if cfg is not None:
                return cfg
        return default_config(A.device)

    def _alloc_outputs(
        self, out, A, B, C, store_d, out_dtype, cu_seqlens_m, A_idx, n_override=None
    ):
        """Fill in the outputs the caller left out; out= buffers win."""
        import torch

        out = dict(out) if out else {}
        n = B.shape[-1] if n_override is None else n_override
        lead = self._lead_shape(A, cu_seqlens_m, A_idx)
        dt = out_dtype if out_dtype is not None else A.dtype
        if store_d and out.get("D") is None:
            if self.mode == "packed_cd_b16x2":
                if C is None:
                    raise ValueError("packed_cd_b16x2 requires C; D matches its shape/dtype")
                out["D"] = torch.empty_like(C)
            else:
                out["D"] = torch.empty((*lead, n), dtype=dt, device=A.device)
        for name in self.outputs:
            if out.get(name) is None:
                output_op = self.output_ops.get(name)
                n_store = (
                    output_op.output_n(n)
                    if output_op is not None and hasattr(output_op, "output_n")
                    else n // 2
                    if self.mode == "acc_pair"
                    else n
                )
                out[name] = torch.empty((*lead, n_store), dtype=dt, device=A.device)
        return out

    def _alloc_sinks(self, epi_args, lead, n, config, device, blockscaled=False, num_seqs=None):
        """Reduce partials are config-shaped scratch, not outputs: allocated
        per call (stream-safe, graph-pool friendly). A caller-provided buffer
        (same name in operands) is used as-is and returned raw. ``num_seqs``
        (varlen_m only): dim==1 partials are per-sequence tile-prefix rows."""
        import torch

        bufs = {}
        for name, op in self.sinks.items():
            if epi_args.get(name) is not None:
                continue
            # NOTE: for full-vector sinks (ColVecSelect), rows the kernel
            # never writes (out-of-range select indices) stay uninitialized —
            # callers who care pass the buffer.
            cta_tile_m = cta_tile_shape_m(
                config.tile_m, config.cluster_m, config.device_capacity, blockscaled
            )
            shape = op.sink_alloc_shape(
                lead,
                n,
                cta_tile_m,
                config.tile_n,
                num_seqs=num_seqs if getattr(op, "dim", 0) == 1 else None,
            )
            bufs[name] = epi_args[name] = torch.empty(
                shape, dtype=op.sink_alloc_dtype(), device=device
            )
        return bufs

    def _finalize_sink(self, name, buf, cu_seqlens_m, plan):
        """Fold a sink's partial buffer to the user-visible value. Under
        varlen_m, M-fold (dim==1) sinks use per-sequence segment finalization
        (rows grouped by the cu_tiles_m prefix, -> (num_seqs, n)); everything
        else keeps the op's plain host_finalize."""
        op = self.sinks[name]
        tile_m = getattr(plan, "cu_tiles_tile_m", None)
        if cu_seqlens_m is not None and tile_m is not None:
            finalize_varlen = getattr(op, "host_finalize_varlen", None)
            if finalize_varlen is not None:
                return finalize_varlen(buf, cu_seqlens_m, tile_m)
        finalize = getattr(op, "host_finalize", None)
        return finalize(buf) if finalize is not None else buf

    def _iface_execute(self, config, dynamic_scheduler, ctx, _launch=True):
        import torch

        A, C, D = ctx["A"], ctx.get("C"), ctx.get("D")
        dyn = dynamic_scheduler or config.is_dynamic_persistent
        epi_args = dict(ctx["operands"])
        for name in self.outputs:
            epi_args[name] = ctx["out"][name]
        blockscaled = ctx.get("SFA") is not None
        cu_seqlens_m_ctx = ctx.get("cu_seqlens_m")
        sink_bufs = self._alloc_sinks(
            epi_args,
            ctx["lead"],
            ctx["n"],
            config,
            A.device,
            blockscaled,
            num_seqs=None if cu_seqlens_m_ctx is None else cu_seqlens_m_ctx.shape[0] - 1,
        )
        semaphore = (
            torch.zeros(1, dtype=torch.int32, device=A.device)
            if dyn and get_device_capacity(A.device)[0] == 9
            else None
        )
        A_bundle = ctx.get("A_bundle")  # runtime-operand transform: A crosses bundled
        plan = self.gemm(
            A if A_bundle is None else A_bundle,
            ctx["B_d"],
            D,
            C,
            epi_args=epi_args,
            tile_M=config.tile_m,
            tile_N=config.tile_n,
            tile_K=None if blockscaled else config.tile_k,
            cluster_M=config.cluster_m,
            cluster_N=config.cluster_n,
            pingpong=config.pingpong,
            persistent=True,
            is_dynamic_persistent=dyn,
            max_swizzle_size=config.max_swizzle_size,
            tile_count_semaphore=semaphore,
            cu_seqlens_m=ctx.get("cu_seqlens_m"),
            A_idx=ctx.get("A_idx"),
            SFA=ctx.get("SFA"),
            SFB=ctx.get("SFB"),
            bs_format_a=ctx.get("bs_format_a"),
            bs_format_b=ctx.get("bs_format_b"),
            rounding_mode=ctx["rounding_mode"],
            epi_key_overrides=ctx["epi_key_overrides"],
            b_kn=ctx["b_kn"],
            swap_ab=config.swap_ab,
            use_tma_gather=config.use_tma_gather,
            concat_layout=ctx.get("concat_layout"),
            transform_a=ctx.get("transform_a"),
            transform_sf=ctx.get("transform_sf"),
            add_to_output=ctx.get("add_to_output", False),
            _launch=_launch,
        )
        return config, dyn, plan, sink_bufs

    def __call__(
        self,
        A,
        B,  # (k, n) or (l, k, n) — torch convention; physical layout free
        C=None,
        *,
        out=None,  # {name: buffer} incl. "D"; missing entries are allocated
        out_dtype=None,
        store_d=True,
        config_constraints=None,  # partial GemmConfig fields, mapping or canonical tuple
        config=None,  # exact GemmConfig pin; None selects/defaults from constraints
        tuned=True,
        dynamic_scheduler=False,
        cu_seqlens_m=None,
        A_idx=None,
        SFA=None,
        SFB=None,
        bs_format_a=None,  # BlockScaledFormat names (see EpiMod.gemm)
        bs_format_b=None,
        rounding_mode=RoundingMode.RN,
        epi_key_overrides=None,
        concat_layout=None,  # tensors whose non-contiguous dim is concat [gate; up]
        transform_a=None,  # A-operand transform (see EpiMod.gemm); with a
        transform_sf=None,  # layout-owning transform, B is the repacked blob
        transform_operands=None,  # {name: tensor} for a runtime-operand transform_a
        #                           (raw values; the bundle is built here from
        #                           the resolved config's tile_M/tile_K)
        add_to_output=False,  # D += result via the TMA reduce-add store atom (see
        #                       EpiMod.gemm); requires out={"D": buf} and C=None
        compile_dispatch: bool = True,
        **operands,  # epilogue operand tensors/scalars by fn-parameter name
    ):
        """Eager torch-facing call: resolve config (autotune via
        quack.gemm_runtime.autotune on first sight of a metadata class), allocate
        missing outputs, launch. Returns {"D": ..., <declared outputs>,
        <finalized reduces>}; reduce sinks come back finalized via their op's
        ``host_finalize`` (partials are internal scratch) unless the caller
        passed the partial buffer as an operand.

        ``config_constraints`` limits native candidates for both tuning and
        deterministic selection; an exact ``config=`` must satisfy them. The
        tuned path covers the non-SR surface (see quack.gemm_runtime.autotune,
        incl. varlen/gather/blockscaled/concat, A-operand transforms and
        dynamic_scheduler=True); other calls resolve with the explicit
        ``config=`` or the per-arch default.

        Under torch.compile the call records the single ``quack::gemm_epi``
        custom op (see quack.gemm_runtime.torch_op); with reduce sinks the config is
        pinned there (partials must be graph-allocated at exact shapes).
        transform_a rides the same op: the handle crosses by semantic digest
        (bind it to a module global for cross-process graphs), runtime
        operand tensors ride the op's input list, and the bundle is rebuilt
        inside the op body. Generated runtime callers set ``compile_dispatch=False``
        because they are already below torch.compile."""
        import torch

        constraints = canonicalize_config_constraints(config_constraints)
        if config is not None and type(config) is not GemmConfig:
            raise TypeError("config must be an exact GemmConfig or None")
        if config is not None and any(
            getattr(config, name) != value for name, value in constraints
        ):
            raise ValueError(
                f"config {config} does not satisfy config_constraints={dict(constraints)!r}"
            )

        owned_fmt = None
        transform_key = None
        if transform_a is not None:
            if not hasattr(transform_a, "semantic_digest"):
                if compile_dispatch and torch.compiler.is_compiling():
                    raise NotImplementedError(
                        "transform_a under torch.compile needs a digest-carrying handle "
                        "(@a_transform / dropout_a / w4_transform)"
                    )
                from torch._vendor.quack.operand_transform.host import as_transform_mod

                transform_a = as_transform_mod(transform_a)
            owned_fmt = transform_a.owned_fmt
            transform_key = transform_a.plan_key()
        # Runtime-operand transforms: the torch-facing surface takes the RAW
        # operand tensors and builds the TransformAOperand bundle itself from
        # the resolved config's tiles (bundles — whose views bake tile_M/
        # tile_K — stay a mod.gemm/direct-layer contract).
        needs_transform_operands = transform_a is not None and transform_a.needs_operands
        if needs_transform_operands:
            from torch._vendor.quack.operand_transform.transform import TransformAOperand

            if isinstance(A, TransformAOperand):
                raise ValueError(
                    "__call__ builds the operand bundle itself from the resolved config: "
                    "pass the plain (M, K) A plus transform_operands={name: tensor} "
                    "(pre-built bundles go to mod.gemm)"
                )
            if not transform_operands:
                raise ValueError(
                    f"transform_a {getattr(transform_a, 'name', transform_a)!r} declares "
                    "runtime operands: pass transform_operands={name: tensor}"
                )
        elif transform_operands:
            raise ValueError("transform_operands requires a transform_a with declared args")
        if add_to_output:
            # The accumulator must be caller-provided: an auto-allocated D
            # would reduce-add onto garbage.
            if C is not None:
                raise ValueError("add_to_output accumulates onto D (no C operand); pass C=None")
            if not store_d or out is None or out.get("D") is None:
                raise ValueError('add_to_output requires the accumulator as out={"D": buf}')

        if compile_dispatch and torch.compiler.is_compiling():
            if dynamic_scheduler or epi_key_overrides is not None:
                raise NotImplementedError(
                    "dynamic_scheduler/epi_key_overrides under torch.compile: not supported yet"
                )
            from torch._vendor.quack.gemm_runtime.torch_op import compile_call

            return compile_call(
                self,
                A,
                B,
                C,
                out=out,
                out_dtype=out_dtype,
                store_d=store_d,
                config_constraints=constraints,
                config=config,
                tuned=tuned,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                rounding_mode=rounding_mode,
                operands=operands,
                transform_a=transform_a,
                transform_sf=transform_sf,
                transform_operands=transform_operands,
                concat_layout=concat_layout,
                add_to_output=add_to_output,
            )

        # ── Eager warm fast path: one metadata key -> captured launch recipe.
        # Everything metadata-derived (config resolution incl. the autotuned
        # winner, output recipes, sink shapes, slot order) was recorded by a
        # previous identical-metadata call; only allocation + launch remain.
        # packed_cd rides it too (the f32 recast is trace-level, cd_packed);
        # concat and transforms are excluded (their per-call views live in
        # mod.gemm, whose own plan cache is the warm path).
        ck = None
        if concat_layout is None and transform_a is None:
            ck = (
                tensor_key(A),
                tensor_key(B),
                tensor_key(C),
                tuple(
                    sorted(
                        (kk, tensor_key(v) if hasattr(v, "stride") else v is not None)
                        for kk, v in operands.items()
                    )
                ),
                None
                if out is None
                else tuple(sorted((kk, tensor_key(v)) for kk, v in out.items())),
                out_dtype,
                store_d,
                # Pinned configs key by identity: they are module-level
                # constants in practice; a recreated equal config just
                # re-records (correct, one extra cold pass).
                None if config is None else id(config),
                constraints,
                tuned,
                dynamic_scheduler,
                tensor_key(cu_seqlens_m),
                tensor_key(A_idx),
                tensor_key(SFA),
                tensor_key(SFB),
                bs_format_a,
                bs_format_b,
                rounding_mode,
                None if epi_key_overrides is None else tuple(sorted(epi_key_overrides.items())),
                transform_key,
                tensor_key(transform_sf),
                add_to_output,
            )
            entry = self._call_cache.get(ck)
            if entry is not None:
                import torch as _t

                plan, recipes, sink_shapes, b_kn_c, swapped, sem_dyn = entry
                outs = dict(out) if out else {}
                for name, shape, dt in recipes:
                    if outs.get(name) is None:
                        outs[name] = _t.empty(shape, dtype=dt, device=A.device)
                epi_values = dict(operands)
                for name in self.outputs:
                    epi_values[name] = outs[name]
                sink_bufs = {}
                for name, shape, dtype in sink_shapes:
                    if epi_values.get(name) is None:
                        buf = _t.empty(shape, dtype=dtype, device=A.device)
                        sink_bufs[name] = epi_values[name] = buf
                B_w = B if b_kn_c else B.mT
                sem = _t.zeros(1, dtype=_t.int32, device=A.device) if sem_dyn else None
                run_gemm_epi_plan(
                    plan,
                    B_w if swapped else A,
                    A if swapped else B_w,
                    outs.get("D") if store_d else None,
                    C,
                    epi_values,
                    tile_count_semaphore=sem,
                    cu_seqlens_m=cu_seqlens_m,
                    A_idx=A_idx,
                    SFA=SFA,
                    SFB=SFB,
                )
                result = dict(outs) if store_d else {kk: v for kk, v in outs.items() if kk != "D"}
                for name, buf in sink_bufs.items():
                    result[name] = self._finalize_sink(name, buf, cu_seqlens_m, plan)
                return result

        varlen_m = cu_seqlens_m is not None
        # concat reads B (k, n) through per-call views, so it vetoes the b_kn
        # trace-time relabel (the interleave lives in mod.gemm).
        b_kn = get_device_capacity(A.device)[0] >= 9 and not concat_layout
        B_d = B if (b_kn or owned_fmt is not None) else B.mT
        n_override = transform_a.padded_n(B) if transform_a is not None else None
        provided_out = frozenset(k for k, v in (out or {}).items() if v is not None)
        out = self._alloc_outputs(
            out, A, B, C, store_d, out_dtype, cu_seqlens_m, A_idx, n_override=n_override
        )
        D = out.get("D") if store_d else None
        lead = self._lead_shape(A, cu_seqlens_m, A_idx)
        n = B.shape[-1] if n_override is None else n_override
        use_tuner = (
            tuned
            and config is None
            and rounding_mode == RoundingMode.RN
            and epi_key_overrides is None
            and not add_to_output  # the reduce-add D mint is not swept
        )
        if use_tuner:
            from torch._vendor.quack.gemm_runtime.autotune import sink_arg_shapes, tuned_mod_gemm

            epi_args = dict(operands)
            for name in self.outputs:
                epi_args[name] = out[name]
            owned_sinks = {}
            if self.sinks:
                # sink_arg_shapes walks the full config space — cache the
                # worst-case shapes per metadata (it's on the warm path).
                l = lead[0] if len(lead) == 2 else None
                num_seqs = None if cu_seqlens_m is None else cu_seqlens_m.shape[0] - 1
                shape_key = (lead[-1], n, l, str(A.device), num_seqs, constraints)
                cache = self.__dict__.setdefault("_sink_shape_cache", {})
                shapes = cache.get(shape_key)
                if shapes is None:
                    shapes = sink_arg_shapes(
                        self,
                        lead[-1],
                        n,
                        l=l,
                        device=A.device,
                        num_seqs=num_seqs,
                        config_constraints=constraints,
                    )
                    cache[shape_key] = shapes
                for name, shape in shapes.items():
                    if epi_args.get(name) is None:
                        epi_args[name] = torch.empty(
                            shape,
                            dtype=self.sinks[name].sink_alloc_dtype(),
                            device=A.device,
                        )
                        owned_sinks[name] = True
            res = tuned_mod_gemm(
                self,
                A,
                B_d,
                D,
                C,
                epi_args=epi_args,
                b_kn=b_kn,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                dynamic_scheduler=dynamic_scheduler,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                concat_layout=concat_layout,
                # raw operand tensors: the sweep builds per-config bundles
                transform_a=transform_a,
                transform_sf=transform_sf,
                transform_operands=transform_operands,
                config_constraints=constraints,
            )
            sink_bufs = {name: res.sinks[name] for name in owned_sinks}
            cfg_used, plan_used = res.config, res.plan
        else:
            ctx = dict(
                A=A,
                B_d=B_d,
                C=C,
                D=D,
                out=out,
                operands=dict(operands),
                n=n,
                lead=lead,
                b_kn=b_kn,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                rounding_mode=rounding_mode,
                epi_key_overrides=epi_key_overrides,
                concat_layout=concat_layout,
                transform_a=transform_a,
                transform_sf=transform_sf,
                add_to_output=add_to_output,
            )
            if config is not None:
                cfg = config
            elif constraints or any(
                hasattr(op, "supports_config")
                for op in (
                    *self.ops.values(),
                    *self.sinks.values(),
                    *self.output_ops.values(),
                    *self.extra_ops,
                )
            ):
                from torch._vendor.quack.gemm_runtime.autotune import _select_mod_config, mod_selection_args

                selection_args = mod_selection_args(
                    operands,
                    {name: out[name] for name in self.outputs},
                    A=A,
                    B=B_d,
                    b_kn=b_kn,
                    cu_seqlens_m=cu_seqlens_m,
                    A_idx=A_idx,
                    SFA=SFA,
                    concat_layout=concat_layout,
                )
                preferred_config = (
                    blockscaled_default_config(
                        A.shape[-2], n, device_capacity=get_device_capacity(A.device)[0]
                    )
                    if SFA is not None
                    else self._default_config(A, B, transform_a)
                )
                cfg = _select_mod_config(
                    self,
                    A.device,
                    constraints,
                    selection_args,
                    preferred_config=preferred_config,
                )
            elif SFA is not None:
                cfg = blockscaled_default_config(
                    A.shape[-2], n, device_capacity=get_device_capacity(A.device)[0]
                )
            else:
                cfg = self._default_config(A, B, transform_a)
            if needs_transform_operands:
                # tile_K None resolves inside bundle() to the 16-bit kernel
                # default, mirrored by fake_bundle at trace.
                ctx["A_bundle"] = transform_a.bundle(A, transform_operands, cfg.tile_m, cfg.tile_k)
            _, _, plan_used, sink_bufs = self._iface_execute(cfg, dynamic_scheduler, ctx)
            cfg_used = cfg
        if ck is not None:
            # Record the launch recipe for identical-metadata replays. Sink
            # shapes come from the winning config's buffers (tuned sweeps
            # allocate worst-case; the recipe stores the exact live shape).
            recipes = tuple(
                (name, tuple(t.shape), t.dtype)
                for name, t in out.items()
                if name not in provided_out
            )
            # tuned sink_bufs are already the winning config's exact slices
            sink_shapes = tuple(
                (name, tuple(buf.shape), buf.dtype) for name, buf in sink_bufs.items()
            )
            self._call_cache[ck] = (
                plan_used,
                recipes,
                sink_shapes,
                b_kn,
                cfg_used.swap_ab,
                (dynamic_scheduler or cfg_used.is_dynamic_persistent)
                and get_device_capacity(A.device)[0] == 9,
            )
        result = dict(out) if store_d else {k: v for k, v in out.items() if k != "D"}
        for name, buf in sink_bufs.items():
            result[name] = self._finalize_sink(name, buf, cu_seqlens_m, plan_used)
        return result

    def plan(
        self,
        A,
        B,  # (k, n) logical, as in __call__
        C=None,
        *,
        out,  # REQUIRED: buffers for D + every declared output (their
        #        metadata selects the kernel); no "D" entry = no D store
        config=None,  # explicit GemmConfig; None = per-arch default. No
        #               autotuning here (autotune via an eager call, or pass
        #               the tuned config) — plan() never launches.
        dynamic_scheduler=False,
        cu_seqlens_m=None,
        A_idx=None,
        SFA=None,
        SFB=None,
        bs_format_a=None,  # BlockScaledFormat names (see EpiMod.gemm)
        bs_format_b=None,
        rounding_mode=RoundingMode.RN,
        epi_key_overrides=None,
        transform_a=None,  # A-operand transform (see EpiMod.gemm); with a
        transform_sf=None,  # layout-owning transform, B is the repacked blob
        transform_operands=None,  # {name: tensor}, as in __call__ (bundle built here)
        add_to_output=False,  # D += result via the TMA reduce-add store atom (see EpiMod.gemm)
        **operands,
    ):
        """Resolve (and compile on a cold cache) WITHOUT launching; returns an
        :class:`EpiPlan` whose ``run()`` makes zero host decisions. Default
        reduce scratch is allocated here and attached to the plan (pass fresh
        ``scratch=`` to run() for concurrent streams)."""
        varlen_m = cu_seqlens_m is not None
        b_kn = get_device_capacity(A.device)[0] >= 9
        B_d = B if b_kn else B.mT
        if transform_a is not None:
            from torch._vendor.quack.operand_transform.host import as_transform_mod

            transform_a = as_transform_mod(transform_a)
        cfg = config if config is not None else self._default_config(A, B, transform_a)
        dyn = dynamic_scheduler or cfg.is_dynamic_persistent
        out = dict(out)
        D = out.get("D")
        for name in self.outputs:
            if out.get(name) is None:
                raise ValueError(f"plan() requires a buffer for output {name!r}")
        n = B.shape[-1]
        A_bundle = None
        if transform_a is not None:
            if transform_a.owned_fmt is not None:
                B_d = B  # the blob crosses kernel-native
                n = transform_a.padded_n(B)
            elif transform_a.needs_operands:
                if not transform_operands:
                    raise ValueError(
                        f"transform_a {transform_a.name!r} declares "
                        "runtime operands: pass transform_operands={name: tensor}"
                    )
                A_bundle = transform_a.bundle(A, transform_operands, cfg.tile_m, cfg.tile_k)
        ctx = dict(
            A=A,
            A_bundle=A_bundle,
            B_d=B_d,
            C=C,
            D=D,
            out=out,
            operands=dict(operands),
            n=n,
            lead=self._lead_shape(A, cu_seqlens_m, A_idx),
            b_kn=b_kn,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            rounding_mode=rounding_mode,
            epi_key_overrides=epi_key_overrides,
            transform_a=transform_a,
            transform_sf=transform_sf,
            add_to_output=add_to_output,
        )
        _, _, gemm_plan, sink_bufs = self._iface_execute(cfg, dyn, ctx, _launch=False)
        return EpiPlan(
            gemm_plan=gemm_plan,
            config=cfg,
            out_names=tuple(self.outputs),
            b_kn=b_kn,
            swapped=cfg.swap_ab,
            scratch={
                **sink_bufs,
                **{k: v for k, v in ctx["operands"].items() if k in self.sinks},
            },
        )


# fn_port values a pinned OPERAND may declare. "value": the op's begin_loop
# fragment must be elementwise congruent with tRS_rD and DENSE (the
# vectorizer rejects zero-stride loop loads); it is indexed like a tile
# fragment in every mode. "sink" ports are fn RETURNS, never operands.
_OPERAND_PORTS = {"apply", "col", "row", "scalar", "tile", "value"}


def _pinned_visit_kind(op):
    """The fn-loop visit kind of a pinned operand op: its declared fn_port —
    one method makes any op compose (no isinstance dispatch)."""
    port = op.fn_port
    if port not in _OPERAND_PORTS:
        raise ValueError(
            f"cannot use {type(op).__name__} (fn_port={port!r}) as a fn-frontend "
            "operand (write a mixin)"
        )
    return port


def gemm_epilogue(
    outputs=(),
    ops=None,
    reduces=None,
    mode=None,
    paired=(),
    outs=None,
    prepass=None,
    prepass_outs=(),
    extra_ops=(),
    vectorize=None,
):
    """Decorator: turn an elementwise fn into a fused GEMM epilogue. See module
    docstring for the contract. ``ops`` pins operand names to explicit EpiOp
    instances when shape inference is ambiguous. ``reduces`` declares reduce
    outputs ({name: ColVecReduce(name) or RowVecReduce(name)}): the fn returns
    the per-element value to accumulate under that name, the buffer arrives in
    ``epi_args`` shaped (l, m, n_tiles) for col / (l, m_tiles, n) for row.

    ``mode='acc_pair'`` is expressed in the fn body with ``unpack``/``pack``
    (see :class:`Pair`): gated is ``gate, up = unpack(acc)`` with a
    half-of-GEMM-N aux buffer
    (per-pair aux is 16-bit n-major; interleave gate/up along N in B exactly
    as with the hand-written kernels; row/tile/c operands arrive paired, col
    operands as one scalar since they broadcast along N). Use
    ``mode='packed_cd_b16x2'`` for dgated:
    ``x, y = unpack(c)`` + ``"D": pack(dx, dy)`` with C/D passed as their
    natural 16-bit n-major tensors at twice GEMM-N.

    ``vectorize=False`` keeps the SM100 fn-loop vectorizer off for this
    epilogue (no effect on other archs — only SM100 vectorizes this loop).
    Escape hatch for the upstream DSL vectorizer's crash bugs (fused
    two-result ``cute.math.sincos``; an arith-computed value reused in both
    ``pack()`` lanes — nondeterministic segfault/double-free/compile hang).
    It disables f32x2 packing of the fn math: free while the epilogue hides
    under the mainloop (rope_posfreq B300 K=4096: novec +0.01% vs vectorized
    +0.46% over bias-only), expensive once exposed (K=512: +72% vs +51% — the
    unpacked stream goes issue-bound). Escape hatch, not a tuning knob.
    Changing it changes the kernel cache key."""

    def wrap(fn):
        return EpiMod(
            fn,
            outputs=outputs,
            ops=ops,
            reduces=reduces,
            mode=mode,
            paired=paired,
            outs=outs,
            prepass=prepass,
            prepass_outs=prepass_outs,
            extra_ops=extra_ops,
            vectorize=vectorize,
        )

    return wrap


def fragment_epilogue(
    outputs=(),
    ops=None,
    outs=None,
    extra_ops=(),
    prepass=None,
    prepass_outs=(),
):
    """Decorate a whole-fragment TensorSSA callback using the EpiMod host stack."""

    def wrap(fn):
        return EpiMod(
            fn,
            outputs=outputs,
            ops=ops,
            outs=outs,
            extra_ops=extra_ops,
            prepass=prepass,
            prepass_outs=prepass_outs,
            fragmentwise=True,
        )

    return wrap


class EpiPlan:
    """Prepared launch for one metadata class (the decode / CUDA-graph entry
    point). ``run()`` performs zero host decisions by construction: no key, no
    compile-cache probe, no allocation — only the B relabel the compiled
    signature demands (off SM90+/varlen) and the launch. Tensors must match
    the metadata plan() saw; that promise is the caller's (an interface layer
    holding this plan keys for it)."""

    __slots__ = ("gemm_plan", "config", "out_names", "b_kn", "scratch", "swapped")

    def __init__(self, *, gemm_plan, config, out_names, b_kn, scratch, swapped=False):
        self.gemm_plan = gemm_plan
        self.config = config
        self.out_names = out_names
        self.b_kn = b_kn
        self.swapped = swapped
        # Default reduce partial buffers, allocated at plan(). Shared across
        # run() calls on one stream; pass scratch= for concurrent streams.
        self.scratch = scratch

    def run(
        self,
        A,
        B,  # (k, n) logical, as given to plan()
        C=None,
        *,
        out,
        scratch=None,
        tile_count_semaphore=None,
        cu_seqlens_m=None,
        A_idx=None,
        SFA=None,
        SFB=None,
        **operands,
    ):
        D = out.get("D")
        if not self.b_kn:
            B = B.mT
        epi_values = dict(self.scratch if scratch is None else scratch)
        epi_values.update(operands)
        for name in self.out_names:
            v = out.get(name)
            if v is not None:
                epi_values[name] = v
        run_gemm_epi_plan(
            self.gemm_plan,
            B if self.swapped else A,
            A if self.swapped else B,
            D,
            C,
            epi_values,
            tile_count_semaphore=tile_count_semaphore,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            SFA=SFA,
            SFB=SFB,
        )


class StaticEpi:
    """Rung-3 escape hatch: the same plan/run interface for a hand-written
    GEMM class (custom EpiOps, dataflow the fn contract can't express). Power
    API: no operand inference and no allocation — B arrives dispatch-shaped
    (n, k) unless b_kn, epi_args are explicit, outputs are epi_args entries.
    Requires every op with a host argument to implement the host schema trio
    (host_arg_key / host_fake_arg / host_call_arg)."""

    def __init__(self, GemmCls):
        self.GemmCls = GemmCls

    def plan(
        self,
        A,
        B,
        D,
        C=None,
        *,
        epi_args,
        config=None,
        dynamic_scheduler=False,
        b_kn=False,
        varlen_m=False,
        gather_A=False,
        epi_key_overrides=None,
    ):
        cfg = config if config is not None else default_config(A.device)
        dyn = dynamic_scheduler or cfg.is_dynamic_persistent
        gemm_plan = build_gemm_epi_plan(
            self.GemmCls,
            get_device_capacity(A.device),
            A,
            B,
            D,
            C,
            epi_values=epi_args,
            epi_key_overrides=epi_key_overrides,
            tile_M=cfg.tile_m,
            tile_N=cfg.tile_n,
            tile_K=cfg.tile_k,
            cluster_M=cfg.cluster_m,
            cluster_N=cfg.cluster_n,
            pingpong=cfg.pingpong,
            persistent=True,
            is_dynamic_persistent=dyn,
            max_swizzle_size=cfg.max_swizzle_size,
            varlen_m=varlen_m,
            gather_A=gather_A,
            b_kn=b_kn,
        )
        return EpiPlan(
            gemm_plan=gemm_plan,
            config=cfg,
            out_names=(),
            b_kn=True,  # B is already dispatch-shaped: run() must not relabel
            scratch={},
        )


def epilogue_from_class(GemmCls) -> StaticEpi:
    """Wrap a hand-written epilogue GEMM class in the plan/run interface."""
    return StaticEpi(GemmCls)
