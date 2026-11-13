# Copyright (c) 2026, Han Guo, Tri Dao.
"""Device half of the fn epilogue frontend: the traced per-element visit /
prepass loops of minted ``@gemm_epilogue`` kernels. The minted class (see
:mod:`quack.epilogue.frontend`) supplies the fn and the operand/output/sink
signatures as class attributes; everything here is generic over them. Loop
shapes mirror the hand-written mixins exactly (packed f32x2 lanes on SM100,
pair views for gated/dgated) — see the frontend module docstring for the
contract and the speed-of-light rules."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

import torch._vendor.quack.layout_utils as layout_utils
from torch._vendor.quack.epilogue.math import F2, F16Lanes, Pair
from torch._vendor.quack.epilogue.mixin import ComposableEpiMixin
from torch._vendor.quack.rounding import RoundingMode


class _EpiModMixinBase(ComposableEpiMixin):
    """Generic hooks for minted epilogue-mod kernels. The minted class supplies
    ``_epi_ops``, ``_epi_mod_fn``, ``_epi_mod_operands`` ((name, kind) pairs),
    ``_epi_mod_outputs``, and ``EpilogueArguments``."""

    _epi_mod_fn = None
    _epi_mod_operands = ()
    _epi_mod_outputs = ()
    _epi_mod_sinks = ()  # names of sink-port ops (fn returns them; op consumes)
    _epi_mod_group_n = 1  # 2 = gated: fn consumes adjacent-N pairs, aux is half-width
    _epi_mod_packed_cd = False  # dgated: C/D pack 2 x implicit_dtype lanes per f32
    _epi_mod_prepass_fn = None  # fn run over the raw accumulator before any store
    _epi_mod_prepass_operands = ()  # ((name, kind), ...) subset the prepass fn reads
    _epi_mod_prepass_outs = ()  # sink-op names the prepass fn returns
    _epi_mod_rounding = RoundingMode.RN  # kernel-global rounding (D store + default for TileStores)
    _epi_mod_vectorize = None  # False = keep the SM100 loop vectorizer off (escape hatch)
    _extra_param_fields = ()  # the fn is a class attr, not a param

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = self._epi_mod_rounding
        self.epi_needs_acc_prepass = self._epi_mod_prepass_fn is not None
        if self._epi_mod_packed_cd:
            assert self.implicit_dtype.width == 16, "packed_cd lanes must be 16-bit"
            assert self.d_dtype.width == 32, "packed_cd D storage must be 32-bit (f32 view)"
            assert self.c_dtype.width == 32, "packed_cd C storage must be 32-bit (f32 view)"
        # Aux-output constraints (gated 16-bit n-major, SM90 tile_N % 32) are
        # asserted by each TileStore op in to_params; the store path itself is
        # the generic ComposableEpiMixin/TileStore one.
        d = self._epi_ops_to_params_dict(args)
        for key in getattr(self, "concat_layout", None) or ():
            if key in d:
                d[key] = layout_utils.concat_to_interleave(d[key], 1)
        d["split_k_semaphore"] = getattr(args, "split_k_semaphore", None)
        d["split_k_workspace"] = getattr(args, "split_k_workspace", None)
        return self.EpilogueParams(**d)

    def _make_sink_tmps(self, ops_by_name, shape):
        """One collection fragment per sink op; scaled reduces get a
        (val, scale) fragment pair so the fold can be a single fused FMA."""
        return tuple(
            (
                (
                    cute.make_rmem_tensor(shape, self.acc_dtype),
                    cute.make_rmem_tensor(shape, self.acc_dtype),
                )
                if getattr(ops_by_name[s], "scaled", False)
                else cute.make_rmem_tensor(shape, self.acc_dtype)
            )
            for s in self._epi_mod_sinks
        )

    @cute.jit
    def _flush_sinks(self, ops_by_name, epi_loop_tensors, sink_tmps):
        for sname, stmp in zip(self._epi_mod_sinks, sink_tmps):
            if const_expr(isinstance(stmp, tuple)):
                ops_by_name[sname].fn_sink_flush(
                    self, epi_loop_tensors[sname], stmp[0], scale=stmp[1]
                )
            else:
                ops_by_name[sname].fn_sink_flush(self, epi_loop_tensors[sname], stmp)

    @cute.jit
    def epi_prepass_subtile(self, params, epi_tensors, tRS_rD, epi_coord, epi_idx):
        """Driver prepass hook (epi_needs_acc_prepass): run the prepass fn over
        this subtile's raw accumulator, collect its returns, flush to the
        prepass sink ops. Scalar unrolled loop — the prepass is a statistics
        sweep, not the store path."""
        pfn = self._epi_mod_prepass_fn
        ops_by_name = {op.name: op for op in self._epi_ops}
        frags = {}
        for name, kind in self._epi_mod_prepass_operands:
            state = ops_by_name[name].begin_loop(self, epi_tensors[name], epi_coord)
            if const_expr(kind == "tile"):
                state = state.to(self.acc_dtype)
            frags[name] = state
        sink_states = {
            name: ops_by_name[name].begin_loop(self, epi_tensors[name], epi_coord)
            for name in self._epi_mod_prepass_outs
        }
        tmps = {
            name: cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            for name in self._epi_mod_prepass_outs
        }
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            kw = {
                name: (frags[name] if kind == "scalar" else frags[name][i])
                for name, kind in self._epi_mod_prepass_operands
            }
            res = pfn(tRS_rD[i], **kw)
            for name in self._epi_mod_prepass_outs:
                tmps[name][i] = res[name]
        for name in self._epi_mod_prepass_outs:
            ops_by_name[name].fn_sink_flush(self, sink_states[name], tmps[name])

    @cute.jit
    def epi_prepass_end(self, params, epi_tensors):
        # Flush register-accumulated statistics to smem (ops that batch the
        # prepass sweep in registers expose fn_prepass_end), then order every
        # thread's raw partial-plane stores before register resolution and any
        # direct stats-output fold reads them.
        ops_by_name = {op.name: op for op in self._epi_ops}
        for name in self._epi_mod_prepass_outs:
            op = ops_by_name[name]
            if const_expr(hasattr(op, "fn_prepass_end")):
                op.fn_prepass_end(self, epi_tensors[name])
        self.epilogue_barrier.arrive_and_wait()
        # Resolve pass (grouped stats under a split-N warp layout): after the
        # barrier publishes every raw partial plane, each consumer warp folds
        # the planes into its own finalized register values. There are no
        # shared writes, so no second barrier is needed.
        resolve = [
            name
            for name in self._epi_mod_prepass_outs
            if hasattr(ops_by_name[name], "fn_prepass_resolve")
            and ops_by_name[name].prepass_resolve_needed(self)
        ]
        for name in resolve:
            ops_by_name[name].fn_prepass_resolve(self, epi_tensors[name])

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        fn = self._epi_mod_fn
        ops_by_name = {op.name: op for op in self._epi_ops}
        paired = self._epi_mod_group_n == 2
        # SM100 element mode with 16-bit full-tile inputs (the C operand,
        # TileLoad residual streams): keep them unwidened and hand the fn
        # F16Lanes pairs — additive uses lower to mixed-precision scalar adds
        # (FHADD.BF16/.F16: the promote folds into the add, exactly, saving
        # the cvt/PRMT per lane); every other use sees the promoted F2 and the
        # unused promotes are DCE'd. Only the packed-lane loop needs this;
        # scalar loops get the same fusion from NVVM automatically.
        mixed_lanes_ok = const_expr(
            self.arch == 100
            and not paired
            and not self._epi_mod_packed_cd
            and self.acc_dtype == Float32
            and cute.size(tRS_rD) % 2 == 0  # only the packed-lane loop consumes F16Lanes
        )
        mixed_names = set()
        frags = {}
        for name, kind in self._epi_mod_operands:
            if const_expr(kind == "apply"):
                # Apply-port op: per-subtile port state; the fn gets a callable.
                frags[name] = ops_by_name[name].fn_prepare(self, epi_loop_tensors[name], paired)
            elif const_expr(kind == "c"):
                assert tRS_rC is not None, f"epilogue operand '{name}' requires the C operand"
                if const_expr(mixed_lanes_ok and tRS_rC.element_type.width == 16):
                    frags[name] = tRS_rC
                    mixed_names.add(name)
                elif const_expr(not self._epi_mod_packed_cd):
                    frags[name] = tRS_rC.to(self.acc_dtype)
                # packed_cd: C is recast/unpacked in the packed branch below.
            elif const_expr(kind == "tile"):
                if const_expr(mixed_lanes_ok and epi_loop_tensors[name].element_type.width == 16):
                    frags[name] = epi_loop_tensors[name]
                    mixed_names.add(name)
                else:
                    frags[name] = epi_loop_tensors[name].to(self.acc_dtype)
            elif const_expr(kind == "value"):
                # Custom value-source op: fn_prepare turns its begin_loop state
                # into the dense per-element fragment the loops index (default
                # fn_prepare is identity for ops whose begin_loop IS the frag).
                frags[name] = ops_by_name[name].fn_prepare(self, epi_loop_tensors[name], paired)
            else:  # "row" / "col" fragments are already acc dtype; "scalar" is a value
                frags[name] = epi_loop_tensors[name]
        if const_expr(self._epi_mod_packed_cd):
            # dgated shape: the accumulator is already per-pair (one dout per
            # gate/up pair); C and D pack two implicit-dtype (16-bit) lanes
            # into each 32-bit element. Structure mirrors the hand-written
            # GemmDGatedMixin: recast C -> widen to f32 -> pair views; scalar
            # calls with vectorize on SM100; pack (dx, dy) back into tRS_rD.
            implicit = self.implicit_dtype
            xy16 = cute.recast_tensor(tRS_rC, implicit)
            xy = xy16.to(Float32)
            xy_pair = cute.flat_divide(xy, cute.make_layout(2))
            xv, yv = xy_pair[0, ...], xy_pair[1, ...]
            dxy = cute.make_rmem_tensor(xy16.layout, Float32)
            dxy_pair = cute.flat_divide(dxy, cute.make_layout(2))
            dxv, dyv = dxy_pair[0, ...], dxy_pair[1, ...]
            n_el = cute.size(tRS_rD)

            def _dense1(view):
                # Zero-stride broadcast frags are invalid vectorized loads.
                out = cute.make_rmem_tensor(n_el, self.acc_dtype)
                for j in cutlass.range(n_el, unroll_full=True):
                    out[j] = view[j]
                return out

            views = {}
            for name, kind in self._epi_mod_operands:
                if const_expr(kind in ("row", "col")):
                    views[name] = _dense1(frags[name])
                elif const_expr(kind != "c"):
                    views[name] = frags[name]  # scalar / dense tile frag / apply pstate
            outs = tuple(
                cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
                for _ in self._epi_mod_outputs
            )
            sink_tmps = self._make_sink_tmps(ops_by_name, tRS_rD.layout.shape)
            val_names = self._epi_mod_outputs + self._epi_mod_sinks
            val_frags = outs + sink_tmps
            vectorize = const_expr(self.arch == 100 and self._epi_mod_vectorize is not False)
            for i in cutlass.range(n_el, vectorize=vectorize):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, views[_n], _i, v))
                        if kind == "apply"
                        else Pair(xv[i], yv[i])
                        if kind == "c"
                        else (views[name] if kind == "scalar" else views[name][i])
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(tRS_rD[i], **kw)
                d = res["D"]  # required: it carries the (dx, dy) pair to pack
                dxv[i], dyv[i] = d[0], d[1]
                for vname, vfrag in zip(val_names, val_frags):
                    if const_expr(isinstance(vfrag, tuple)):
                        # Scaled sink: the fn returns the (val, scale) factors.
                        v, s = res[vname]
                        vfrag[0][i], vfrag[1][i] = v, s
                    else:
                        vfrag[i] = res[vname]
            dxy16 = dxy.to(implicit)
            tRS_rD.store(cute.recast_tensor(dxy16, Float32).load())
            self._flush_sinks(ops_by_name, epi_loop_tensors, sink_tmps)
            return outs

        if const_expr(paired):
            # Gated pairs: adjacent-N accumulator lanes feed one fn call; aux
            # fragments are half-width. Same structure as the hand-written
            # GemmGatedMixin: flat_divide pair views built OUTSIDE the loop so
            # every in-loop access is a plain loop index (the SM100 vectorizer
            # rejects affine indices like 2*i), scalar calls + vectorize=True.
            aux_shape = cute.recast_layout(2, 1, tRS_rD.layout).shape
            outs = tuple(
                cute.make_rmem_tensor(aux_shape, self.acc_dtype) for _ in self._epi_mod_outputs
            )
            # Sink values span both lanes (full N): collect through pair views.
            # (Scaled sinks are rejected in acc_pair mode at EpiMod init: a
            # tuple return already means the two lanes here.)
            sink_tmps = tuple(
                cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
                for _ in self._epi_mod_sinks
            )
            sink_views = tuple(
                (p[0, ...], p[1, ...])
                for p in (cute.flat_divide(t, cute.make_layout(2)) for t in sink_tmps)
            )
            acc_pair = cute.flat_divide(tRS_rD, cute.make_layout(2))
            acc0, acc1 = acc_pair[0, ...], acc_pair[1, ...]
            n_groups = cute.size(acc0)

            def _dense(view):
                # Broadcast-vector fragments have zero-stride modes, which the
                # vectorizer rejects as loop loads; materialize a stride-1 copy
                # with an unrolled scalar loop (legal on zero-stride views).
                out = cute.make_rmem_tensor(n_groups, self.acc_dtype)
                for j in cutlass.range(n_groups, unroll_full=True):
                    out[j] = view[j]
                return out

            views = {}
            for name, kind in self._epi_mod_operands:
                if const_expr(kind in ("scalar", "apply")):
                    views[name] = frags[name]
                else:
                    p = cute.flat_divide(frags[name], cute.make_layout(2))
                    if const_expr(kind == "col"):
                        # colvec broadcasts along N: both lanes are identical.
                        views[name] = _dense(p[0, ...])
                    elif const_expr(kind == "row"):
                        views[name] = (_dense(p[0, ...]), _dense(p[1, ...]))
                    else:  # tile / c views are dense by construction
                        views[name] = (p[0, ...], p[1, ...])
            vectorize = const_expr(self.arch == 100 and self._epi_mod_vectorize is not False)
            for i in cutlass.range(cute.size(acc0), unroll_full=True, vectorize=vectorize):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, views[_n], _i, v))
                        if kind == "apply"
                        else views[name]
                        if kind == "scalar"
                        else (
                            views[name][i]
                            if kind == "col"
                            else Pair(views[name][0][i], views[name][1][i])
                        )
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(Pair(acc0[i], acc1[i]), **kw)
                for oname, ofrag in zip(self._epi_mod_outputs, outs):
                    ofrag[i] = res[oname]
                for (s0, s1), sname in zip(sink_views, self._epi_mod_sinks):
                    v = res[sname]
                    s0[i], s1[i] = v[0], v[1]
                if const_expr("D" in res):
                    d = res["D"]
                    acc0[i], acc1[i] = d[0], d[1]
            for sname, stmp in zip(self._epi_mod_sinks, sink_tmps):
                ops_by_name[sname].fn_sink_flush(self, epi_loop_tensors[sname], stmp)
            return outs

        outs = tuple(
            cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            for _ in self._epi_mod_outputs
        )
        # Sink values are collected into a plain fragment per sink op (a
        # (val, scale) fragment pair for scaled reduces), then handed to the
        # op's fn_sink_flush (fragment-level: the op owns the fold into its —
        # possibly aliased, possibly coupled — accumulators).
        sink_tmps = self._make_sink_tmps(ops_by_name, tRS_rD.layout.shape)
        # Names written by the fn, in collection order after "D".
        val_names = self._epi_mod_outputs + self._epi_mod_sinks
        val_frags = outs + sink_tmps
        if const_expr(self.arch == 100 and cute.size(tRS_rD) % 2 == 0):
            # Packed f32x2 lanes: same loop shape as the hand-written SM100 mixins.
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, frags[_n], _i, v))
                        if kind == "apply"
                        else frags[name]
                        if kind == "scalar"
                        else F16Lanes(frags[name][2 * i], frags[name][2 * i + 1])
                        if name in mixed_names
                        else F2(frags[name][2 * i], frags[name][2 * i + 1])
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(F2(tRS_rD[2 * i], tRS_rD[2 * i + 1]), **kw)
                if const_expr("D" in res):
                    d = res["D"]
                    tRS_rD[2 * i], tRS_rD[2 * i + 1] = d[0], d[1]
                for vname, vfrag in zip(val_names, val_frags):
                    if const_expr(isinstance(vfrag, tuple)):
                        v, s = res[vname]
                        vfrag[0][2 * i], vfrag[0][2 * i + 1] = v[0], v[1]
                        vfrag[1][2 * i], vfrag[1][2 * i + 1] = s[0], s[1]
                    else:
                        v = res[vname]
                        vfrag[2 * i], vfrag[2 * i + 1] = v[0], v[1]
        else:
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, frags[_n], _i, v))
                        if kind == "apply"
                        else (frags[name] if kind == "scalar" else frags[name][i])
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(tRS_rD[i], **kw)
                if const_expr("D" in res):
                    tRS_rD[i] = res["D"]
                for vname, vfrag in zip(val_names, val_frags):
                    if const_expr(isinstance(vfrag, tuple)):
                        v, s = res[vname]
                        vfrag[0][i], vfrag[1][i] = v, s
                    else:
                        vfrag[i] = res[vname]
        self._flush_sinks(ops_by_name, epi_loop_tensors, sink_tmps)
        return outs
