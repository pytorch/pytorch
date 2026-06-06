# Copyright (c) 2025, Tri Dao.
"""ComposableEpiMixin: composes EpiOps into epilogue hook methods.

Subclasses declare _epi_ops as a class-level tuple of EpiOp instances — the
static *schema* for the epilogue. The mixin auto-generates epi_smem_bytes,
epi_get_smem_struct, epi_get_smem_tensors, epi_begin, epi_begin_loop,
epi_end_loop, epi_end, and EpilogueParams by querying each op.

Host-side, `_epi_ops_to_params_dict` (called from each subclass's
`epi_to_underlying_arguments`) filters `_epi_ops` automatically: it shadows
the class-level tuple with an instance-level tuple containing only the ops
whose argument tensor is non-None. All later iteration (host- and
device-side) walks the filtered tuple, so each EpiOp's hook methods can
assume their `param`/`arg_tensor` is non-None.

The two host-side hooks that run *before* `epi_to_underlying_arguments`
(`resolve_epi_m_major` and the classmethod `epi_smem_bytes`) filter inline
from `args`, preserving the same non-None invariant for `op.epi_m_major_score`
and `op.smem_bytes`. They have to run first because `epi_to_underlying_arguments`
itself depends on static attributes set up before it (e.g. `gemm.epi_tile`,
`gemm.epi_c_stage`), and those attributes are themselves derived from
`epi_m_major` and the epi smem budget — a chicken-and-egg ordering we resolve by
letting these two hooks see the raw `args` and filter inline.

epi_get_smem_tensors, epi_begin, and epi_begin_loop all return dicts keyed by
op name, so consumers access values by name (e.g. epi_smem_tensors["mAuxOut"],
epi_loop_tensors["alpha"]). Because inactive ops are filtered out, consumers
must use `.get(name)` (returns None for inactive ops) rather than `[name]`.

EpilogueParams is auto-generated from the full class-level _epi_ops (via
param_fields()) plus any _extra_param_fields declared on the subclass.
Subclasses still define EpilogueArguments and epi_to_underlying_arguments
manually.
"""

from dataclasses import make_dataclass, MISSING

import cutlass.cute as cute
from cutlass import Int32, const_expr

from .cute_dsl_utils import ParamsBase
from .epi_ops import EpiContext, EpiSmemBytes, Scalar


def _make_epi_params(epi_ops, extra_fields, bases):
    """Build EpilogueParams dataclass from epi_ops + extra fields.

    Required fields (default=MISSING) are placed first, then optional fields.
    """
    required, optional = [], []
    for op in epi_ops:
        for name, typ, default in op.param_fields():
            (required if default is MISSING else optional).append((name, typ, default))
    for name, typ, default in extra_fields:
        (required if default is MISSING else optional).append((name, typ, default))
    fields = [(n, t) for n, t, _ in required] + [(n, t, d) for n, t, d in optional]
    return make_dataclass("EpilogueParams", fields, bases=bases)


class ComposableEpiMixin:
    """Base mixin that composes EpiOps into the standard epilogue hooks."""

    _epi_ops = ()
    _extra_param_fields = ()  # [(name, type, default), ...] for non-op params (e.g. act_fn)
    _epi_param_bases = (ParamsBase,)  # Base classes for the auto-generated EpilogueParams

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._epi_ops:
            # Auto-generate EpilogueParams if not explicitly defined on this class
            if "EpilogueParams" not in cls.__dict__:
                cls.EpilogueParams = _make_epi_params(
                    cls._epi_ops, cls._extra_param_fields, cls._epi_param_bases
                )

    # --- Host-side: args → params ---

    def _filter_epi_ops(self, args):
        """Shadow `_epi_ops` with an instance-level tuple of only the ops whose
        arg is non-None. Called automatically by `_epi_ops_to_params_dict`, so
        subclass `epi_to_underlying_arguments` methods don't need to invoke it
        directly. After this runs, op hook methods can assume their
        `param`/`arg_tensor` is non-None."""
        self._epi_ops = tuple(
            op
            for op in type(self)._epi_ops
            if getattr(args, op.name, None) is not None
            or (
                op.name == "mRowVecReduce"
                and args.local_reduce_feeds_main
                and args.local_reduce_dim == 0
            )
        )

    def _epi_ops_to_params_dict(self, args):
        """Filter `_epi_ops` to active ops, then merge each op's to_params into
        a single dict. Subclasses call this from epi_to_underlying_arguments,
        add custom fields, then construct self.EpilogueParams(**d). Filtering
        here means every later iteration of self._epi_ops (host- and
        device-side) walks only active ops, and each op hook can assume its
        arg is non-None."""
        self._filter_epi_ops(args)
        d = {}
        for op in self._epi_ops:
            d.update(op.to_params(self, args))
        return d

    def resolve_epi_m_major(self, args):
        # Runs inside _setup_attributes, before epi_to_underlying_arguments,
        # because epi_m_major drives epi_tile / smem layout choices that
        # epi_to_underlying_arguments later consumes. self._epi_ops is still
        # the class-level schema at this point, so we filter inline from args
        # to keep op.epi_m_major_score's non-None invariant.
        score = 0
        for op in type(self)._epi_ops:
            arg = getattr(args, op.name, None)
            if arg is not None:
                score += op.epi_m_major_score(arg, self)
        return score >= 0

    # --- Host-side: smem allocation (queried from ops) ---

    @classmethod
    def epi_smem_bytes(cls, args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # Runs inside _compute_stages, before epi_to_underlying_arguments,
        # because the AB/epi stage counts (and therefore epi_c_stage) depend
        # on the epi smem budget that this returns; epi_to_underlying_arguments
        # then consumes epi_c_stage to build TileLoad's staged smem layout.
        # Stays a classmethod because _compute_stages is a classmethod and may
        # be invoked without an instance, so we filter inline from args.
        result = EpiSmemBytes()
        for op in cls._epi_ops:
            arg = getattr(args, op.name, None)
            if arg is not None:
                result += op.smem_bytes(arg, cta_tile_shape_mnk, epi_tile, warp_shape_mnk)
        return result

    def epi_get_smem_struct(self, params):
        fields = []
        for op in self._epi_ops:
            result = op.smem_struct_field(self, params)
            if result is not None:
                fields.append(result)

        # cute.struct rejects empty annotations. When every active op contributes
        # no smem (e.g. only Scalar ops, or no active ops at all), return a
        # zero-byte placeholder matching gemm_base's default epi struct.
        if not fields:
            return cute.struct.MemRange[Int32, 0]

        # Sort smallest-to-largest so smaller fields pack ahead of larger
        # higher-aligned fields, reducing smem wasted to alignment padding.
        def _field_bytes(name_ftype):
            wrapper = type("_F", (), {"__annotations__": {name_ftype[0]: name_ftype[1]}})
            return cute.struct(wrapper).size_in_bytes()

        fields.sort(key=_field_bytes)
        annotations = {name: ftype for name, ftype in fields}
        EpiSharedStorage = type("EpiSharedStorage", (), {"__annotations__": annotations})
        return cute.struct(EpiSharedStorage)

    def epi_get_smem_tensors(self, params, storage):
        return {
            op.name: op.get_smem_tensor(self, params, storage.epi)
            for op in self._epi_ops
            if not isinstance(op, Scalar)
        }

    def epi_get_tma_atoms(self, params, *, loc=None, ip=None):
        atoms = []
        for op in self._epi_ops:
            atoms.extend(op.tma_atoms(self, params))
        return atoms

    # --- Device-side: kernel execution (delegates to ops) ---

    @cute.jit
    def epi_begin(
        self,
        params,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
        tRS_rD_layout=None,
    ):
        ctx = EpiContext(
            self,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
            tRS_rD_layout,
        )
        results = {
            op.name: op.begin(
                self,
                getattr(params, op.name),
                epi_smem_tensors.get(op.name),
                ctx,
            )
            for op in self._epi_ops
        }
        # self._epi_ops is filtered to active ops, so any op needing a fence
        # has a non-None tensor; no inner None check required.
        has_async_data = any(op.needs_async_fence() for op in self._epi_ops)
        if const_expr(has_async_data):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epilogue_barrier.arrive_and_wait()
        return results

    def epi_begin_loop(self, params, epi_tensors, epi_coord):
        return {
            op.name: op.begin_loop(self, epi_tensors[op.name], epi_coord) for op in self._epi_ops
        }

    def epi_tile_load_g2s_copy_fns(
        self,
        params,
        epi_smem_tensors,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        return tuple(
            op.load_g2s_copy_fn(
                self,
                params,
                epi_smem_tensors.get(op.name),
                tile_coord_mnkl,
                varlen_manager,
                epi_pipeline,
            )
            for op in self._epi_ops
            if op.is_tile_load()
        )

    @cute.jit
    def epi_tile_load_s2r(self, params, epi_tensors, stage_idx):
        for op in self._epi_ops:
            op.load_s2r(self, getattr(params, op.name), epi_tensors[op.name], stage_idx)

    @cute.jit
    def epi_end_loop(
        self,
        params,
        epi_tensors,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        for op in self._epi_ops:
            op.end_loop(
                self,
                getattr(params, op.name),
                epi_tensors[op.name],
                epi_coord,
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )

    @cute.jit
    def epi_end(
        self,
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        for op in self._epi_ops:
            op.end(
                self,
                getattr(params, op.name),
                epi_tensors[op.name],
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )
