# Copyright (c) 2025, Tri Dao.
"""ComposableEpiMixin: composes EpiOps into epilogue hook methods.

Subclasses declare _epi_ops as a tuple of EpiOp instances. The mixin auto-generates
epi_smem_bytes_per_stage, epi_get_smem_struct, epi_get_smem_tensors, epi_begin,
epi_begin_loop, epi_end_loop, epi_end, and EpilogueParams by querying each op.

epi_get_smem_tensors, epi_begin, and epi_begin_loop all return dicts keyed by op
name, so consumers access values by name (e.g. epi_smem_tensors["mAuxOut"],
epi_loop_tensors["alpha"]).

EpilogueParams is auto-generated from _epi_ops (via param_fields()) plus any
_extra_param_fields declared on the subclass. Subclasses still define
EpilogueArguments and epi_to_underlying_arguments manually.
"""

from dataclasses import make_dataclass, MISSING

import cutlass.cute as cute
from cutlass import const_expr

from .cute_dsl_utils import ParamsBase
from .epi_ops import EpiContext, Scalar


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
    _epi_has_async_ops = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._epi_ops:
            cls._epi_has_async_ops = any(op.needs_async_fence() for op in cls._epi_ops)
            # Auto-generate EpilogueParams if not explicitly defined on this class
            if "EpilogueParams" not in cls.__dict__:
                cls.EpilogueParams = _make_epi_params(
                    cls._epi_ops, cls._extra_param_fields, cls._epi_param_bases
                )

    # --- Host-side: args → params ---

    def _epi_ops_to_params_dict(self, args):
        """Merge each op's to_params into a single dict. Subclasses call this,
        add custom fields, then construct self.EpilogueParams(**d)."""
        d = {}
        for op in self._epi_ops:
            d.update(op.to_params(self, args))
        return d

    def resolve_epi_m_major(self, args):
        score = sum(
            op.epi_m_major_score(getattr(args, op.name, None), self) for op in self._epi_ops
        )
        return score >= 0

    # --- Host-side: smem allocation (queried from ops) ---

    @classmethod
    def epi_smem_bytes_per_stage(cls, args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        return sum(
            op.smem_bytes(
                getattr(args, op.name, None), cta_tile_shape_mnk, epi_tile, warp_shape_mnk
            )
            for op in cls._epi_ops
        )

    def epi_get_smem_struct(self, params):
        fields = []
        for op in self._epi_ops:
            result = op.smem_struct_field(self, params)
            if result is not None:
                fields.append(result)

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
        )
        results = {
            op.name: op.begin(
                self,
                getattr(params, op.name, None),
                epi_smem_tensors.get(op.name),
                ctx,
            )
            for op in self._epi_ops
        }
        if const_expr(self._epi_has_async_ops):
            has_async_data = any(
                getattr(params, op.name, None) is not None
                for op in self._epi_ops
                if op.needs_async_fence()
            )
            if const_expr(has_async_data):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                epilogue_barrier.arrive_and_wait()
        return results

    def epi_begin_loop(self, params, epi_tensors, epi_coord):
        return {
            op.name: op.begin_loop(self, epi_tensors[op.name], epi_coord) for op in self._epi_ops
        }

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
                getattr(params, op.name, None),
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
                getattr(params, op.name, None),
                epi_tensors[op.name],
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )
