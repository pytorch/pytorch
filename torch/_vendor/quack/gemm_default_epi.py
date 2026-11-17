# Copyright (c) 2025, Wentao Guo, Tri Dao.
from typing import NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from torch._vendor.quack.cute_dsl_utils import mlir_namedtuple
from torch._vendor.quack.epilogue.mixin import ComposableEpiMixin
from torch._vendor.quack.epilogue.ops import Scalar, RowVecLoad, ColVecLoad
from torch._vendor.quack.epilogue.quantize_out import BlockScaleFactorStore
from torch._vendor.quack.gemm_sm80 import GemmSm80
from torch._vendor.quack.gemm_sm90 import GemmSm90
from torch._vendor.quack.gemm_sm100 import GemmSm100
from torch._vendor.quack.gemm_sm120 import GemmSm120
from torch._vendor.quack.rounding import RoundingMode
import torch._vendor.quack.layout_utils as layout_utils
import torch._vendor.quack.utils as utils


@cute.jit
def apply_linear_epilogue(
    tRS_rD: cute.Tensor,
    tRS_rC: Optional[cute.Tensor],
    alpha,
    beta,
    tDrRowVec: Optional[cute.Tensor],
    tDrColVec: Optional[cute.Tensor],
) -> None:
    """The default (linear) epilogue math: D = alpha * D + beta * C + rowvec + colvec.

    tRS_rD is mutated in place (acc dtype). alpha/beta are scalar-or-pointer params
    (None means the term's default: alpha absent, beta = 1.0). The vec operands are
    register fragments shaped like tRS_rD. Shared by GemmDefaultEpiMixin and the
    split-K staged reduction kernel (quack/split_k_reduce.py) so the two apply
    bitwise-identical math.
    """
    if const_expr(alpha is not None):
        a = utils.load_scalar_or_pointer(alpha)
        rD = tRS_rD.load() * a
        tRS_rD.store(rD)
    # Apply C with beta scaling
    if const_expr(tRS_rC is not None):
        if const_expr(
            beta is None and tRS_rC.element_type.width == 16 and tRS_rD.element_type == Float32
        ):
            # Plain 16-bit C add: scalar adds so the widen folds into the add
            # (PTX add.rn.f32.{f16,bf16} -> FHADD on SM100/SM120 — exact, so
            # bitwise-identical to cvt+add; pre-Blackwell lowers to the same
            # cvt+FADD the TensorSSA form produced).
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] + tRS_rC[i].to(tRS_rD.element_type)
        else:
            rD = tRS_rD.load()
            if const_expr(beta is None):
                # beta is None, default behavior: add C (beta=1.0)
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                b = utils.load_scalar_or_pointer(beta)
                rD += b * tRS_rC.load().to(tRS_rD.element_type)
            tRS_rD.store(rD)
    if const_expr(tDrRowVec is not None):
        for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
            tRS_rD[i] += tDrRowVec[i]
    if const_expr(tDrColVec is not None):
        for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
            tRS_rD[i] += tDrColVec[i]


class GemmDefaultEpiMixin(ComposableEpiMixin):
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
        # D quantize codecs (at most one active): the driver's store loop runs
        # the active codec on the final D fragment right before DStore's
        # convert (see gemm_base.epilogue / ComposableEpiMixin._epi_store_quant).
        # Each op's to_params pairs the flat sfd_norm_const arg into its param.
        BlockScaleFactorStore("mSFD"),
        BlockScaleFactorStore("mSFDCol", direction="col"),
    )
    # Split-K (serial/parallel): the epilogue runs only on the finalizing split, which
    # needs the per-tile completion flag and the raw-f32-partials workspace. Both are
    # CuTe tensors over the (cluster-rounded) tile domain; their layouts own the
    # address computation. Flag: (ntile_m, ntile_n, L) Int32. Workspace:
    # (cta_tile_m * cta_tile_n, ntile_m, ntile_n, L) Float32, each tile's region a
    # flat (epi_subtile, thread, fragment)-striped dump of the accumulator fragments.
    _extra_param_fields = (
        ("split_k_semaphore", Optional[cute.Tensor], None),
        ("split_k_workspace", Optional[cute.Tensor], None),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        split_k_semaphore: Optional[cute.Tensor] = None
        split_k_workspace: Optional[cute.Tensor] = None
        # Blocked (L, rm, rk, 32, 4, 4) output scale factors for quantized D
        # (mxfp8/mxfp4/nvfp4); optional fp32 norm constant folded into the SF.
        # mSFD: SF vectors along N (row direction). mSFDCol: SF vectors along
        # M ((L, rn, rm_k, 32, 4, 4) blocked over (N, M), for backward
        # consumers contracting over this output's M). At most one of the two.
        mSFD: Optional[cute.Tensor] = None
        sfd_norm_const: Optional[Float32 | cute.Tensor] = None
        mSFDCol: Optional[cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        for key in ("mRowVecBroadcast", "mColVecBroadcast"):
            if key in self.concat_layout and key in d:
                d[key] = layout_utils.concat_to_interleave(d[key], 1)
        d["split_k_semaphore"] = getattr(args, "split_k_semaphore", None)
        d["split_k_workspace"] = getattr(args, "split_k_workspace", None)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Tensor, ...]:
        """Return a tuple of register tensors (one per aux output).

        The returned tuple must be the same length as the tuple returned
        from :meth:`epi_setup_aux_out`. The default impl returns ``()`` —
        no aux outputs.
        """
        # Use .get(): inactive ops are filtered out of epi_loop_tensors.
        # Under split-K this runs exactly once per output tile, on the finalizing
        # entity, with the fully reduced accumulator in tRS_rD — no per-split gating.
        apply_linear_epilogue(
            tRS_rD,
            tRS_rC,
            params.alpha if const_expr(hasattr(params, "alpha")) else None,
            params.beta if const_expr(hasattr(params, "beta")) else None,
            epi_loop_tensors.get("mRowVecBroadcast"),
            epi_loop_tensors.get("mColVecBroadcast"),
        )
        return ()


class GemmDefaultSm80(GemmDefaultEpiMixin, GemmSm80):
    pass


class GemmDefaultSm90(GemmDefaultEpiMixin, GemmSm90):
    pass


class GemmDefaultSm100(GemmDefaultEpiMixin, GemmSm100):
    pass


class GemmDefaultSm120(GemmDefaultEpiMixin, GemmSm120):
    pass
