import copy
import dataclasses
import logging
import os
import random
import re
import subprocess
from dataclasses import dataclass, fields
from typing import cast, Dict, List, Optional, Tuple

from torch._inductor import config

from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer

from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CKTemplate, CUTLASSTemplate
from .cutlass_epilogue_gen import (
    CutlassEVTEpilogueArgumentFormatter,
    CutlassEVTEpilogueTypeFormatter,
)

log = logging.getLogger(__name__)

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
{{instance_definition}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.
extern "C" {
{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  try {
  {{kernel.check_not_null(X)}}
  {{kernel.check_not_null(W)}}
  {{kernel.check_not_null(Bias)}}
  {{kernel.check_not_null(Y)}}
  int64_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
  int64_t M = {{kernel.size(X, -2)}};
  int64_t K = {{kernel.size(X, -1)}};
  int64_t N = {{kernel.size(W, -1)}};
  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  {{instance_type}}::Arguments arguments;
  {{template.render_gemm_arguments(argument_template, epilogue_template, should_swap_xw,
                                    X, W, Bias, Y, alpha, beta, kernel, epilogue_args)}}
  {{instance_type}} gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }
  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
  }
  catch (std::exception& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }
  catch (...) {
    return -1;
  }
  return 0;
}
}
"""


GEMM_ARGS_CUTLASS_2X = r"""
  int64_t batch_stride_x = {{kernel.stride(X, -3)}};
  int64_t row_stride_x = {{kernel.row_or_column_stride(X)}};
  int64_t batch_stride_w = {{kernel.stride(W, -3)}};
  int64_t row_stride_w = {{kernel.row_or_column_stride(W)}};
  int64_t batch_stride_bias = {{kernel.stride(Bias, -3)}};
  int64_t row_stride_bias = {{kernel.row_or_column_stride(Bias)}};
  int64_t batch_stride_y = {{kernel.stride(Y, -3)}};
  int64_t row_stride_y = {{kernel.row_or_column_stride(Y)}};
  // Initialize GemmUniversalInstance arguments.
  arguments = {
    {{template.gemm_mode()}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K)
    },  // GemmCoord problem_size
    {{split_k if split_k > 1 else 'B'}},  // int batch_count
    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue
    {{template.cutlass_type_cast(X, kernel.ptr(X))}},  // void const * ptr_A
    {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // void const * ptr_B
    {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // void const * ptr_C
    {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // void * ptr_D
    batch_stride_x,  // int64_t batch_stride_A
    batch_stride_w,  // int64_t batch_stride_B
    batch_stride_bias,  // int64_t batch_stride_C
    batch_stride_y,  // int64_t batch_stride_D
    row_stride_x,  // typename LayoutA::Stride::LongIndex lda
    row_stride_w,  // typename LayoutB::Stride::LongIndex ldb
    row_stride_bias,  // typename LayoutC::Stride::LongIndex ldc
    row_stride_y,  // typename LayoutC::Stride::LongIndex ldd
  };
"""


GEMM_ARGS_CUTLASS_3X = r"""
  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    {{template.gemm_mode()}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>({{M}}),
      static_cast<coord_t>({{N}}),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      {{template.cutlass_type_cast(X, kernel.ptr(X))}},  // ElementA const* ptr_A
      {
        {{template.cute_int(kernel.stride(X, -2), "stride_x0")}},
        {{template.cute_int(kernel.stride(X, -1), "stride_x1")}},
        {{template.cute_int(kernel.stride(X, -3), "batch_stride_x")}}
      },  // StrideA dA
      {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // ElementB const* ptr_B
      {
        {{template.cute_int(kernel.stride(W, -1), "stride_w1")}},
        {{template.cute_int(kernel.stride(W, -2), "stride_w0")}},
        {{template.cute_int(kernel.stride(W, -3), "batch_stride_w")}}
      },  // StrideB dB
    },  // MainloopArguments mainloop
    {{epilogue_arguments}}
  };
"""

GEMM_ARGS_CUTLASS_3X_EPILOGUE = r"""
    // see https://tinyurl.com/4rk89z48
    {
      {{epilogue_args}},  // thread, typename FusionCallbacks::Arguments ( EVT ) or ThreadEpilogueOp::Params (non-EVT )
      {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // ElementC const* ptr_C
      {
        {{template.cute_int(kernel.stride(Bias, -2, 1), "stride_bias0")}},
        {{template.cute_int(kernel.stride(Bias, -1, 1), "stride_bias1")}},
        {{template.cute_int(kernel.stride(Bias, -3), "batch_stride_bias")}}
      },  // StrideC dC
      {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // ElementD const* ptr_D
      {
        {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}},
        {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}},
        {{template.cute_int(kernel.stride(Y, -3), "batch_stride_y")}}
      },  // StrideD dD
    },  // EpilogueArguments epilogue
"""


class CUTLASSGemmTemplate(CUTLASSTemplate):
    """
    CUTLASS GEMM template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
        can_fuse_epilogue: Optional[bool] = None,
    ):
        """
        Args:
            input_nodes: input nodes of the kernel
            layout: layout of the output node
            alpha: alpha value of the GEMM operation
            beta: beta value of the GEMM operation
            input_reorder: reorder of the input nodes
            can_fuse_epilogue: If set to True, will only list and use operators capable of flexible epilogue fusions.
                               If False, it will not use those. If None, both may be listed, but it will not allow fusions.
                               Defaults to None
        """
        super().__init__("cutlass_gemm", input_nodes, layout, input_reorder)
        self.alpha = alpha
        self.beta = beta
        self.can_fuse_epilogue = can_fuse_epilogue

    @staticmethod
    def add_cutlass_gemm_choices(
        choices,
        layout,
        input_nodes,
        alpha=1,
        beta=0,
        input_reorder=None,
        fuseable=True,
        non_fuseable=True,
    ):
        if non_fuseable:
            if fuseable:
                # list both fuseable and non-fuseable ops, and treat them all as non-fuseable
                can_fuse_epilogue = False
            else:
                can_fuse_epilogue = None

            cutlass_template = CUTLASSGemmTemplate(
                input_nodes,
                layout,
                alpha=alpha,
                beta=beta,
                input_reorder=input_reorder,
                can_fuse_epilogue=can_fuse_epilogue,
            )
            ops = cutlass_template.gen_ops()
            for op in ops:
                cutlass_template.maybe_append_choice(
                    choices,
                    op=op,
                )
        else:
            ops = []
        if fuseable:
            cutlass_template_evt = CUTLASSGemmTemplate(
                input_nodes,
                layout,
                alpha=alpha,
                beta=beta,
                input_reorder=input_reorder,
                can_fuse_epilogue=True,
            )
            # This will list only ops capable of EVT fusion
            ops_evt = cutlass_template_evt.gen_ops()
            for op in ops_evt:
                cutlass_template_evt.maybe_append_choice(
                    choices,
                    op=op,
                )
        else:
            ops_evt = []
        log.debug(
            "Added %d cutlass gemm configs and %d fuseable gemm configs.",
            len(ops),
            len(ops_evt),
        )

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                #include "cutlass/gemm/gemm.h"
                #include "cutlass/gemm/device/gemm_universal.h"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"
                #include "cutlass/gemm/kernel/gemm_universal.hpp"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/default_epilogue.hpp"
                #include "cutlass/epilogue/thread/linear_combination.h"
                #include "cutlass/gemm/dispatch_policy.hpp"
                #include "cutlass/gemm/kernel/tile_scheduler.hpp"
                #include "cutlass/util/distribution.h"
                #include "cutlass/util/packed_stride.hpp"
                #include "cutlass/util/tensor_view_io.h"
            """
        )
        return res

    @staticmethod
    def cutlass_layout(torch_layout) -> "Optional[cutlass_lib.LayoutType]":  # type: ignore[name-defined]  # noqa: F821
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        if torch_layout.stride[-1] == 1:
            return cutlass_lib.LayoutType.RowMajor
        elif torch_layout.stride[-2] == 1:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: "cutlass_lib.LayoutType",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_lib.LayoutType":  # type: ignore[name-defined]  # noqa: F821
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return cutlass_lib.LayoutType.RowMajor

    @staticmethod
    def layout_match(torch_layout, cutlass_layout) -> bool:
        return CUTLASSGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout

    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        alignment = cutlass_utils.get_max_alignment(torch_layout)
        cuda_arch = cutlass_utils.get_cuda_arch()
        if cuda_arch and int(cuda_arch) >= 90 and alignment < op_element.alignment:
            return False
        else:
            op_element.alignment = alignment
            return True

    @staticmethod
    def has_tma_epilogue(op) -> bool:
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        result = False
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result

    @staticmethod
    def supports_evt(op: "cutlass_library.gemm_op.GemmOperation") -> bool:  # type: ignore[name-defined]  # noqa: F821
        """
        returns True if the op is capable of flexible epilogue fusions
        using epilogue visitor trees.

        See https://github.com/NVIDIA/cutlass/blob/e01b9b5029b7caca5a43c29f7d2714d7cf1dcae8/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu#L283-L285 # noqa: B950
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind != cutlass_lib.GemmKind.Universal3x:
            return False
        if op.epilogue_schedule not in (
            cutlass_lib.EpilogueScheduleType.TmaWarpSpecialized,
            cutlass_lib.EpilogueScheduleType.TmaWarpSpecializedCooperative,
        ):
            return False

        return True

    def render_evt_epilogue_declaration(
        self,
        template_output_node_name: str,
        evt_type_name: str,
        epilogue_nodes: List[IRNode],
    ) -> str:
        """Generates the epilogue for the EVT epilogue fusion"""
        return CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(
            template_output_node_name, evt_type_name, epilogue_nodes
        )

    def define_gemm_instance(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
        output_buffer_name: str,
        epilogue_nodes: Optional[List[IRNode]] = None,
    ) -> Tuple[str, str]:
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        from torch._inductor.codegen.cuda.cutlass_lib_extensions.gemm_operation_extensions import (
            EmitGemmUniversal3xInstanceWithEVT,
        )

        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            if epilogue_nodes is not None and len(epilogue_nodes) > 0:
                emitter = EmitGemmUniversal3xInstanceWithEVT()
                op.epilogue_functor = lambda epilogue_functor_type_name: self.render_evt_epilogue_declaration(
                    output_buffer_name, epilogue_functor_type_name, epilogue_nodes
                )
            else:
                emitter = cutlass_gemm_op.EmitGemmUniversal3xInstance()
            op_def = emitter.emit(op)
            pattern = re.compile(r"\s*struct\s(.*?)\s:")
            decl = [line for line in op_def.split("\n") if "struct " in line][-1]
        else:
            if epilogue_nodes is not None and len(epilogue_nodes) > 0:
                raise RuntimeError(
                    "EVT epilogue fusion is not supported for Cutlass 2.x ops."
                )
            emitter = cutlass_gemm_op.EmitGemmInstance()
            op_def = emitter.emit(op)
            op_def = op_def.replace(
                "cutlass::gemm::device::Gemm", "cutlass::gemm::device::GemmUniversal"
            )
            op_def = op_def.replace("false,", "")
            pattern = re.compile(r"\s*using\s(.*?)\s=")
            decl = op_def.split("\n")[2]
        match = pattern.match(decl)
        if match is None:
            raise RuntimeError("Invalid Gemm config: \n" + op_def)
        op_type = match.groups()[0]
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            op_def += f"\n  using {op_type}_device_type = cutlass::gemm::device::GemmUniversalAdapter<{op_type}>;\n"
            op_type = f"{op_type}_device_type"
        return op_def, op_type

    @staticmethod
    def should_swap_XW(
        bias: IRNode,
        beta: float,
    ) -> bool:
        return True

        # TODO(ipiszy): Check whether it's necessary to swap X/W.
        # strides = bias.get_stride()
        # if strides[-1] != 1:
        #     return True
        # for stride in strides[:-1]:
        #     if stride != 0:
        #         return True
        # return False

    @staticmethod
    def swap_XW(
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        # Swap X and W in GemmOperation.
        new_op = copy.deepcopy(op)
        new_op.A.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.A.layout)
        new_op.B.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.B.layout)
        new_op.A, new_op.B = new_op.B, new_op.A
        new_op.C.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.C.layout)
        new_op.D.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.D.layout)
        return new_op

    def filter_op(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        # Skip simt kernels
        if (
            op.tile_description.math_instruction.opcode_class
            == cutlass_lib.OpcodeClass.Simt
        ):
            return None

        # Only keep GemmUniversal kernels
        if op.gemm_kind not in {
            cutlass_lib.GemmKind.Universal,
            cutlass_lib.GemmKind.Universal3x,
        }:
            return None
        # Filter ops by dtypes.
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        accumulator_torch_dtype = cutlass_utils.get_accumulator_dtype(
            [X.get_dtype(), W.get_dtype()],
        )
        if not (
            cutlass_utils.dtype_match(X.get_dtype(), op.A.element)
            and cutlass_utils.dtype_match(W.get_dtype(), op.B.element)
            and cutlass_utils.dtype_match(
                self.output_node.get_layout().dtype, op.C.element
            )
            and cutlass_utils.dtype_match(
                accumulator_torch_dtype, op.accumulator_type()
            )
        ):
            return None

        # Filter ops by input layouts.
        if not (
            self.layout_match(X.get_layout(), op.A.layout)
            and self.layout_match(W.get_layout(), op.B.layout)
        ):
            return None

        # Update op.
        op = copy.deepcopy(op)

        # Set output layout.
        op.D.layout = CUTLASSGemmTemplate.cutlass_layout(self.output_node.get_layout())

        # Filter ops by alignments and set alignments.
        if not (
            self.set_alignment(X.get_layout(), op.A)
            and self.set_alignment(W.get_layout(), op.B)
            and self.set_alignment(self.output_node.get_layout(), op.D)
        ):
            return None

        # Set epilogue.
        # TODO: update epilogue functor according to epilogues.
        op.element_epilogue = op.accumulator_type()

        # Set bias layout and alignment.
        if len(self.input_nodes) >= 3 and self.input_nodes[2] is not None:
            Bias = self.input_nodes[2]
            bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
            if op.gemm_kind != cutlass_lib.GemmKind.Universal3x:
                if bias_layout != op.D.layout:
                    # For cutlass2, bias and output layout must match
                    return None
            else:
                op.C.layout = bias_layout
            if not self.set_alignment(Bias.get_layout(), op.C):
                return None
        else:
            if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
                op.C.element = cutlass_lib.DataType.void
            else:
                op.C.layout = op.D.layout
        supports_evt: bool = self.supports_evt(op)
        if (self.can_fuse_epilogue is not None) and (
            self.can_fuse_epilogue != supports_evt
        ):
            return None
        if inductor_cuda_config.cutlass_only_evt_capable_ops and not supports_evt:
            return None
        return op

    def gen_ops(self) -> "List[cutlass_gemm_op.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        ops = cutlass_utils.gen_ops()[cutlass_lib.OperationKind.Gemm]
        res: Dict[str, cutlass_gemm_op.GemmOperation] = dict()
        num_3x_ops = 0
        num_2x_ops = 0
        for op_dict in ops.values():
            for op_list in op_dict.values():
                for op in op_list:
                    assert isinstance(op, cutlass_gemm_op.GemmOperation)
                    filter_res = self.filter_op(op)
                    if (
                        filter_res is not None
                        and res.get(filter_res.configuration_name(), None) is None
                    ):
                        res[filter_res.configuration_name()] = filter_res
        for op in res.values():
            if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
                num_3x_ops += 1
            else:
                num_2x_ops += 1
        log.debug(
            "Got cutlass configs: total number of ops: %d, "
            "total number of 3x ops: %d, total number of 2x ops: %d",
            len(res),
            num_3x_ops,
            num_2x_ops,
        )
        return list(res.values())[: inductor_cuda_config.cutlass_max_profiling_configs]

    def gemm_mode(self) -> str:
        sizes = self.output_node.get_size()
        if len(sizes) > 2:
            return "cutlass::gemm::GemmUniversalMode::kBatched"
        else:
            return "cutlass::gemm::GemmUniversalMode::kGemm"

    def render_gemm_arguments(
        self,
        argument_template: str,
        epilogue_template: str,
        should_swap_xw: bool,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        kernel: CUDATemplateKernel,
        epilogue_args,
    ) -> str:
        options = dict(
            alpha=self.alpha,
            beta=self.beta,
            X=X,
            W=W,
            Y=Y,
            Bias=Bias,
            template=self,
            kernel=kernel,
            M="M",
            N="N",
            epilogue_args=epilogue_args,
        )

        if epilogue_template is not None:
            if should_swap_xw:
                # Swap
                def clone_with_transposed_stride(node: IRNode) -> IRNode:
                    old_layout = node.get_layout()
                    new_stride = list(old_layout.stride)
                    new_stride[-2], new_stride[-1] = new_stride[-1], new_stride[-2]
                    new_layout = FixedLayout(
                        old_layout.device,
                        old_layout.dtype,
                        list(old_layout.size),
                        new_stride,
                        old_layout.offset,
                    )
                    return Buffer(node.get_name(), new_layout)

                new_X = clone_with_transposed_stride(X)
                new_W = clone_with_transposed_stride(W)
                new_Bias = clone_with_transposed_stride(Bias)
                new_Y = clone_with_transposed_stride(Y)
                options["X"], options["W"], options["Bias"], options["Y"] = (
                    new_W,
                    new_X,
                    new_Bias,
                    new_Y,
                )
                options["M"], options["N"] = "N", "M"

            epilogue_arguments = self._template_from_string(epilogue_template).render(
                **options
            )
            arguments = self._template_from_string(argument_template).render(
                epilogue_arguments=epilogue_arguments, **options
            )
        else:
            arguments = self._template_from_string(GEMM_ARGS_CUTLASS_2X).render(
                split_k=1, **options
            )
        return arguments

    def render(  # type: ignore[override]
        self,
        kernel: CUDATemplateKernel,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
        template_buffer_node: Optional[CUDATemplateBuffer] = None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        **kwargs,
    ) -> str:
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            assert self.can_fuse_epilogue and CUTLASSGemmTemplate.supports_evt(
                op
            ), "op does not support EVT epilogue fusion"
            assert (
                template_buffer_node is not None
            ), "Template node is required for epilogue fusion"
            assert isinstance(
                template_buffer_node, CUDATemplateBuffer
            ), f"Template node has to be a CUDATemplateBuffer, is type {type(template_buffer_node)}"
            assert (
                template_buffer_node.name is not None
            ), "Output node has to be a Buffer with a name"
            # This is the name of the output of the Matmul, before epilogues are applied.
            # it is not necessarily materialized in global memory if we have an epilogue

        template_output_node_name = (
            template_buffer_node.name if template_buffer_node is not None else None
        )

        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        assert isinstance(
            op, cutlass_gemm_op.GemmOperation
        ), "op argument is required and has to be an instance of GemmOperation"
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])

        assert len(self.input_nodes) >= 2 and self.output_node is not None
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        epilogue_template: Optional[str] = None
        should_swap_xw: bool = False
        epilogue_args = f"{{ElementComputeEpilogue({self.alpha}), ElementComputeEpilogue({self.beta})}}"
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            if Bias is not None and self.has_tma_epilogue(op):
                if self.should_swap_XW(Bias, self.beta):
                    # TMA epilogue requires bias vector in column major to get best perf.
                    op = self.swap_XW(op)
                    should_swap_xw = True
            if epilogue_nodes is not None and len(epilogue_nodes) > 0:
                epilogue_args = (
                    CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(
                        cast(str, template_output_node_name), epilogue_nodes
                    )
                )
            epilogue_template = GEMM_ARGS_CUTLASS_3X_EPILOGUE
            argument_template = GEMM_ARGS_CUTLASS_3X
        else:
            # TODO: Support split_k.
            argument_template = GEMM_ARGS_CUTLASS_2X

        instance_definition, instance_type = self.define_gemm_instance(
            op, cast(str, template_output_node_name), epilogue_nodes
        )
        options = dict(
            alpha=self.alpha,
            beta=self.beta,
            X=X,
            W=W,
            Y=Y,
            Bias=Bias,
            epilogue_template=epilogue_template,
            argument_template=argument_template,
            should_swap_xw=should_swap_xw,
            template=self,
            kernel=kernel,
            instance_definition=instance_definition,
            instance_type=instance_type,
            input_reorder=self.input_reorder,
            epilogue_args=epilogue_args,
        )
        res = self._template_from_string(GEMM_TEMPLATE).render(**options)
        return res


@dataclass
class CKGemmOperation:
    a_layout: str
    b_layout: str
    c_layout: str

    a_element_dtype: str
    b_element_dtype: str
    c_element_dtype: str

    acc_dtype: str
    c_shuffle_dtype: str

    a_elementwise_op: str
    b_elementwise_op: str
    c_elementwise_op: str

    gemm_specialization: str

    block_size: int

    m_per_block: int
    n_per_block: int
    k_per_block: int

    a_k1: int
    b_k1: int

    m_per_xdl: int
    n_per_xdl: int

    m_xdl_per_wave: int
    n_xdl_per_wave: int

    a_block_transfer_thread_cluster_lengths_ak0_m_ak1: Tuple[
        int, int, int
    ]  # or sequence[int]?
    a_block_transfer_thread_cluster_arrange_order: Tuple[
        int, int, int
    ]  # or sequence[int]?
    a_block_transfer_src_access_order: Tuple[int, int, int]  # or sequence[int]?

    a_block_transfer_src_vector_dim: int
    a_block_transfer_src_scalar_per_vector: int
    a_block_transfer_dst_scalar_per_vector_ak1: int
    a_block_lds_extra_m: bool

    b_block_transfer_thread_cluster_lengths_bk0_n_bk1: Tuple[
        int, int, int
    ]  # or sequence[int]?
    b_block_transfer_thread_cluster_arrange_order: Tuple[
        int, int, int
    ]  # or sequence[int]?
    b_block_transfer_src_access_order: Tuple[int, int, int]  # or sequence[int]?

    b_block_transfer_src_vector_dim: int
    b_block_transfer_src_scalar_per_vector: int
    b_block_transfer_dst_scalar_per_vector_bk1: int
    b_block_lds_extra_n: bool

    c_shuffle_m_xdl_per_wave_per_shuffle: int
    c_shuffle_n_xdl_per_wave_per_shuffle: int

    c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block: (
        Tuple[int, int, int, int]
    )
    c_shuffle_block_transfer_scalar_per_vector_n_per_block: int

    block_gemm_pipeline_scheduler: Optional[str]
    block_gemm_pipeline_version: Optional[str]

    a_compute_dtype: Optional[str]
    b_compute_dtype: Optional[str]

    def name(self):
        # cpp alias for template instance
        return f"ck_devicegemm_xdl_shuffle_v3_{self.key_name()}"

    def key_name(self):
        # TBD; must be unique per instance. Intended to use as dict key
        return f"{'_'.join(['K' + f.name.replace('_', '').lower() + 'V' + ('x'.join(map(str, iter(getattr(self, f.name)))) if isinstance(getattr(self, f.name), tuple) else str(getattr(self, f.name)).replace(':', '')) for f in fields(self)])}"


class CKGemmTemplate(CKTemplate):
    gemm_template = r"""
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    {{kernel_definition}} {
        auto gemm = {{instance_type}} {};
        auto invoker = gemm.MakeInvoker();

        constexpr auto M = {{M}};
        constexpr auto N = {{N}};
        constexpr auto K = {{K}};
        constexpr auto StrideA = std::is_same_v<{{a_layout}}, Row> ? K : M; 
        constexpr auto StrideB = std::is_same_v<{{b_layout}}, Row> ? N : K; 
        constexpr auto StrideC = std::is_same_v<{{c_layout}}, Row> ? N : M;  
        constexpr auto KBatch = 1; // split k into batches

        auto argument = gemm.MakeArgument(
            reinterpret_cast<const {{a_element_dtype}}*>(X),
            reinterpret_cast<const {{b_element_dtype}}*>(W),
            reinterpret_cast<{{c_element_dtype}}*>(Y),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            KBatch,
            {{a_elementwise_op}} {},
            {{b_elementwise_op}} {},
            {{c_elementwise_op}} {}  
        );
        if (!gemm.IsSupportedArgument(argument)) {
            // magic spell to assign +inf to benchmarking time in select_algorithm.py:1052 (1/2)
            std::cerr << "invalid argument for gemm instance " << gemm.GetTypeString() << std::endl; 
            argument.Print();
            return -23;
        }
        // run the kernel
        float elapsed_time = invoker.Run(argument, StreamConfig{stream, /* time kernel */ false, /* log level */ kDEBUG_LOG});
        return 0;
    } // kernel definition
    } // extern C
    """

    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__(
            "ck_gemm_template",
            input_nodes=input_nodes,
            layout=layout,
            input_reorder=input_reorder,
        )
        self.alpha = alpha
        self.beta = beta

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK GEMM header(s)

                #include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK GEMM globals

                using Row = ck::tensor_layout::gemm::RowMajor;
                using Col = ck::tensor_layout::gemm::ColumnMajor;

                using cudaStream_t = hipStream_t; 

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;
            """
        )
        return res

    def filter_op(self, op: CKGemmOperation) -> Optional[CKGemmOperation]:
        # TBD return None if alignment or layout or dtype is invalid
        def torch_layout_to_ck_layout(torch_layout):
            if torch_layout.stride[-1] == 1:
                return "Row"
            elif torch_layout.stride[-2] == 1:
                return "Col"
            else:
                return None

        X_meta, W_meta, Y_meta = map(lambda T: T.get_layout(), [*self.input_nodes, self.output_node])
        # disable the instance if dtypes don't match
        if op.a_element_dtype != self._TORCH_DTYPE_TO_CK[X_meta.dtype]:
            return None
        if op.b_element_dtype != self._TORCH_DTYPE_TO_CK[W_meta.dtype]:
            return None
        if op.c_element_dtype != self._TORCH_DTYPE_TO_CK[Y_meta.dtype]:
            return None
        # disable the instance if layouts don't match
        if op.a_layout != torch_layout_to_ck_layout(X_meta):
            return None
        if op.b_layout != torch_layout_to_ck_layout(W_meta):
            return None
        if op.c_layout != torch_layout_to_ck_layout(Y_meta):
            return None
        # try to avoid launching the instance with invalid problem size

        M = X_meta.size[-2]
        K = X_meta.size[-1]
        N = W_meta.size[-1]

        if not any(m_padding in op.gemm_specialization for m_padding in ["MPadding", "MNPadding", "MKPadding", "MNKPadding"]):
            if M % op.m_per_block != 0:
                return None
        if not any(n_padding in op.gemm_specialization for n_padding in ["NPadding", "MNPadding", "NKPadding", "MNKPadding"]):
            if N % op.n_per_block != 0:
                return None
        if not any(k_padding in op.gemm_specialization for k_padding in ["KPadding", "MKPadding", "NKPadding", "MNKPadding"]):
            if K % op.k_per_block != 0:
                return None

        if (K if op.a_layout == "Row" else M) % op.a_block_transfer_src_scalar_per_vector != 0:
            return None
        if (N if op.b_layout == "Row" else K) % op.b_block_transfer_src_scalar_per_vector != 0:
            return None
        if (N if op.c_layout == "Row" else M) % op.c_shuffle_block_transfer_scalar_per_vector_n_per_block != 0:
            return None

        return op

    def emit_ck_instance(self, op: CKGemmOperation):
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} = 
        ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
            {{template_params}}>;

"""
        template_type = r"""
    Operation_{{operation_name}}
"""
        template_params = []
        for f in fields(op):
            field_value = getattr(op, f.name)
            if isinstance(field_value, tuple):
                template_params.append(
                    f"/* {f.name} */ S<{', '.join(map(str, iter(field_value)))}>"
                )
            else:
                if field_value is not None:
                    template_params.append(f"/* {f.name} */ {field_value}")
        return self._template_from_string(template_definition).render(
            operation_name=op.name(),
            template_params=(",\n" + 12 * " ").join(template_params),
        ), self._template_from_string(template_type).render(operation_name=op.name())

    def render(self, kernel: CUDATemplateKernel, op: CKGemmOperation, **kwargs):
        epilogue_nodes = kwargs.get("epilogue_nodes", None)
        assert epilogue_nodes is None or 0 == len(epilogue_nodes)
        template_buffer_node = kwargs.get("template_buffer_node", None)
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        instance_definition, instance_type = self.emit_ck_instance(op)
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None  # TBD support gemm_bias

        return self._template_from_string(self.gemm_template).render(
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W, Bias],
                outputs=[Y],
                names_str="X, W, Bias, Y",
                input_reorder=self.input_reorder,
            ),
            instance_type=instance_type,
            M=kernel.size(X, -2),
            K=kernel.size(X, -1),
            N=kernel.size(W, -1),
            a_elementwise_op=op.a_elementwise_op,
            b_elementwise_op=op.b_elementwise_op,
            c_elementwise_op=op.c_elementwise_op,
            a_element_dtype=op.a_element_dtype,
            b_element_dtype=op.b_element_dtype,
            c_element_dtype=op.c_element_dtype,
            a_layout=op.a_layout,
            b_layout=op.b_layout,
            c_layout=op.c_layout,
        )

    def gen_ops(self):
        op_instances = []
        # all string attributes must be either type aliases or global constants in C++
        # fallback: known working op instance for problem size M=2240 K=256 N=2048
        default_instances = [CKGemmOperation(
            a_layout="Row",
            b_layout="Row",
            c_layout="Row",
            a_element_dtype="F16",
            b_element_dtype="F16",
            c_element_dtype="F16",
            a_compute_dtype="F16",
            b_compute_dtype="F16",
            acc_dtype="F32",
            c_shuffle_dtype="F16",
            a_elementwise_op="PassThrough",
            b_elementwise_op="PassThrough",
            c_elementwise_op="PassThrough",
            gemm_specialization="GemmSpecialization::Default",
            block_size=256,
            m_per_block=224,
            n_per_block=256,
            k_per_block=64,
            a_k1=8,
            b_k1=2,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=7,
            n_xdl_per_wave=8,
            a_block_transfer_thread_cluster_lengths_ak0_m_ak1=(8, 32, 1),
            a_block_transfer_thread_cluster_arrange_order=(1, 0, 2),
            a_block_transfer_src_access_order=(1, 0, 2),
            a_block_transfer_src_vector_dim=2,
            a_block_transfer_src_scalar_per_vector=8,
            a_block_transfer_dst_scalar_per_vector_ak1=8,
            a_block_lds_extra_m=0,
            b_block_transfer_thread_cluster_lengths_bk0_n_bk1=(8, 32, 1),
            b_block_transfer_thread_cluster_arrange_order=(0, 2, 1),
            b_block_transfer_src_access_order=(0, 2, 1),
            b_block_transfer_src_vector_dim=1,
            b_block_transfer_src_scalar_per_vector=8,
            b_block_transfer_dst_scalar_per_vector_bk1=2,
            b_block_lds_extra_n=0,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=2,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(1, 32, 1, 8),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v3")]

        grep_result = subprocess.run(
            [
                "grep",
                "-inR",
                "DeviceGemm_Xdl_CShuffleV3",
                os.path.join(config.rocm.ck_dir, "library"),
            ],
            capture_output=True,
            text=True,
        )

        def maybe_int(s):
            try:
                return int(s)
            except ValueError:
                return s

        for line in grep_result.stdout.strip().split("\n"):
            s_template_args = line.split("DeviceGemm_Xdl_CShuffleV3")[-1].strip("<>, ")
            template_args = []
            i_current = 0
            while i_current < len(s_template_args):
                if s_template_args[i_current] == " ":
                    i_current += 1
                    continue
                elif s_template_args[i_current : i_current + 2] == "S<":
                    i_next = s_template_args.find(">", i_current)
                    template_args.append(
                        tuple(
                            map(int, s_template_args[i_current + 2 : i_next].split(","))
                        )
                    )
                    i_current = i_next + 2
                else:
                    i_next = s_template_args.find(",", i_current)
                    template_args.append(
                        maybe_int(
                            s_template_args[
                                i_current : i_next if i_next != -1 else None
                            ]
                        )
                    )
                    if i_next != -1:
                        i_current = i_next + 1
                if i_next == -1:
                    break
            new_instance = CKGemmOperation(
                *template_args,
                *((None,) * (len(fields(CKGemmOperation)) - len(template_args))),
            )
            if new_instance.a_compute_dtype is None:
                new_instance.a_compute_dtype = new_instance.c_element_dtype
            if new_instance.b_compute_dtype is None:
                new_instance.b_compute_dtype = new_instance.c_element_dtype

            schedulers = [
                "BlockGemmPipelineScheduler::Intrawave",
                "BlockGemmPipelineScheduler::Interwave",
            ]
            gemm_specs = [
                "GemmSpecialization::Default",
                "GemmSpecialization::MPadding",
                "GemmSpecialization::NPadding",
                "GemmSpecialization::KPadding",
                "GemmSpecialization::MNPadding",
                "GemmSpecialization::MKPadding",
                "GemmSpecialization::NKPadding",
                "GemmSpecialization::MNKPadding",
            ]
            if new_instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched":
                for scheduler in schedulers:
                    for spec in gemm_specs:
                        op_instances.append(
                            dataclasses.replace(
                                new_instance,
                                block_gemm_pipeline_scheduler=scheduler,
                                gemm_specialization=spec,
                            )
                        )
            else:
                for spec in gemm_specs:
                    op_instances.append(
                        dataclasses.replace(new_instance, gemm_specialization=spec)
                    )

        filtered_instances = list(filter(lambda op: self.filter_op(op), op_instances))
        # chosen_instances = filtered_instances[:config.rocm.n_max_profiling_configs]
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = random.sample(filtered_instances, config.rocm.n_max_profiling_configs)
        log.debug(f"generated {len(chosen_instances)} ck instances: {chosen_instances}")

        return chosen_instances

    @staticmethod
    def add_ck_gemm_choices(
        choices,
        layout,
        input_nodes,
        alpha=1,
        beta=0,
        input_reorder=None,
    ):
        template = CKGemmTemplate(
            input_nodes,
            layout,
            alpha=alpha,
            beta=beta,
            input_reorder=input_reorder,
        )
        ops = template.gen_ops()
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )
