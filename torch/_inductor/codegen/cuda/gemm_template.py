import copy
import logging
import re
from typing import List, Optional

import torch

# import cutlass libs
import gemm_operation as cutlass_gemm_op
import library as cutlass_lib

from ...ir import Buffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer

from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate

log = logging.getLogger(__name__)


# Only supports alpha * A@B + beta * C now.
# TODO: Support arbitrary epilogue after epilogue visitor is released in cutlass 3.2.
GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
{{instance_definition}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, compuates the Gemm kernel using the given workspace ptr.
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
  {{template.render_gemm_arguments(argument_template, epilogue_template, should_swap_xw, X, W, Bias, Y, alpha, beta, kernel)}}
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
      { {{template.cute_int(kernel.stride(X, -2), "stride_x0")}}, {{template.cute_int(kernel.stride(X, -1), "stride_x1")}}, {{template.cute_int(kernel.stride(X, -3), "batch_stride_x")}}},  // StrideA dA
      {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // ElementB const* ptr_B
      { {{template.cute_int(kernel.stride(W, -1), "stride_w1")}}, {{template.cute_int(kernel.stride(W, -2), "stride_w0")}}, {{template.cute_int(kernel.stride(W, -3), "batch_stride_w")}}},  // StrideB dB
    },  // MainloopArguments mainloop
    {{epilogue_arguments}}
  };
"""


GEMM_ARGS_CUTLASS_3X_EPILOGUE = r"""
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // ElementC const* ptr_C
      { {{template.cute_int(kernel.stride(Bias, -2, 1), "stride_bias0")}}, {{template.cute_int(kernel.stride(Bias, -1, 1), "stride_bias1")}}, {{template.cute_int(kernel.stride(Bias, -3), "batch_stride_bias")}}},  // StrideC dC
      {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // ElementD const* ptr_D
      { {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}}, {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}}, {{template.cute_int(kernel.stride(Y, -3), "batch_stride_y")}}},  // StrideD dD
    },  // EpilogueArguments epilogue
"""


class CUTLASSGemmTemplate(CUTLASSTemplate):
    # Calculates alpha * X@W + beta * Bias

    def __init__(
        self,
        input_nodes: List[IRNode],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: List[int]=None,
    ):
        super().__init__("cutlass_gemm", input_nodes, layout, input_reorder)
        self.alpha = alpha
        self.beta = beta

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
            """
        )
        return res

    @staticmethod
    def cutlass_layout(torch_layout) -> Optional[cutlass_lib.LayoutType]:
        if torch_layout.stride[-1] == 1:
            return cutlass_lib.LayoutType.RowMajor
        elif torch_layout.stride[-2] == 1:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: cutlass_lib.LayoutType,
    ) -> cutlass_lib.LayoutType:
        if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return cutlass_lib.LayoutType.RowMajor

    @staticmethod
    def layout_match(torch_layout, cutlass_layout) -> bool:
        return CUTLASSGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout

    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        alignment = cutlass_utils.get_alignment(torch_layout)
        if alignment < op_element.alignment:
            return False
        else:
            op_element.alignment = alignment
            return True

    @staticmethod
    def has_tma_epilogue(op) -> bool:
        result = False
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result

    @staticmethod
    def define_gemm_instance(
        op: cutlass_gemm_op.GemmOperation,
    ) -> str:
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            emitter = cutlass_gemm_op.EmitGemmUniversal3xInstance()
            op_def = emitter.emit(op)
            pattern = re.compile(r"\s*struct\s(.*?)\s:")
            decl = [line for line in op_def.split("\n") if "struct " in line][-1]
        else:
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
    def swap_XW(op: cutlass_gemm_op.GemmOperation) -> cutlass_gemm_op.GemmOperation:
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
        op: cutlass_gemm_op.GemmOperation,
    ) -> cutlass_gemm_op.GemmOperation:
        # Skip simt kernels
        if (
            op.tile_description.math_instruction.opcode_class
            == cutlass_lib.OpcodeClass.Simt
        ):
            return None

        # Skip sparse and grouped kernels
        if op.gemm_kind in {
            cutlass_lib.GemmKind.Sparse,
            cutlass_lib.GemmKind.Grouped,
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
            and cutlass_utils.dtype_match(self.output_node.get_layout().dtype, op.C.element)
            and cutlass_utils.dtype_match(accumulator_torch_dtype, op.accumulator_type())
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

        return op

    def gen_ops(self) -> List[cutlass_gemm_op.GemmOperation]:
        ops = cutlass_utils.gen_ops()[cutlass_lib.OperationKind.Gemm]
        res = set()
        num_3x_ops = 0
        num_2x_ops = 0
        for key, op_list in ops.items():
            for op in op_list:
                filter_res = self.filter_op(op)
                if filter_res is not None:
                    res.add(filter_res)
        for op in res:
            if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
                num_3x_ops += 1
            else:
                num_2x_ops += 1
        log.debug(f"Got cutlass configs: {len(res)=}, {num_3x_ops=}, {num_2x_ops=}")
        return list(res)

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
                options['X'], options['W'], options['Bias'], options['Y'] = new_W, new_X, new_Bias, new_Y
                options['M'], options['N'] = "N", "M"

            epilogue_arguments = self._template_from_string(epilogue_template).render(**options)
            arguments = self._template_from_string(argument_template).render(
                epilogue_arguments=epilogue_arguments, **options
            )
        else:
            arguments = self._template_from_string(GEMM_ARGS_CUTLASS_2X).render(
                split_k=1, **options
            )
        return arguments

    def render(
        self,
        kernel: CUDATemplateKernel,
        op: cutlass_gemm_op.GemmOperation,
        output_node: IRNode = None,
    ) -> str:
        if output_node is not None:
            self.output_node = output_node
        assert len(self.input_nodes) >= 2 and self.output_node is not None
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 or self.input_nodes[2] is None else self.input_nodes[2]

        epilogue_template: Optional[str] = None
        argument_template: Optional[str] = None
        should_swap_xw: bool = False

        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            if Bias is not None and self.has_tma_epilogue(op):
                if self.should_swap_XW(Bias, self.beta):
                    # TMA epilogue requires bias vector in column major to get best perf.
                    op = self.swap_XW(op)
                    should_swap_xw = True
            epilogue_template = GEMM_ARGS_CUTLASS_3X_EPILOGUE
            argument_template = GEMM_ARGS_CUTLASS_3X
        else:
            # TODO: Support split_k.
            argument_template = GEMM_ARGS_CUTLASS_2X

        instance_definition, instance_type = self.define_gemm_instance(op)
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
        )

        res = self._template_from_string(GEMM_TEMPLATE).render(**options)
        return res
