import copy
import logging
import re
from typing import cast, Dict, List, Optional, Tuple, Union

from ... import ir
from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, ChoiceCaller, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer

from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
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
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[ir.IRNode],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        """
        Adds Cutlass GEMM configurations choices to the auto-tuning list.

        This function mutates the passed list of choices by appending the choices for Cutlass GEMM configs to it.

        Args:
            choices (list): The list to which choices are appended.
            layout (ir.Layout): The layout configuration.
            input_nodes (list): The list of input nodes.
            alpha (float,int): Scaling factor, defaults to 1.
            beta (float,int): Offset, defaults to 0.
            input_reorder (list, optional): Order of the inputs, defaults to None.
            **extra_kwargs: Additional keyword arguments.

        """

        cutlass_template = CUTLASSGemmTemplate(
            input_nodes,  # type: ignore[arg-type]
            layout,
            alpha=alpha,
            beta=beta,
            input_reorder=input_reorder,
        )
        ops = cutlass_template.gen_ops()
        for op in ops:
            cutlass_template.maybe_append_choice(
                choices,
                op=op,
            )
        if len(ops) == 0:
            input_layouts = [node.get_layout() for node in input_nodes]
            input_strides = [node.get_stride() for node in input_nodes]
            output_layout = layout
            warning_msg = f"No suitable Cutlass GEMM configs found, fallbacks used ( {len(ops)=}, {output_layout=}, {input_layouts=}, {input_strides=} )"  # noqa: B950
            log.warning(warning_msg)
        log.debug(
            "Added %d Cutlass gemm configs.",
            len(ops),
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
