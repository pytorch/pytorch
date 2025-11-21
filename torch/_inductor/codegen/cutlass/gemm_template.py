# mypy: allow-untyped-defs
import copy
import enum
import functools
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.codegen.cutlass.cache import maybe_fetch_ops
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.select_algorithm import create_inputs_key
from torch._inductor.utils import clear_on_fresh_cache

from ... import ir
from ...config import cutlass as inductor_cutlass_config
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUDATemplateBuffer,
    FixedLayout,
    IRNode,
    Layout,
    ReinterpretView,
)
from ...utils import is_dynamic, Placeholder
from ...virtualized import V
from ..common import IndentedBuffer
from . import utils as cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from .python_evt import CutlassEVTCodegen, scaled_mm_evt
from .utils import (
    ACCUMULATOR_DTYPES,
    dtype_match,
    torch_dtype_to_cutlass_type,
    XW_DTYPES,
)


GemmOperation = Any
EVTArgRenames = Any

log = logging.getLogger(__name__)

# Jinja template for GEMM Kernel, used by the CUTLASSGemm3xTemplate class below.
GEMM_TEMPLATE_CUTLASS_3X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
{{epilogue_visitor_tree}}
{{instance_definition}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.
extern "C" {
PT_EXPORT {{kernel_call_signature}} {
  try {
  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  static cutlass::KernelHardwareInfo hw_info;
  if (hw_info.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
  }
  {{instance_type}}::Arguments arguments;
  {{template.render_gemm_arguments(argument_template, epilogue_template, should_swap_xw,
                                    X, W, Bias, Y, alpha, beta, kernel, epilogue_args)}}
  {{instance_type}} gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }
  // check for null pointers after workspace size, since querying workspace size doesn't require valid data pointers
#ifndef CUTLASS_BACKEND_DISABLE_CHECKS
  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
#endif
#ifdef CUTLASS_DEBUG_TRACE_LEVEL
#if CUTLASS_DEBUG_TRACE_LEVEL == 1
  {
    // Print the maximum number of active blocks per SM for the kernel if CUTLASS_DEBUG_TRACE_LEVEL == 1
    // we don't need a print statement, it's happening inside the function.
    gemm_op.maximum_active_blocks();
  }
#endif
#endif
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

// configuration name: {{op_conf_name}}
"""

# Jinja template for Cutlass 3.x GEMM Kernel arguments, used by the CUTLASSGemmTemplate class below.
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
        {{template.cute_int(kernel.batch_stride(X), "batch_stride_x")}}
      },  // StrideA dA
      {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // ElementB const* ptr_B
      {
        {{template.cute_int(kernel.stride(W, -1), "stride_w1")}},
        {{template.cute_int(kernel.stride(W, -2), "stride_w0")}},
        {{template.cute_int(kernel.batch_stride(W), "batch_stride_w")}}
      },  // StrideB dB
    },  // MainloopArguments mainloop
    {{epilogue_arguments}},
    hw_info
  };
  arguments.scheduler.max_swizzle_size = swizzle;
"""

# Jinja template for Cutlass 3.x GEMM Kernel arguments if epilogue fusion is applied,
# used by the CUTLASSGemmTemplate class below.
GEMM_ARGS_CUTLASS_3X_EPILOGUE = r"""
    // see https://tinyurl.com/4rk89z48
    {
      {{epilogue_args}},  // thread, typename FusionCallbacks::Arguments ( EVT ) or ThreadEpilogueOp::Params (non-EVT )
      {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // ElementC const* ptr_C
      {
        {{template.cute_int(kernel.stride(Bias, -2, 1), "stride_bias0")}},
        {{template.cute_int(kernel.stride(Bias, -1, 1), "stride_bias1")}},
        {{template.cute_int(kernel.batch_stride(Bias), "batch_stride_bias")}}
      },  // StrideC dC
      {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // ElementD const* ptr_D
      {
        {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}},
        {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}},
        {{template.cute_int(kernel.batch_stride(Y), "batch_stride_y")}}
      },  // StrideD dD
    },  // EpilogueArguments epilogue
"""

# Jinja template for GEMM Kernel, used by the CUTLASS2xGemmTemplate class below.
GEMM_TEMPLATE_CUTLASS_2X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
{{instance_definition}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.
extern "C" {
PT_EXPORT {{kernel_call_signature}} {
  try {
  int B = {{kernel.size(Y, 0, -3, default_value=1)}};
  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  static cutlass::KernelHardwareInfo hw_info;
  if (hw_info.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
  }
  {{instance_type}}::Arguments arguments;
  {{template.render_gemm_arguments(instance_type, argument_template, epilogue_template, should_swap_xw,
                                    X, W, Bias, Meta, Y, alpha, beta, kernel, epilogue_args)}}
  {{instance_type}} gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }

  // check for null pointers after workspace size, since querying workspace size doesn't require valid data pointers
#ifndef CUTLASS_BACKEND_DISABLE_CHECKS
  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
#endif
#ifdef CUTLASS_DEBUG_TRACE_LEVEL
#if CUTLASS_DEBUG_TRACE_LEVEL == 1
  {
    // Print the maximum number of active blocks per SM for the kernel if CUTLASS_DEBUG_TRACE_LEVEL == 1
    // we don't need a print statement, it's happening inside the function.
    gemm_op.maximum_active_blocks();
  }
#endif
#endif

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

# Jinja template for Cutlass 2.x GEMM Kernel arguments, used by the CUTLASS2xGemmTemplate class below.
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

GEMM_ARGS_SPARSE_CUTLASS_2X = r"""
  using TensorRefA = cutlass::TensorRef<{{instance_type}}::ElementA,
                                        {{instance_type}}::LayoutA>;
  using TensorRefB = cutlass::TensorRef<{{instance_type}}::ElementB,
                                        {{instance_type}}::LayoutB>;
  using TensorRefC = cutlass::TensorRef<{{instance_type}}::ElementC,
                                        {{instance_type}}::LayoutC>;
  using TensorRefE = cutlass::TensorRef<{{instance_type}}::ElementE,
                                        {{instance_type}}::LayoutE>;
  // Note that "X" and "W" names may be misleading here.  Namely, for
  // sparse GEMM, the first argument is always sparse, while typically
  // weight matrix, implied by name "W" will be sparse in
  // applications.  Thus, just remember that here: "X" refers to first
  // argument, that is sparse, and "W" to second, that is dense.
  TensorRefA X_ref({{template.cutlass_type_cast(X, kernel.ptr(X))}}, {{kernel.row_or_column_stride(X)}});
  TensorRefB W_ref({{template.cutlass_type_cast(W, kernel.ptr(W))}}, {{kernel.row_or_column_stride(W)}});
  TensorRefC Y_ref({{template.cutlass_type_cast(Y, kernel.ptr(Y))}}, {{kernel.row_or_column_stride(Y)}});
  TensorRefE Meta_ref({{template.cutlass_sparse_meta_type_cast(Meta, kernel.ptr(Meta))}},
                      TensorRefE::Layout::packed({ {{kernel.size(Meta, 0)}}, {{kernel.size(Meta, 1)}} }));
  // Initialize GemmSparse arguments.
  arguments = {
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(2 * K),
    },  // GemmCoord problem_size
    X_ref,  // TensorRef<ElementA const, LayoutA> ref_A
    W_ref,  // TensorRef<ElementB const, LayoutB> ref_B
    Y_ref,  // TensorRef<ElementC const, LayoutC> ref_C
    Y_ref,  // TensorRef<ElementC, LayoutC> ref_D
    Meta_ref,  // TensorRef<ElementE const, LayoutE> ref_E
    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue,
  };
"""

# Additional includes which are necessary if the standalone test / debug runner is generated as well
GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES = r"""
#ifdef GENERATE_STANDALONE_RUNNER
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include <iostream>
#endif
"""

# Jinja template for the standalone runner that may be generated as part of the code.
GEMM_STANDALONE_RUNNER_TEMPLATE = r"""
#ifdef GENERATE_STANDALONE_RUNNER
/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed, float max=1.0, float min=-1.0) {
  if (block.size()<=0) return false;
  Element scope_max(static_cast<Element>(max)), scope_min(static_cast<Element>(min));
  cutlass::reference::device::BlockFillRandomUniform(
    (Element*)block.get(), block.size(), seed, scope_max, scope_min);

  return true;
}

{% if Meta is defined and Meta is not none %}
template <class Element>
bool initialize_block_meta(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed) {
  if (block.size()<=0) return false;
  cutlass::reference::device::BlockFillRandomSparseMeta(
    (Element*)block.get(), block.size(), seed, {{instance_type}}::kMetaSizeInBits);
  return true;
}
{% endif %}

extern "C" int run_standalone(uint64_t seed, int repetitions) {
    std::cout << "Starting GEMM Standalone test run with seed " << seed << std::endl;
    size_t workspace_size = 0;
    size_t* workspace_size_ptr = &workspace_size;

    int M = {{kernel.get_layout_args()[0]}};
    int N = {{kernel.get_layout_args()[1]}};
    int K = {{kernel.get_layout_args()[2]}};
    int B = {{kernel.get_layout_args()[3]}};
    int lda = {{kernel.get_layout_args()[4]}};
    int ldb = {{kernel.get_layout_args()[5]}};
    int ldc = {{kernel.get_layout_args()[6]}};
    int ldd = {{kernel.get_layout_args()[7]}};
    uint8_t swizzle = {{kernel.runtime_arg_values[0]}};

    using ElementA = {{kernel.cutlass_dtype(X)}};
    using ElementB = {{kernel.cutlass_dtype(W)}};
    using ElementC = {{kernel.cutlass_dtype(Bias, default_dtype='uint8_t')}}; // may not be void
    using ElementD = {{kernel.cutlass_dtype(Y)}};
    {% if Meta is defined and Meta is not none %}
    using ElementE = {{kernel.cutlass_dtype(Meta)}};
    {% endif %}

    cutlass::DeviceAllocation<ElementA> X_data({{kernel.max_valid_index(X)+1}});
    initialize_block(X_data, seed++);
    cutlass::DeviceAllocation<ElementB> W_data({{kernel.max_valid_index(W)+1}});
    initialize_block(W_data, seed++);
    cutlass::DeviceAllocation<ElementC> Bias_data({{kernel.max_valid_index(Bias)+1}});
    initialize_block(Bias_data, seed++);
    cutlass::DeviceAllocation<ElementD> Y_data({{kernel.max_valid_index(Y)+1}});
    {% if Meta is defined and Meta is not none %}
    cutlass::DeviceAllocation<ElementE> Meta_data({{kernel.max_valid_index(Meta)+1}});
    initialize_block_meta(Meta_data, seed++);
    {% endif %}

    cutlass::DeviceAllocation<uint8_t> workspace_data;
    // Call once with workspace_size_ptr set to get workspace size

    std::cout << "Calling once to get workspace size" << std::endl;
    {{test_call_statement}};
    // Allocate workspace if necessary
    if (workspace_size > 0) {
        workspace_data.reset(workspace_size);
        std::cout << "Allocated workspace size of " << workspace_size << " bytes" << std::endl;
    }
    std::cout << "Calling Kernel as {{test_call_statement}};" << std::endl;
    workspace_size_ptr = nullptr;
    for (int i=0; i<repetitions; i++) {
        {{test_call_statement}};
    }
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device synchronize failed with error "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    return 0;
}

int main(int argc, char** argv) {
    // warmup
    run_standalone(1, 2);
    // repeat
    return run_standalone(2, 10);
}

#endif
"""  # noqa: B950


@clear_on_fresh_cache
class CUTLASSGemmTemplate(CUTLASSTemplate, ABC):
    """
    CUTLASS GEMM Template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    filtered_ops_cache: dict[str, list[Any]] = {}
    cache_clear = staticmethod(filtered_ops_cache.clear)

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[list[int]] = None,
        use_fast_accum: Optional[bool] = None,
    ) -> None:
        """
        Args:
            input_nodes (List[Buffer]): List of input nodes of the GEMM kernel.
            layout (Layout): Layout type of the resulting output node.
            alpha (float): The scaling factor for the product of the inputs in the GEMM operation.
            beta (float): The scaling factor applied to the output matrix.
            input_reorder (Optional[List[int]]): Specifies the reordering of the input nodes. If not provided,
                            no reordering is performed. Defaults to None.
        """
        super().__init__(
            str(Placeholder.KERNEL_NAME), input_nodes, layout, input_reorder
        )
        self.alpha = alpha
        self.beta = beta
        self.use_fast_accum = use_fast_accum
        assert 2 <= len(input_nodes) <= 5
        assert self._are_inputs_layout_compatible(
            [node.get_layout() for node in input_nodes]
        )

        self.cache_key: str = create_inputs_key(self.input_nodes)

    @staticmethod
    @abstractmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[list[int]] = None,
        use_fast_accum: Optional[bool] = None,
        **extra_kwargs,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_supported_ops() -> "list[cutlass_library.gemm_operation.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _has_tma_epilogue(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_template(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_template_args(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[str, Optional[str]]:
        raise NotImplementedError

    @abstractmethod
    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _shape_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _alignment_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _set_bias_layout_and_alignment(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _define_gemm_instance(
        self,
        op: GemmOperation,
        evt_name: Optional[str] = None,
    ) -> tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def _get_extra_inputs_and_names(
        self,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[Optional[Buffer], list[Optional[Buffer]], list[str]]:
        raise NotImplementedError

    @abstractmethod
    def _update_arg_names_for_test_call_statement(
        self,
        arg_names: list[str],
        input_nodes: list[Buffer],
    ) -> list[str]:
        raise NotImplementedError

    def _add_cutlass_gemm_choices(
        self,
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[list[int]] = None,
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

        ops = self.gen_ops()

        # pre-computation
        layout_repr: str = str(layout)
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]] = (
            TensorMeta.from_irnodes(self.input_nodes)
        )
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]] = (
            TensorMeta.from_irnodes(self.output_node)
        )

        with dynamo_timed("CUTLASSGemmTemplate.maybe_append_choice"):
            for name, op in ops:
                for (
                    swizzle
                ) in inductor_cutlass_config.cutlass_max_profiling_swizzle_options:
                    description = f"{name} swizzle={swizzle}"
                    self.maybe_append_choice(
                        choices,
                        op=op,
                        name=name,
                        description=description,
                        input_key=self.cache_key,
                        layout_repr=layout_repr,
                        input_tensor_meta=input_tensor_meta,
                        output_tensor_meta=output_tensor_meta,
                        swizzle=swizzle,
                    )

        if len(ops) == 0:
            log.info(
                "No suitable Cutlass GEMM configs found, fallbacks used "
                "( len(ops)=%d, output_layout=%s, input_layouts=%s, input_strides=%s )",
                len(ops),
                layout,
                [node.get_layout() for node in input_nodes],
                [node.get_stride() for node in input_nodes],
            )
        log.debug(
            "Added %d Cutlass gemm configs.",
            len(ops),
        )

    def header(self) -> IndentedBuffer:
        """
        Returns a buffer containing CUDA C++ code for the header section of the CUTLASS GEMM template.
        This section primarily includes the necessary header files.

        Returns:
            IndentedBuffer: An instance of IndentedBuffer that contains the generated CUDA C++ header code.
        """
        res = super().header()
        res.splice(
            """
                #include "cutlass/gemm/gemm.h"
                #include "cutlass/gemm/device/gemm_universal.h"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"
                #include "cutlass/gemm/kernel/gemm_universal.hpp"
                #include "cutlass/gemm/device/gemm_sparse.h"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/default_epilogue.hpp"
                #include "cutlass/epilogue/thread/linear_combination.h"
                #include "cutlass/epilogue/thread/activation.h"
                #include "cutlass/gemm/dispatch_policy.hpp"
                #include "cutlass/gemm/kernel/tile_scheduler.hpp"
                #include "cutlass/tensor_ref.h"
                #include "cutlass/util/distribution.h"
                #include "cutlass/util/packed_stride.hpp"
                #include "cutlass/util/tensor_view_io.h"
            """
        )
        if inductor_cutlass_config.generate_test_runner and not is_dynamic(
            *self.input_nodes, self.output_node
        ):
            res.splice(GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES)
        return res

    @staticmethod
    def cutlass_layout(torch_layout: ir.Layout) -> "Optional[cutlass_lib.LayoutType]":  # type: ignore[name-defined]  # noqa: F821
        """
        Converts an ir.Layout instance into the corresponding cutlass_library.LayoutType enum value
        (RowMajor, ColumnMajor, or None if no matching value is found ).

        Args:
            torch_layout (ir.Layout): The layout that needs to be looked up.

        Returns:
            cutlass_lib.LayoutType: The converted layout corresponding to the `torch_layout` or None if no matching
            value is found.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
            return cutlass_lib.LayoutType.RowMajor
        elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-2], 1):
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: "cutlass_lib.LayoutType",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_lib.LayoutType":  # type: ignore[name-defined]  # noqa: F821
        """Helper method: Flips a given cutlass layout (cutlass_lib.LayoutType) from RowMajor
        to ColumnMajor or vice versa"""
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return cutlass_lib.LayoutType.RowMajor

    @staticmethod
    @functools.lru_cache(32)
    def layout_match(
        torch_layout: ir.Layout,
        cutlass_layout: "cutlass_lib.LayoutType",  # type: ignore[name-defined] # noqa: F821
    ) -> bool:
        """Helper Method: Determines whether a given torch layout matches a given Cutlass layout"""
        return CUTLASSGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout

    @staticmethod
    def set_layout(tensor_desc: "TensorDescription", torch_layout: ir.Layout) -> None:  # type: ignore[name-defined]  # noqa: F821
        """
        Helper method: Sets the layout of a given tensor description to match the given torch layout
        """
        if CUTLASSGemmTemplate.layout_match(torch_layout, tensor_desc.layout):
            return
        tensor_desc.layout = CUTLASSGemmTemplate.cutlass_layout(torch_layout)

    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        """
        Helper method to update the alignment of a given CUTLASS GEMM op operand's element.

        This method modifies the alignment of the given Cutlass GEMM op operand's element to match the
        layout of the corresponding ir.Buffer node.

        Args:
            torch_layout: The layout of the corresponding ir.Buffer node.
            op_element: The Cutlass GEMM op operand's element whose alignment is to be updated.

        Returns:
            bool: True if the alignment was successfully updated, False otherwise.
        """
        alignment = cutlass_utils.get_max_alignment(torch_layout)
        cuda_arch = cutlass_utils.get_cuda_arch()
        if cuda_arch and int(cuda_arch) >= 90 and alignment < op_element.alignment:
            return False
        else:
            op_element.alignment = alignment
            return True

    @staticmethod
    def should_swap_XW(
        bias: IRNode,
    ) -> bool:
        """
        Helper method to determine whether we should do an explicit transpose by switching the order of the
        matmul operands. This might be necessary when we can't otherwise arrive at the right memory
        layout for the given Bias operand.

        Note: This method is a workaround for CUDA Errors that seemingly non-deterministically
        occurred in practice in some CUTLASS GEMM Kernels with Linear epilogues that have a bias term.
        it might make sense to check on newer Cutlass releases whether it makes sense to keep
        returning True in certain cases or whether it becomes unnecessary.
        """
        # If bias is row major, swap all M and N dimensions
        if (
            bias is not None
            and len(bias.get_stride()) >= 2
            and bias.get_stride()[-1] in (0, 1)
        ):
            log.debug("GEMM Layout swapped X and W -> explicit transpose")
            return True
        return False

    @staticmethod
    def swap_XW(
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """
        Swap operands X and W (aka operans A and B) of the GEMM operation. This
        requires transposing the operands, which is done by swapping the strides.
        Note that we don't change the apparent external layout, just the operand layout.
        this is intentional.
        """
        new_op = copy.deepcopy(op)
        new_op.A.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.A.layout)
        new_op.B.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.B.layout)
        new_op.A, new_op.B = new_op.B, new_op.A
        new_op.C.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.C.layout)
        new_op.D.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.D.layout)
        return new_op

    def fix_op_layout(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined] # noqa: F821
        X: Buffer,
        W: Buffer,
        Bias: Optional[Buffer],
        Y: Union[Buffer, ReinterpretView],
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        # This is a workaround to deal with cases where the input layouts have changed
        # between autotuning and rendering. This happens if the inputs layout
        # are FlexibleLayout instances. In this case, we need to update the
        # op's input layouts. It is a hack, because now the op
        # we benchmarked is not the same as the op we render,
        # but there is no simple way to fix this in the autotuner, since that would
        # potentially disable other optimizations.
        a_layout = X.get_layout()
        b_layout = W.get_layout()
        c_layout = Bias.get_layout() if Bias is not None else None

        d_layout = copy.deepcopy(Y.get_layout())
        match_list = [
            CUTLASSGemmTemplate.layout_match(buf.get_layout(), op_layout)
            for buf, op_layout in zip(
                (X, W, Bias, Y),
                (op.A.layout, op.B.layout, op.C.layout, op.D.layout),
            )
            if buf is not None
        ]
        all_match = all(match_list)
        if all_match:
            return op
        log.warning(
            f"Cutlass GEMM Layout change: Input and/or output layouts have changed between autotuning/retuning and call to render on {self}. Applying workaround. This can lead to suboptimal performance. Match List: {match_list}"  # noqa: G004, B950
        )
        new_op = copy.deepcopy(op)

        if a_layout is not None:
            new_op.A.layout = CUTLASSGemmTemplate.cutlass_layout(a_layout)
        if b_layout is not None:
            new_op.B.layout = CUTLASSGemmTemplate.cutlass_layout(b_layout)
        if c_layout is not None:
            new_op.C.layout = CUTLASSGemmTemplate.cutlass_layout(c_layout)
            new_op.C.element = cutlass_utils.torch_dtype_to_cutlass_type(c_layout.dtype)
        if d_layout is not None:
            new_op.D.layout = CUTLASSGemmTemplate.cutlass_layout(d_layout)
        return new_op

    def _dtype_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        """
        Checking dtypes of A, B, acc, D here.

        Empirically speaking, CUTLASS2x ops have same dtype for C and D.
        """
        X = self.input_nodes[0]
        W = self.input_nodes[1]

        accumulator_torch_dtype = cutlass_utils.get_accumulator_dtype(
            [X.get_dtype(), W.get_dtype()],
        )
        if not (
            cutlass_utils.dtype_match(X.get_dtype(), op.A.element)
            and cutlass_utils.dtype_match(W.get_dtype(), op.B.element)
            and cutlass_utils.dtype_match(
                self.output_node.get_layout().dtype, op.D.element
            )
            and cutlass_utils.dtype_match(
                accumulator_torch_dtype, op.accumulator_type()
            )
        ):
            return False

        return True

    @classmethod
    def global_filter_ops(
        cls,
        ops: list["cutlass_library.gemm_op.GemmOperation"],  # type: ignore[name-defined]  # noqa: F821
    ) -> list["cutlass_library.gemm_op.GemmOperation"]:  # type: ignore[name-defined]  # noqa: F821
        """
        Filter ops without using information about the torch op, input nodes and output node.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib  # type: ignore[import]

        # Skip simt kernels
        ops = [
            op
            for op in ops
            if op.tile_description.math_instruction.opcode_class
            != cutlass_lib.OpcodeClass.Simt
        ]

        # only keep the set of row x column ops
        # for other layout, we modify in place in filter_op, after deepcopy
        ops = [
            op
            for op in ops
            if op.A.layout.name == "RowMajor" and op.B.layout.name == "ColumnMajor"
        ]

        # filter by supported accumulator types
        ops = [
            op
            for op in ops
            if any(
                dtype_match(torch_dtype, op.accumulator_type())
                for torch_dtype in ACCUMULATOR_DTYPES
            )
        ]

        # check if dtypes of A and B are supported
        ops = [
            op
            for op in ops
            if any(dtype_match(torch_dtype, op.A.element) for torch_dtype in XW_DTYPES)
            and any(dtype_match(torch_dtype, op.B.element) for torch_dtype in XW_DTYPES)
        ]

        return ops

    def filter_op(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """
        Helper method:

        Determines whether a given Cutlass GEMM op definition is suitable for the current
        input / output of the operation that this template is supposed to implement.

        Takes memory layout, dtype and support for EVT operations into account,
        and filters potentially problematic ops.

        Returns None if the op is not suitable, otherwise returns the op to be used, which might
        have been mutated.
        """

        if op.gemm_kind not in self._get_supported_ops():
            return None

        X = self.input_nodes[0]
        W = self.input_nodes[1]

        # Filter ops according to the shape match.
        if not self._shape_match(op):
            return None

        # Filter ops by dtypes.
        if not self._dtype_match(op):
            return None

        # Filter ops by alignment.
        if not self._alignment_match(op):
            log.debug(
                "Skipping due to alignment mismatch. op: %s", op.configuration_name()
            )
            return None

        # only use stream k for static shape
        if op.tile_scheduler.name == "StreamK":
            static_shape = PythonWrapperCodegen.statically_known_list_of_ints_or_none(
                tuple(X.get_size()) + tuple(W.get_size())
            )
            if not static_shape:
                return None

        # Update op.
        op = copy.deepcopy(op)

        # set layouts for X and W
        self.set_layout(op.A, X.get_layout())
        self.set_layout(op.B, W.get_layout())

        # Set output layout.
        op.D.layout = CUTLASSGemmTemplate.cutlass_layout(self.output_node.get_layout())

        # Filter ops by alignments and set alignments.
        status = (
            self.set_alignment(X.get_layout(), op.A)
            and self.set_alignment(W.get_layout(), op.B)
            and self.set_alignment(self.output_node.get_layout(), op.D)
        )
        if not status:
            log.debug(
                "Skipping due to alignment setting failure. op: %s",
                op.configuration_name(),
            )
            return None

        if inductor_cutlass_config.cutlass_tma_only and not self._has_tma_epilogue(op):
            return None

        # Set epilogue.
        # TODO: update epilogue functor according to epilogues.
        op.element_epilogue = op.accumulator_type()

        if self.use_fast_accum is not None:
            is_op_fast_accum = "fastaccum" in op.configuration_name()
            if self.use_fast_accum ^ is_op_fast_accum:
                return None

        # Set bias layout and alignment.
        status = self._set_bias_layout_and_alignment(op)
        if not status:
            log.debug(
                "Skipping due to bias layout and alignment setting failure. op: %s",
                op.configuration_name(),
            )
            return None

        # Apply regex filters at the end when configuration name doesn't change anymore
        if inductor_cutlass_config.cutlass_op_allowlist_regex:
            if not re.search(
                inductor_cutlass_config.cutlass_op_allowlist_regex,
                op.configuration_name(),
            ):
                return None
        if inductor_cutlass_config.cutlass_op_denylist_regex is not None:
            if re.search(
                inductor_cutlass_config.cutlass_op_denylist_regex,
                op.configuration_name(),
            ):
                return None

        return op

    def gen_ops(self) -> "list[tuple[str, cutlass_gemm_op.GemmOperation]]":  # type: ignore[name-defined]  # noqa: F821
        """
        Creates a list of Cutlass GemmOperation instances that match the operation this template is designed to represent.
        The matching is carried out with respect to the input and output specifications of the operation.

        No function arguments.

        Returns:
            List[tuple[str, cutlass_gemm_op.GemmOperation]]: A list of (cutlass_name, GemmOperation)
            tuples that are compatible with the operation requirements of this template.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op

        if self.cache_key in self.filtered_ops_cache:
            log.debug("Using cached ops for %s", self.cache_key)
            return self.filtered_ops_cache[self.cache_key]

        with dynamo_timed("CUTLASSGemmTemplate.maybe_fetch_ops"):
            maybe_ops = maybe_fetch_ops()
        if maybe_ops is None:
            log.debug("Cannot fetch ops from cache, generating ops from scratch")
            full_ops = cutlass_utils.gen_ops()
            ops = pytree.tree_flatten(full_ops)[0]
        else:
            log.debug("Using cached ops from cache")
            ops = maybe_ops

        ops = self.global_filter_ops(ops)

        res: dict[str, cutlass_gemm_op.GemmOperation] = {}
        start_time = time.time()
        for op in ops:
            # if changed, need to also change CUTLASS_OPERATION_KIND
            assert isinstance(op, cutlass_gemm_op.GemmOperation)
            filter_res = self.filter_op(op)
            if (
                filter_res is not None
                and res.get(filter_res.configuration_name(), None) is None
            ):
                res[filter_res.configuration_name()] = filter_res
        log.info(
            "Got cutlass configs: total number of ops: %d. Filtering took %.2f seconds",
            len(res),
            time.time() - start_time,
        )
        sorted_res = sorted(res.items())
        ret_res = sorted_res[: inductor_cutlass_config.cutlass_max_profiling_configs]
        if len(self.filtered_ops_cache) < 50:
            self.filtered_ops_cache[self.cache_key] = ret_res
        else:
            log.debug("Not caching ops since filtered_ops_cache has reached size 50.")
        return ret_res

    def gemm_mode(self) -> str:
        """
        Returns a Cutlass GEMM mode string for the current operation, dependent on whether this op implements
        a batched GEMM or a simple GEMM without batch dimension.

        Returns:
        str: A string indicating the Cutlass GEMM mode. If the output node has more than two dimensions,
            "cutlass::gemm::GemmUniversalMode::kBatched" is returned, otherwise
            "cutlass::gemm::GemmUniversalMode::kGemm" is returned.
        """
        sizes = self.output_node.get_size()
        if len(sizes) > 2:
            return "cutlass::gemm::GemmUniversalMode::kBatched"
        else:
            return "cutlass::gemm::GemmUniversalMode::kGemm"

    def render(  # type: ignore[override]
        self,
        kernel: CUDATemplateKernel,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
        template_buffer_node: Optional[CUDATemplateBuffer] = None,
        epilogue_nodes: Optional[list[BaseSchedulerNode]] = None,
        **kwargs,
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        Renders the Cutlass based CUDA C++ code for the GEMM Kernel that this template is designed to implement,
        including potentially fused epilogues.

        Args:
            kernel (CUDATemplateKernel): The kernel to be rendered.
            op (cutlass_gemm_op.GemmOperation, optional): A GEMM operation that is required to be compatible with the
                input and output definitions as well as a possible epilogue. Defaults to None.
            **kwargs: Additional keyword arguments. Currently unused.

        Returns:
            str: Cutlass based CUDA C++ code fragment as a string, to be used by the current
            CUDATemplateKernel or autotuning code.

        Note:
            All inputs and their corresponding buffer addresses and names take precedence over previously
            passed inputs to the template at construction time. However, they should be layout compatible.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        assert isinstance(op, cutlass_gemm_op.GemmOperation), (
            "op argument is required and has to be an instance of GemmOperation"
        )

        if epilogue_nodes and not self._has_tma_epilogue(op):
            raise NotImplementedError(
                "Non-TMA epilogue visitor tree is not supported in Cutlass."
            )

        assert len(self.input_nodes) >= 2 and self.output_node is not None
        X, W = self.input_nodes[0], self.input_nodes[1]
        for input_node in self.input_nodes:
            if not isinstance(X.layout, FixedLayout):
                input_node.freeze_layout()

        Y = self.output_node
        if template_buffer_node is not None:
            Y = template_buffer_node

        Bias, extra_inputs, extra_names = self._get_extra_inputs_and_names(op)

        # Define Kernel call signature
        # Important: This step also populates Kernel name to node mapping data structures,
        # which are required further below ( for example by the template renderer )
        inputs = [X, W, Bias, *extra_inputs]
        names = ["X", "W", "Bias", *extra_names] + ["Y"]
        names_str = ",".join(names)
        if self.input_reorder is not None:
            input_reorder = self.input_reorder
        else:
            input_reorder = None

        # The layouts might have changed between autotuning and this call if they were FlexibleLayout
        # we need to adapt, which might lead to suboptimal performance.
        op = self.fix_op_layout(op, X, W, Bias, Y)

        # to make op mutable without affecting others
        op = copy.deepcopy(op)
        is_scaled_mm = len(self.input_nodes) in (4, 5)
        if Bias is not None and not is_scaled_mm:
            assert Bias.get_dtype() == X.get_dtype()
            # This might have been set to void during filtering, when the assumption was still that there's no C
            # operand
            op.C.element = op.A.element

            assert op.C.element == op.D.element, (
                f"Expect C and D to have the same dtype, found {op.C.element} and {op.D.element}"
            )

        argument_template, epilogue_template = self._get_template_args(op)
        should_swap_xw: bool = False
        if Bias is not None and self._has_tma_epilogue(op):
            if (
                op.epilogue_schedule
                != cutlass_lib.EpilogueScheduleType.EpilogueTransposed
                and self.should_swap_XW(Bias)
            ):
                # TMA epilogue requires bias vector in column major to get best perf.
                op = self.swap_XW(op)
                should_swap_xw = True

        name_to_buffer = {node.get_name(): node for node in self.input_nodes}
        # handle the fake output buffer during lowering
        name_to_buffer[Y.get_name()] = Y  # type: ignore[assignment]

        if epilogue_nodes or is_scaled_mm:
            if epilogue_nodes:
                (
                    input_names,
                    output_names,
                    var_name_to_buffer_name,
                    evt_py_code,
                ) = CutlassEVTCodegen.ir_to_evt_python_code(
                    Y.get_name(), epilogue_nodes, V.kernel.removed_buffers
                )

                # TODO: mlazos remove this by returning buffer metadata from
                # ir_to_evt_python code
                for name, buf in (
                    V.graph.name_to_buffer | V.graph.graph_inputs
                ).items():
                    if name not in name_to_buffer:
                        name_to_buffer[name] = buf  # type: ignore[assignment]

                D_output_name = var_name_to_buffer_name["D"]
                D_output_buffer = name_to_buffer[D_output_name]
                Y = D_output_buffer  # type: ignore[assignment]
                # Interestingly, I don't think the rest of the layout matters here since we
                # use the properties of the Y buffer to fill in D's properties in the epilogue
                # args. This is needed though because it defines types expected in the epilogue args.
                op.D.element = cutlass_utils.torch_dtype_to_cutlass_type(
                    D_output_buffer.get_dtype()
                )

                assert output_names, "There should be at least one write"

                epilogue_inputs = [name_to_buffer[name] for name in input_names]
                outputs = [name_to_buffer[name] for name in output_names]
            else:  # Scaled MM, we read the two scale matrices (and optional bias) and write a single output
                bias = None if len(self.input_nodes) < 5 else self.input_nodes[4]
                bias_name = bias.get_name() if bias else None

                (
                    evt_read_names,
                    var_name_to_buffer_name,
                    evt_py_code,
                ) = scaled_mm_evt(
                    self.input_nodes[2].get_name(),  # scale_A
                    self.input_nodes[3].get_name(),  # scale_B
                    bias_name,
                    Y.get_name(),
                )

                input_names = list(evt_read_names)
                output_names = []  # We only need Y
                epilogue_inputs = [self.input_nodes[2], self.input_nodes[3]]
                if bias:
                    epilogue_inputs.append(bias)
                outputs = []

            acc_dtype = cutlass_utils.get_accumulator_dtype(
                [X.get_dtype(), W.get_dtype()]
            )
            assert acc_dtype, "Could not determine accumulator dtype"

            evt_name, evt_args, evt_code, evt_arg_renames = self._render_evt(
                op,
                evt_py_code,
                var_name_to_buffer_name,
                name_to_buffer,
                Y.get_dtype(),
                acc_dtype,
            )

            inputs = [
                X,
                W,
                Bias,
                *epilogue_inputs,  # type: ignore[list-item]
                Y,
                *extra_inputs,
            ]
            input_names = [evt_arg_renames.get(name) for name in input_names]
            output_names = [evt_arg_renames.get(name) for name in output_names]

            names_str = ",".join(
                ["X", "W", "Bias", *input_names, "Y", *output_names, *extra_names]
            )
        else:
            evt_name = None
            outputs = [Y]
            evt_args = f"{{ElementComputeEpilogue({self.alpha}), ElementComputeEpilogue({self.beta})}}"
            evt_code = ""

        kernel_call_signature = kernel.def_kernel(
            inputs=inputs,  # type: ignore[arg-type]
            outputs=outputs,  # type: ignore[arg-type]
            names_str=names_str,
            input_reorder=input_reorder,
        )

        test_call_statement = self.test_call_statement(kernel, inputs, names_str)

        instance_definition, instance_type = self._define_gemm_instance(op, evt_name)

        options = {
            "alpha": self.alpha,
            "beta": self.beta,
            "X": X,
            "W": W,
            "Y": Y,
            "kernel_call_signature": kernel_call_signature,
            "Bias": Bias,
            "epilogue_template": epilogue_template,
            "argument_template": argument_template,
            "should_swap_xw": should_swap_xw,
            "template": self,
            "kernel": kernel,
            "instance_definition": instance_definition,
            "instance_type": instance_type,
            "input_reorder": self.input_reorder,
            "epilogue_args": evt_args,
            "test_call_statement": test_call_statement,
            "op_conf_name": op.configuration_name(),
            "epilogue_visitor_tree": evt_code,
        }
        options.update(dict(zip(extra_names, extra_inputs)))
        res = self._template_from_string(self._get_template()).render(**options)
        if inductor_cutlass_config.generate_test_runner and not is_dynamic(
            X, W, Y, Bias
        ):
            test_runner_code = self._template_from_string(
                GEMM_STANDALONE_RUNNER_TEMPLATE
            ).render(**options)
            res += "\n\n" + test_runner_code

        # splice to remove trailing spaces in each line
        buf = IndentedBuffer()
        buf.splice(res)
        return buf.getvalue()

    def test_call_statement(
        self,
        kernel,
        input_nodes,
        names_str: str = "",
    ) -> str:
        """
        Helper method to render the Cutlass CUDA C++ code required for calling the GEMM operation in the standalone
        test runner that might also be generated along with the rest of the code, if the corresponding config is
        enabled.

        Returns a C++ statement that calls the GEMM operation with the correct arguments.
        """
        _, __, arg_types = kernel.args.cpp_argdefs(cutlass_utils.DTYPE_TO_CUTLASS_TYPE)
        arg_names = [name.strip() for name in names_str.strip().split(",")]
        arg_names = self._update_arg_names_for_test_call_statement(
            arg_names, input_nodes
        )
        arguments = [
            f"(({arg_type}){arg_name}_data.get())"
            for arg_type, arg_name in zip(arg_types, arg_names)
        ]
        return f"{kernel.kernel_name}({', '.join(arguments)}, M, N, K, B, lda, ldb, ldc, ldd, 0, 0, 0, swizzle, workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);"  # noqa: B950

    def _render_evt(
        self,
        op: GemmOperation,
        evt_py_code: str,
        buffer_renames: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        output_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
    ) -> tuple[str, str, str, EVTArgRenames]:  # type: ignore[name-defined]  # noqa: F821
        raise NotImplementedError("_render_evt in CUTLASSGemmTemplate not implemented")


class CUTLASS3xGemmTemplate(CUTLASSGemmTemplate):
    """
    CUTLASS 3x GEMM Template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[list[int]] = None,
        use_fast_accum: Optional[bool] = None,
    ):
        super().__init__(
            input_nodes, layout, alpha, beta, input_reorder, use_fast_accum
        )

    @staticmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[list[int]] = None,
        use_fast_accum: Optional[bool] = None,
        **extra_kwargs,
    ) -> None:
        template = CUTLASS3xGemmTemplate(
            input_nodes,
            layout,
            alpha,
            beta,
            input_reorder,
            use_fast_accum,
        )
        template._add_cutlass_gemm_choices(
            choices, layout, input_nodes, alpha, beta, input_reorder, **extra_kwargs
        )

    @staticmethod
    @functools.lru_cache(1)
    def _get_supported_ops() -> "list[cutlass_library.gemm_operation.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        import cutlass_library.library as cutlass_lib

        return [cutlass_lib.GemmKind.Universal3x]

    def _get_template(self) -> str:
        return GEMM_TEMPLATE_CUTLASS_3X

    def _get_template_args(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[str, Optional[str]]:
        return (GEMM_ARGS_CUTLASS_3X, GEMM_ARGS_CUTLASS_3X_EPILOGUE)

    @staticmethod
    def _has_tma_epilogue(  # noqa: F821 # type: ignore[arg-type,name-defined]
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined,arg-type] # noqa: F821
    ) -> bool:  # type: ignore[name-defined]
        """Helper method: Determine whether a given Cutlass GEMM op has a TMA Epilogue"""
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        result = False
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result

    @staticmethod
    def supports_epilogue_fusion(op: GemmOperation) -> bool:
        return CUTLASS3xGemmTemplate._has_tma_epilogue(op)

    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for General Matrix Multiply (GEMM).

        This function checks compatibility of A, B, and possibly C operand layouts for
        a General Matrix Multiply (GEMM) operation, expressed as 'alpha * matmul(A, B) + beta * C'.
        It verifies requirements such as matching data types, minimum rank, and suitability
        for broadcasting, as defined by PyTorch operations like `torch.matmul`, `torch.aten.mm`,
        `addmm`, `bmm`, `baddbmm`, etc.

        Args:
            layouts (List[Layout]): List containing 2 or 3 Layout objects representing
                                    the input matrices A, B, and possibly C.

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
        assert 2 <= len(layouts) <= 5
        # Check if A and B are compatible
        A_layout, B_layout = layouts[:2]
        if len(A_layout.size) < 1:
            return False
        if len(B_layout.size) < 1:
            return False
        A_size = list(V.graph.sizevars.size_hints(A_layout.size))
        B_size = list(V.graph.sizevars.size_hints(B_layout.size))
        if len(A_size) < 2:
            A_size.insert(0, 1)
        if len(B_size) < 2:
            A_size.insert(1, 1)
        # Are batch dims broadcastable?
        while len(A_size) < len(B_size):
            A_size.insert(0, 1)
        while len(B_size) < len(A_size):
            B_size.insert(0, 1)
        K = max(A_size[-1], B_size[-2])
        M = A_size[-2]
        N = B_size[-1]
        if K != A_size[-1] and A_size[-1] != 1:
            return False
        if K != B_size[-2] and B_size[-1] != 1:
            return False
        # check batch dim broadcastable
        for i in range(len(A_size) - 2):
            if A_size[i] != B_size[i] and A_size[i] != 1 and B_size[i] != 1:
                return False
        if len(layouts) == 3:
            C_layout = layouts[2]
            C_size = [V.graph.sizevars.size_hint(i) for i in C_layout.size]
            while len(C_size) < len(A_size):
                C_size.insert(0, 1)
            # check batch dims
            for i in range(len(A_size) - 2):
                bd = max(A_size[i], B_size[i])
                if bd != C_size[i] and C_size[i] != 1:
                    return False
            if len(C_size) > len(A_size):
                # This may happen if the last elements of C are contiguous and
                # their multiplied size equals the last dim size of B
                if M != C_size[len(A_size) - 2] and C_size[len(A_size) - 2] != 1:
                    return False
                remaining_size = 1
                for i in range(len(A_size) - 1, len(C_size)):
                    remaining_size *= C_size[i]
                if N != remaining_size and remaining_size != 1:
                    return False
                return True
            assert len(C_size) == len(A_size)
            if M != C_size[-2] and C_size[-2] != 1:
                return False
            if N != C_size[-1] and C_size[-1] != 1:
                return False
        return True

    def _render_evt(
        self,
        op: GemmOperation,
        evt_py_code: str,
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        output_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
    ) -> tuple[str, str, str, EVTArgRenames]:
        from .lib_extensions.evt_extensions import create_example_tensors, trace

        acc_dtype = torch_dtype_to_cutlass_type(accumulator_dtype)
        output_dtype = torch_dtype_to_cutlass_type(output_dtype)

        examples = create_example_tensors(
            var_name_to_buffer_name,
            name_to_buffer,  # type: ignore[arg-type]
            V.graph.sizevars.size_hint,
        )
        evt_name, evt_args, evt_code, arg_renames = trace(
            evt_py_code,
            examples,
            acc_dtype,
            output_dtype,
            op.tile_description,  # type: ignore[attr-defined]
            op.epilogue_schedule,  # type: ignore[attr-defined]
            {k: name_to_buffer[v] for k, v in var_name_to_buffer_name.items()},  # type: ignore[arg-type,misc]
            V.graph.sizevars.size_hint,
        )

        return (
            evt_name,
            evt_args,
            evt_code,
            arg_renames,
        )

    def _shape_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        return True

    def _alignment_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        return True

    def _set_bias_layout_and_alignment(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        has_bias = len(self.input_nodes) == 3 and self.input_nodes[2] is not None
        if has_bias:
            Bias = self.input_nodes[2]
            # bias dtype
            op.C.element = cutlass_utils.torch_dtype_to_cutlass_type(
                Bias.get_layout().dtype
            )

            # Bias layout
            bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
            op.C.layout = bias_layout

            # Bias alignment
            status = self.set_alignment(Bias.get_layout(), op.C)
            if not status:
                return False
        else:
            op.C.element = cutlass_lib.DataType.void
        return True

    def _define_gemm_instance(
        self,
        op: GemmOperation,
        evt_name: Optional[str] = None,
    ) -> tuple[str, str]:
        """Defines and renders the Cutlass / CUDA C++ code for a given GEMM operation instance.

        This function uses the Cutlass library to generate key parts of the codegen process. General Matrix Multiply
        forms a core part of a number of scientific applications, so this efficient and adaptable implementation is
        crucial.

        Args:
            op (cutlass_library.gemm_op.GemmOperation): This is the core GEMM operation that we are defining and rendering.

        Returns:
            tuple[str, str]: A tuple where the first part is a string that constitutes the defined GEMM operation in C++
                             code (render) and the second part is the string that specifies the operation type.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        from .lib_extensions import gemm_operation_extensions as gemm_extensions

        emitter = gemm_extensions.EmitGemmUniversal3xInstanceWithEVT(evt_name=evt_name)  # type: ignore[call-arg]

        if not hasattr(op, "epilogue_functor") or not isinstance(
            op.epilogue_functor, enum.Enum
        ):
            op = copy.deepcopy(op)
            op.epilogue_functor = cutlass_lib.EpilogueFunctor.LinearCombination

        op_def = emitter.emit(op)
        pattern = re.compile(r"\s*struct\s(.*?)\s:")
        decl = [line for line in op_def.split("\n") if "struct " in line][-1]

        match = pattern.match(decl)
        if match is None:
            raise RuntimeError("Invalid Gemm config: \n" + op_def)
        op_type = match.groups()[0]
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            op_def += f"\n  using {op_type}_device_type = cutlass::gemm::device::GemmUniversalAdapter<{op_type}>;\n"
            op_type = f"{op_type}_device_type"

        return op_def, op_type

    def _get_extra_inputs_and_names(
        self,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[Optional[Buffer], list[Optional[Buffer]], list[str]]:
        Bias = self.input_nodes[2] if len(self.input_nodes) == 3 else None
        inputs: list[Optional[Buffer]] = []
        names: list[str] = []
        return (Bias, inputs, names)

    def _update_arg_names_for_test_call_statement(
        self,
        arg_names: list[str],
        input_nodes: list[Buffer],
    ) -> list[str]:
        if input_nodes[2] is None:
            del arg_names[2]
        else:
            # Reorder them as Bias, A, B
            if self.input_reorder is not None:
                arg_names[0 : len(self.input_reorder)] = [
                    arg_names[i] for i in self.input_reorder
                ]
        return arg_names

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
        """
        Render the Cutlass CUDA C++ code required for passing arguments to the GEMM operation.

        Args:
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            should_swap_xw (bool): Determines whether X, W operands should be swapped. If True, applies an explicit
            transpose operation to X and W.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The bias tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs.
            beta (float): Scaling factor for the output tensor.
            kernel (CUDATemplateKernel): CUDA Template kernel for the operation.
            epilogue_args (any): Additional arguments for the epilogue state.

        Returns:
            str: A block of CUDA C++ code as a string, ready to be used as arguments for the GEMM operation.

        Note: If `should_swap_xw` is True, a transpose operation will be applied to the X, W, Bias, and Y
        tensors. This operation also implies the M and N dimensions of Bias and GEMM output to be swapped
        before the function call.
        """
        options = {
            "alpha": alpha,
            "beta": beta,
            "X": X,
            "W": W,
            "Y": Y,
            "Bias": Bias,
            "template": self,
            "kernel": kernel,
            "M": "M",
            "N": "N",
            "epilogue_args": epilogue_args,
        }
        assert epilogue_template is not None

        if should_swap_xw:
            # Swap
            def clone_with_transposed_stride(node: IRNode) -> IRNode:
                old_layout = node.get_layout()
                new_stride = list(old_layout.stride)  # type: ignore[union-attr]
                new_stride[-2], new_stride[-1] = new_stride[-1], new_stride[-2]
                assert old_layout.device is not None
                new_layout = FixedLayout(
                    old_layout.device,
                    old_layout.dtype,
                    list(old_layout.size),  # type: ignore[union-attr]
                    new_stride,
                    old_layout.offset,  # type: ignore[union-attr]
                )
                return Buffer(name=node.get_name(), layout=new_layout)

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

        return arguments


class CUTLASS2xGemmTemplate(CUTLASSGemmTemplate):
    """CUTLASS 2x GEMM Template, which is used to generate CUTLASS GEMM kernels"""

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[list[int]] = None,
    ):
        super().__init__(input_nodes, layout, alpha, beta, input_reorder)

    @staticmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[list[int]] = None,
        use_fast_accum: Optional[bool] = False,
        **extra_kwargs,
    ) -> None:
        template = CUTLASS2xGemmTemplate(
            input_nodes, layout, alpha, beta, input_reorder
        )
        template._add_cutlass_gemm_choices(
            choices, layout, input_nodes, alpha, beta, input_reorder, **extra_kwargs
        )

    @staticmethod
    def _get_supported_ops() -> "list[cutlass_library.gemm_operation.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        import cutlass_library.library as cutlass_lib

        return [cutlass_lib.GemmKind.Universal, cutlass_lib.GemmKind.Sparse]

    @staticmethod
    def _has_tma_epilogue(self) -> bool:
        return False

    def _get_template(self) -> str:
        return GEMM_TEMPLATE_CUTLASS_2X

    def _get_template_args(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[str, Optional[str]]:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            return (GEMM_ARGS_SPARSE_CUTLASS_2X, None)

        return (GEMM_ARGS_CUTLASS_2X, None)

    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for set of operations supported by this class.

        Args:
            layouts (List[Layout]): List containing Layout objects representing
                                    the input matrices.

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
        assert len(layouts) == 2 or len(layouts) == 3
        # Check if A and B are compatible
        A_layout, B_layout = layouts[:2]
        if len(A_layout.size) != 2:
            return False
        if len(B_layout.size) != 2:
            return False
        A_size = [int(i) for i in A_layout.size]
        B_size = [int(i) for i in B_layout.size]
        K = max(A_size[1], B_size[0])
        return (K == A_size[1] or K == 2 * A_size[1]) and K == B_size[0]

    def _shape_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        X, W = self.input_nodes[0], self.input_nodes[1]

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            return X.get_size()[1] * 2 == W.get_size()[0]

        return X.get_size()[1] == W.get_size()[0]

    def _alignment_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind != cutlass_lib.GemmKind.Sparse:
            return True

        # SparseGemm in CUTLASS has specific alignment check that for
        # small k could make some of the choices throw kMisalignedOperand
        # CUTLASS error when run, see:
        # https://github.com/NVIDIA/cutlass/blob/e01b9b5029b7caca5a43c29f7d2714d7cf1dcae8/include/cutlass/gemm/kernel/sparse_gemm.h#L198-L200  # noqa: B950
        # So, let's skip these choices if that would be the case.
        X = self.input_nodes[0]
        return (X.get_size()[1] * 2) % op.tile_description.tile_shape[2] == 0

    def _set_bias_layout_and_alignment(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            op.C.layout = op.D.layout
            return True

        if len(self.input_nodes) >= 3 and self.input_nodes[2] is not None:
            Bias = self.input_nodes[2]
            bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
            if bias_layout != op.D.layout:
                # For cutlass2, bias and output layout must match
                return False
            if not self.set_alignment(Bias.get_layout(), op.C):
                return False
        else:
            op.C.layout = op.D.layout
        return True

    def _define_gemm_instance(
        self,
        op: GemmOperation,
        evt_name: Optional[str] = None,
    ) -> tuple[str, str]:
        """Defines and renders the Cutlass / CUDA C++ code for a given GEMM operation instance.

        This function uses the Cutlass library to generate key parts of the codegen process. General Matrix Multiply
        forms a core part of a number of scientific applications, so this efficient and adaptable implementation is
        crucial.

        Args:
            op (cutlass_library.gemm_op.GemmOperation): This is the core GEMM operation that we are defining and rendering.

        Returns:
            tuple[str, str]: A tuple where the first part is a string that constitutes the defined GEMM operation in C++
                             code (render) and the second part is the string that specifies the operation type.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            emitter = cutlass_gemm_op.EmitSparseGemmInstance()
        else:
            emitter = cutlass_gemm_op.EmitGemmInstance()
        op_def = emitter.emit(op)
        op_def = op_def.replace(
            "cutlass::gemm::device::Gemm", "cutlass::gemm::device::GemmUniversal"
        )
        if op.gemm_kind != cutlass_lib.GemmKind.Sparse:
            op_def = op_def.replace("false,", "")
        pattern = re.compile(r"\s*using\s(.*?)\s=")
        decl = op_def.split("\n")[2]

        match = pattern.match(decl)
        if match is None:
            raise RuntimeError("Invalid Gemm config: \n" + op_def)
        op_type = match.groups()[0]
        return op_def, op_type

    def _get_extra_inputs_and_names(
        self,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[Optional[Buffer], list[Optional[Buffer]], list[str]]:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            Bias = None
            Meta = self.input_nodes[2]
        else:
            Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]
            Meta = None
        inputs = [Meta]
        names = ["Meta"]
        return (Bias, inputs, names)

    def _update_arg_names_for_test_call_statement(
        self,
        arg_names: list[str],
        input_nodes: list[Buffer],
    ) -> list[str]:
        if input_nodes[3] is None:
            del arg_names[3]
        if input_nodes[2] is None:
            del arg_names[2]
        return arg_names

    def render_gemm_arguments(
        self,
        instance_type: str,
        argument_template: str,
        epilogue_template: str,
        should_swap_xw: bool,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Meta: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        kernel: CUDATemplateKernel,
        epilogue_args,
    ) -> str:
        """
        Render the Cutlass CUDA C++ code required for passing arguments to the GEMM operation.

        Args:
            instance_type (str): GEMM instance type.
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            should_swap_xw (bool): Determines whether X, W operands should be swapped. If True, applies an explicit
            transpose operation to X and W.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The bias tensor.
            Meta (IRNode): The meta tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs.
            beta (float): Scaling factor for the output tensor.
            kernel (CUDATemplateKernel): CUDA Template kernel for the operation.
            epilogue_args (any): Additional arguments for the epilogue state.

        Returns:
            str: A block of CUDA C++ code as a string, ready to be used as arguments for the GEMM operation.

        Note: If `should_swap_xw` is True, a transpose operation will be applied to the X, W, Bias, and Y
        tensors. This operation also implies the M and N dimensions of Bias and GEMM output to be swapped
        before the function call.
        """
        options = {
            "instance_type": instance_type,
            "alpha": alpha,
            "beta": beta,
            "X": X,
            "W": W,
            "Y": Y,
            "Bias": Bias,
            "Meta": Meta,
            "template": self,
            "kernel": kernel,
            "M": "M",
            "N": "N",
            "epilogue_args": epilogue_args,
        }

        if epilogue_template is None:
            arguments = self._template_from_string(argument_template).render(
                split_k=1, **options
            )
            return arguments

        epilogue_arguments = self._template_from_string(epilogue_template).render(
            **options
        )
        arguments = self._template_from_string(argument_template).render(
            epilogue_arguments=epilogue_arguments, **options
        )

        return arguments
