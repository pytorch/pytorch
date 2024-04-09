
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_jezng/av/cavkrmvfrksyskri2gjlv4t722cnoa4xtsdn4ne3v4kjavmx4u4s.py
# Source Nodes: [matmul], Original ATen: [aten.mm]
# matmul => mm
cuda_fused_mm_0 = async_compile.cuda(r'''
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/distribution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"

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


// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with PT_EXPORT to make
// them visible.
#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define PT_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define PT_EXPORT __declspec(dllexport)
#else
#define PT_EXPORT
#endif
#endif
using bfloat16 = nv_bfloat16;

using namespace cute;
#define CUTLASS_CHECK(status)                                                      \
{                                                                                  \
  cutlass::Status error = status;                                                  \
  if (error != cutlass::Status::kSuccess) {                                        \
    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \
        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \
    throw std::runtime_error(msg);                                                 \
  }                                                                                \
}

// Used as pass-through functor in EVT just for type casting / rounding
template <typename T>
struct identity_op {
  CUTLASS_HOST_DEVICE
  T operator()(T val) const { return val; }
};



using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    void, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_epilogue::SharedStorage)>,
  cutlass::gemm::KernelTmaWarpSpecialized
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem
using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_mainloop,
    cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem :
  public cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_base { };


  using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_device_type = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem>;

// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.
extern "C" {
PT_EXPORT int cuda_fused_mm_0(const half* X, const half* W, half* Y, size_t* workspace_size, uint8_t* workspace, cudaStream_t stream) {
  try {
  int64_t B = 1;
  int64_t M = 2048L;
  int64_t K = 4096L;
  int64_t N = 512L;
  using ElementComputeEpilogue = cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_device_type::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  static cutlass::KernelHardwareInfo hw_info;
  if (hw_info.sm_count == 0) {
    // @TODO kadeng: Add support for Multi-GPU machines with heterogeneous SM counts
    // for now we just pick the SM count of the first GPU
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
  }
  cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_device_type::Arguments arguments;

  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    cutlass::gemm::GemmUniversalMode::kGemm,  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      (cutlass::half_t*)(X),  // ElementA const* ptr_A
      {
        4096L /* stride_x0 */,
        cute::Int<1>{} /* stride_x1 */,
        0 /* batch_stride_x */
      },  // StrideA dA
      (cutlass::half_t*)(W),  // ElementB const* ptr_B
      {
        cute::Int<1>{} /* stride_w1 */,
        512L /* stride_w0 */,
        0 /* batch_stride_w */
      },  // StrideB dB
    },  // MainloopArguments mainloop

    // see https://tinyurl.com/4rk89z48
    {
      {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // thread, typename FusionCallbacks::Arguments ( EVT ) or ThreadEpilogueOp::Params (non-EVT )
      nullptr,  // ElementC const* ptr_C
      {
        cute::Int<1>{} /* stride_bias0 */,
        cute::Int<1>{} /* stride_bias1 */,
        0 /* batch_stride_bias */
      },  // StrideC dC
      (cutlass::half_t*)(Y),  // ElementD const* ptr_D
      {
        512L /* stride_y0 */,
        cute::Int<1>{} /* stride_y1 */,
        0 /* batch_stride_y */
      },  // StrideD dD
    },  // EpilogueArguments epilogue,
    hw_info
  };
  cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmem_device_type gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }
  // check for null pointers after workspace size, since querying workspace size doesn't require valid data pointers
#ifndef CUTLASS_BACKEND_DISABLE_CHECKS

  {
    if (!X) {
      int64_t X_size = 8388608L;
      if (X_size > 0) {
        throw std::runtime_error("input X is null but size is not 0!");
      }
    }
  }


  {
    if (!W) {
      int64_t W_size = 2097152L;
      if (W_size > 0) {
        throw std::runtime_error("input W is null but size is not 0!");
      }
    }
  }



  {
    if (!Y) {
      int64_t Y_size = 1048576L;
      if (Y_size > 0) {
        throw std::runtime_error("input Y is null but size is not 0!");
      }
    }
  }

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


#ifdef GENERATE_STANDALONE_RUNNER
/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed, float max=1.0, float min=-1.0) {
  if (block.size()<=0) return false;
  Element scope_max(static_cast<Element>(max)), scope_min(static_cast<Element>(min));
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

extern "C" int run_standalone(uint64_t seed, int repetitions) {
    std::cout << "Starting GEMM Standalone test run with seed " << seed << std::endl;
    size_t workspace_size = 0;
    size_t* workspace_size_ptr = &workspace_size;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = uint8_t; // may not be void
    using ElementD = cutlass::half_t;

    cutlass::DeviceAllocation<ElementA> X_data(8388608);
    initialize_block(X_data, seed++);
    cutlass::DeviceAllocation<ElementB> W_data(2097152);
    initialize_block(W_data, seed++);
    cutlass::DeviceAllocation<ElementC> Bias_data(0);
    initialize_block(Bias_data, seed++);
    cutlass::DeviceAllocation<ElementD> Y_data(1048576);

    cutlass::DeviceAllocation<uint8_t> workspace_data;
    // Call once with workspace_size_ptr set to get workspace size

    std::cout << "Calling once to get workspace size" << std::endl;
    cuda_fused_mm_0(((const half*)X_data.get()), ((const half*)W_data.get()), ((half*)Y_data.get()), workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);;
    // Allocate workspace if neccessary
    if (workspace_size > 0) {
        workspace_data.reset(workspace_size);
        std::cout << "Allocated workspace size of " << workspace_size << " bytes" << std::endl;
    }
    std::cout << "Calling Kernel as cuda_fused_mm_0(((const half*)X_data.get()), ((const half*)W_data.get()), ((half*)Y_data.get()), workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);;" << std::endl;
    workspace_size_ptr = nullptr;
    for (int i=0; i<repetitions; i++) {
        cuda_fused_mm_0(((const half*)X_data.get()), ((const half*)W_data.get()), ((half*)Y_data.get()), workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);;
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
''', 'so')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_jezng/k7/ck76avlwhdvwrzj4dowe3ad4dg6ynv2hoxttd7jbdbtxbwnat5jn.py
# Source Nodes: [mul, sub], Original ATen: [aten.mul, aten.sub]
# mul => mul
# sub => sub
triton_poi_fused_mul_sub_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = 3.3
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 512), (512, 1))
    assert_size_stride(arg2_1, (2048, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 512), (512, 1), torch.float16)
        stream0 = get_raw_stream(0)
        cuda_fused_mm_0.cuda_fused_mm_0(c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf0.data_ptr()), None, None, c_void_p(stream0))
        del arg0_1
        del arg1_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [mul, sub], Original ATen: [aten.mul, aten.sub]
        triton_poi_fused_mul_sub_1.run(buf1, arg2_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg2_1
        return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2048, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
