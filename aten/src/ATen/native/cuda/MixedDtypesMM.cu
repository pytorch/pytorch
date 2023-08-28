#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>

#ifndef USE_ROCM
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>

#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/gemm/kernel/default_gemm.h>

#include <cutlass_extensions/epilogue_helpers.h>
#include <cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h>
#include <cutlass_extensions/gemm/kernel/fpA_intB_gemm.h>
#include <cutlass_extensions/gemm/threadblock/default_mma.h>
#endif

#ifndef USE_ROCM
#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }
#endif

namespace at {
namespace native {

Tensor
_fp16_uint8_mm(const Tensor& self, const Tensor& mat2, const Tensor& scale,
              const Tensor& bias) {
#ifndef USE_ROCM
    // For now, only CC 8.x devices are supported.
    const auto dprops = at::cuda::getCurrentDeviceProperties();
    const auto is_sm8x = dprops->major == 8;
    TORCH_CHECK(is_sm8x,
                "_fp16_uint8_mm: Supported only on GPUs with compute "
                "capability 8.x");

  // Check arguments dimensions.
  TORCH_CHECK(self.dim() == 2,
              "_fp16_uint8_mm: Expected self argument to be 2D tensor, got ",
              self.dim(), " dims");
  TORCH_CHECK(mat2.dim() == 2,
              "_fp16_uint8_mm: Expected mat2 argument to be 2D tensor, got ",
              mat2.dim(), " dims");

  // Check arguments datatypes.
  TORCH_CHECK(self.dtype() == at::kHalf,
              "_fp16_uint8_mm: The self datatype ", self.dtype(),
              " is not supported");
  TORCH_CHECK(mat2.dtype() == at::kByte,
              "_fp16_uint8_mm: The mat2 datatype ", mat2.dtype(),
              " is not supported");
  TORCH_CHECK(scale.dtype() == at::kHalf,
              "_fp16_uint8_mm: The scale datatype ", scale.dtype(),
              " is not supported");
  TORCH_CHECK(bias.dtype() == at::kHalf,
              "_fp16_uint8_mm: The bias datatype ", bias.dtype(),
              " is not supported");

  // FIXME: add all the other checks!

  const int length_m = self.size(0);
  const int length_k = mat2.size(0);
  const int length_n = mat2.size(1);

  using ElementInputA = cutlass::half_t;
  using ElementInputB = uint8_t;
  using ElementOutput = ElementInputA;

  using SmArch = cutlass::arch::Sm80;
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Operator = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;

  constexpr auto ThreadblockK = 64;
  constexpr auto ElementsPerCacheLine = 128 * 8 / cutlass::sizeof_bits<uint8_t>::value;
  constexpr auto ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
  using LayoutOutput = LayoutInputA;

  constexpr auto ElementsPerAccessA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr auto ElementsPerAccessB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  constexpr auto ElementsPerAccessC = ElementsPerAccessA;
  constexpr auto Stages = 4;
  constexpr auto split_k_factor = 1; // FIXME: wrong results if !=1,
                                     // even if GemmFpAIntB
                                     // instantiated with SplitKSerial
                                     // set to false.

  using ElementAccumulator = float;

  using EpilogueTag = fastertransformer::EpilogueOpBias;
  using EpilogueOp = typename fastertransformer::Epilogue<
      ElementOutput,
      ElementsPerAccessC,
      ElementAccumulator,
      EpilogueTag>::Op;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInputA,
      LayoutInputA,
      ElementsPerAccessA,
      ElementInputB,
      LayoutInputB,
      ElementsPerAccessB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      ThreadblockSwizzle,
      Stages,
      true,
      Operator>::GemmKernel;
  using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<
      typename DefaultGemmKernel::Mma,
      typename DefaultGemmKernel::Epilogue,
      typename DefaultGemmKernel::ThreadblockSwizzle,
      SmArch,
      DefaultGemmKernel::kSplitKSerial>;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

  auto result = self.new_empty({length_m, length_n});

  const auto ldb = length_k * GemmKernel::kInterleave;

  typename Gemm::Arguments arguments(
      {length_m, length_n, length_k},
      {(ElementInputA*)self.data_ptr(), length_k},
      {(ElementInputB*)mat2.data_ptr(), ldb},
      {(ElementInputA*)scale.data_ptr(), 0},
      {(ElementInputA*)bias.data_ptr(), 0},
      {(ElementOutput*)result.data_ptr(), length_n},
      split_k_factor,
      {ElementAccumulator(1.f), ElementAccumulator(0.f)});

  Gemm gemm_op;

  cutlass::Status status;

  // Verify that GEMM operation with given arguments can be performed
  // by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status);

  // Allocate workspace for CUTLASS mixed datatypes GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = self.new_empty({(int64_t)workspace_size},
                                  at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS mixed datatypes GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  // Perform mixed datatypes GEMM operation.
  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return result;
#else
  AT_ERROR("_fp16_uint8_mm: ROCm doesn't support CUTLASS");
  return Tensor{};
#endif
}

}  // namespace native
}  // namespace at
