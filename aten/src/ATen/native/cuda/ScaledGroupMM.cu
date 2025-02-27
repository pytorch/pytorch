#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>

// Two warninngs in Cutlass included header files
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wset-but-not-used")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-parameter")

// Determine if the architecture supports rowwise scaled mm
// Currently failing on windows with:
// https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && \
    CUDA_VERSION >= 12000

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)

#include <cute/tensor.hpp>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/version.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

namespace {

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename DtypeScale,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput>
__global__ void prepare_gemm_data(
    DtypeA* A,
    DtypeB* B,
    DtypeOutput* output,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    DtypeScale** inputA_scale_ptrs,
    DtypeScale** inputB_scale_ptrs,
    ProblemShape* problem_sizes,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    const int32_t* offs,
    int32_t M,
    int32_t N,
    int32_t K) {
  int32_t tid = threadIdx.x;
  int32_t delta = 0;
  if (offs != nullptr) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    delta = offs[tid] - start;
  }
  if (M < 0) {
    M = delta;
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * K;
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1] * N;
    B_ptrs[tid] = B + tid * N * K;
  } else if (N < 0) {
    // TODO double check
    N = delta;
    A_ptrs[tid] = A + tid * M * K;
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1];
  } else if (K < 0) {
    // TODO double check
    K = delta;
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1];
    output_ptrs[tid] = output + tid * M * N;
  } else {
    A_ptrs[tid] = A + tid * M * K;
    B_ptrs[tid] = B + tid * N * K;
    output_ptrs[tid] = output + tid * M * N;
  }
  problem_sizes[tid] = ProblemShape(M, N, K);
  // TODO this works for ragged mata only
  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  stride_output[tid] =
      cutlass::make_cute_packed_stride(StrideOutput{}, {M, N, 1});
}

constexpr int kNumSMsForH100 = 132;

using DtypeScale = float;
using DtypeAccum = float;
using ProblemShape = cutlass::gemm::GroupProblemShape<
    cute::Shape<int32_t, int32_t, int32_t>>; // <M,N,K> per
                                             // group

template <bool FastAccum>
struct Schedule;

template <>
struct Schedule<true> {
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;
};

template <>
struct Schedule<false> {
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;
};

#endif

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}


template <typename FastAccum, typename BiasType>
void f8f8bf16_grouped_gemm_impl_sm90(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  using DtypeA = cutlass::float_e4m3_t;
  using DtypeB = cutlass::float_e4m3_t;
  using DtypeOutput = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 16 / sizeof(DtypeA);
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 16 / sizeof(DtypeB);
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  // Tag indicating the minimum SM that supports the intended feature
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = typename Schedule<FastAccum::value>::TileShape;
  using ClusterShape = typename Schedule<FastAccum::value>::ClusterShape;
  using KernelSchedule = typename Schedule<FastAccum::value>::KernelSchedule;
  using EpilogueSchedule =
      typename Schedule<FastAccum::value>::EpilogueSchedule; // Epilogue to
                                                             // launch

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          DtypeAccum,
          DtypeAccum,
          DtypeOutput,
          LayoutOutput*,
          AlignmentOutput,
          DtypeOutput,
          LayoutOutput*,
          AlignmentOutput,
          EpilogueSchedule,
          cutlass::epilogue::fusion::
              LinearCombination<DtypeOutput, DtypeAccum>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          DtypeA,
          LayoutA*,
          AlignmentA,
          DtypeB,
          LayoutB*,
          AlignmentB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;

  int32_t M, N, K, group_count;

  M = mat_a.size(-2);
  K = mat_a.size(-1);
  N = mat_b.size(-1);

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    // if both inputs are ragged, K is dynamic, M and N come from inputs
    group_count = offs->size(0);
    K = -1;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    M = -1;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    N = -1;
  } else {
    // regular bmm
    group_count = mat_a.size(0);
  }

  const int64_t problem_shape_size =
      group_count * ((int64_t)sizeof(ProblemShape::UnderlyingProblemShape));

  const int64_t stride_size = 3 * group_count * ((int64_t)sizeof(StrideA));

  const int group_alignment = 16/sizeof(void*);  
  const int aligned_group_count = round_up_to_nearest_multiple(group_count, group_alignment);
  int64_t input_args_size =
      aligned_group_count * 5 * sizeof(void*) + problem_shape_size + stride_size;

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  DtypeA** inputA_ptrs = reinterpret_cast<DtypeA**>(buf_ptr);
  DtypeB** inputB_ptrs = reinterpret_cast<DtypeB**>(inputA_ptrs + aligned_group_count);
  DtypeOutput** output_ptrs =
      reinterpret_cast<DtypeOutput**>(inputB_ptrs + aligned_group_count);
  DtypeScale** inputA_scale_ptrs =
      reinterpret_cast<DtypeScale**>(output_ptrs + aligned_group_count);
  DtypeScale** inputB_scale_ptrs =
      reinterpret_cast<DtypeScale**>(inputA_scale_ptrs + aligned_group_count);
  static_assert(
      sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
  StrideA* stride_A =
      reinterpret_cast<StrideA*>(inputB_scale_ptrs + aligned_group_count);
  StrideB* stride_B = reinterpret_cast<StrideB*>(stride_A + group_count);
  StrideOutput* stride_output =
      reinterpret_cast<StrideOutput*>(stride_B + group_count);
  ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          stride_output + group_count);

  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  std::cout << "Problem size " << sizeof(ProblemShape::UnderlyingProblemShape)
            << " stride " << sizeof(StrideA) << "\n";
  prepare_gemm_data<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA*>(mat_a.data_ptr()),
      reinterpret_cast<DtypeB*>(mat_b.data_ptr()),
      reinterpret_cast<DtypeOutput*>(out.data_ptr()),
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      inputA_scale_ptrs,
      inputB_scale_ptrs,
      problem_sizes,
      stride_A,
      stride_B,
      stride_output,
      offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
      M,
      N,
      K);

  auto buf_cpu = mat_a.new_empty(
      input_args_size, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  AT_CUDA_CHECK(cudaMemcpy(
      (char*)buf_cpu.data_ptr(),
      buf_ptr,
      input_args_size,
      cudaMemcpyDeviceToHost));
  char* buf_ptr_cpu = (char*)buf_cpu.data_ptr();
  DtypeA** inputA_ptrs_h = reinterpret_cast<DtypeA**>(buf_ptr_cpu);
  DtypeB** inputB_ptrs_h =
      reinterpret_cast<DtypeB**>(inputA_ptrs_h + aligned_group_count);
  DtypeOutput** output_ptrs_h =
      reinterpret_cast<DtypeOutput**>(inputB_ptrs_h + aligned_group_count);
  DtypeScale** inputA_scale_ptrs_h =
      reinterpret_cast<DtypeScale**>(output_ptrs_h + aligned_group_count);
  DtypeScale** inputB_scale_ptrs_h =
      reinterpret_cast<DtypeScale**>(inputA_scale_ptrs_h + aligned_group_count);
  StrideA* stride_A_h =
      reinterpret_cast<StrideA*>(inputB_scale_ptrs_h + aligned_group_count);
  StrideB* stride_B_h = reinterpret_cast<StrideB*>(stride_A_h + group_count);
  StrideOutput* stride_output_h =
      reinterpret_cast<StrideOutput*>(stride_B_h + group_count);
  ProblemShape::UnderlyingProblemShape* problem_sizes_h =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          stride_output_h + group_count);

  std::cout << "PTRS " << mat_a.data_ptr() << " " << mat_b.data_ptr() << " "
            << out.data_ptr() << "\n";
  for (int i = 0; i < group_count; i++) {
    std::cout << "A " << (void*)inputA_ptrs_h[i] << "\n";
    std::cout << "B " << (void*)inputB_ptrs_h[i] << "\n";
    std::cout << "O " << (void*)output_ptrs_h[i] << "\n";
    std::cout << "sizes " << problem_sizes_h[i] << "\n";
    std::cout << "strideA" << stride_A_h[i] << "\n";
    std::cout << "strideB" << stride_B_h[i] << "\n";
    std::cout << "stride_output" << stride_output_h[i] << "\n";
  }
//   int device_id = 0;
//   cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);


  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes, nullptr},
      {(const DtypeA**)inputA_ptrs,
       stride_A,
       (const DtypeB**)inputB_ptrs,
       stride_B},
      {{},
       (const DtypeOutput**)output_ptrs,
       stride_output,
       output_ptrs,
       stride_output}};
       //kernel_hw_info};

  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.;
  fusion_args.beta = 0.;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  // Single alpha and beta for all groups
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  arguments.epilogue.thread = fusion_args;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  std::cout << "allocating workspace " << workspace_size << "\n";
  auto workspace = allocator.allocate(workspace_size);
  std::cout << "workspace_ptr" << workspace.get() << "\n"; 
  Gemm gemm;
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "cutlass cannot implement");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess,
      "cutlass cannot initialize");
  auto status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass cannot run, error ", int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename BiasType>
void dispatch_fp8_grouped_gemm_on_fast_accum(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  if (use_fast_accum) {
    f8f8bf16_grouped_gemm_impl_sm90<std::true_type, BiasType>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  } else {
    TORCH_CHECK(
        false, "CUTLASS doesn't support grouped gemm without fast accum yet");
    // f8f8bf16_grouped_gemm_impl_sm90<std::true_type, BiasType>(
    //     mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  }
}

void dispatch_fp8_grouped_gemm_on_bias_dtype(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  if (bias.has_value() && bias->dtype() == at::kBFloat16) {
    dispatch_fp8_grouped_gemm_on_fast_accum<cutlass::bfloat16_t>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  } else {
    dispatch_fp8_grouped_gemm_on_fast_accum<float>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  }
}

} // namespace

namespace at::cuda::detail {
void f8f8bf16_grouped_mm(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs_a,
    std::optional<at::Tensor> offs_b,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
#if defined(BUILD_ROWWISE_FP8_KERNEL)
  dispatch_fp8_grouped_gemm_on_bias_dtype(
      mat_a, mat_b, scale_a, scale_b, offs_a, bias, use_fast_accum, out);
#else
#endif
}

} // namespace at::cuda::detail
