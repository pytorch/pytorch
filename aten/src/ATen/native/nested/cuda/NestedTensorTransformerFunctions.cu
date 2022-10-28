#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#ifndef USE_ROCM
#ifndef _WIN32
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#endif
#endif

#include <ATen/NestedTensorImpl.h>

#define BLOCK_DIM 256
#define GRID_DIM_Y 16

namespace at {
namespace native {

template <typename T>
__global__ void remove_padding_transform0213_2(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  int input_offset =
      batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i2 = i / sizes_i[1];
    const int i13 = i % sizes_i[1];
    const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
    const int i3 = i13 % (sizes_i[1] / input_sizes[1]);

    output[offset + i] = input
        [input_offset + i1 * input_sizes[2] * input_sizes[3] +
         i2 * input_sizes[3] + i3];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i2 = i / sizes_i[1];
    const int i13 = i % sizes_i[1];
    const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
    const int i3 = i13 % (sizes_i[1] / input_sizes[1]);
    output[offset + i] = input
        [input_offset + i1 * input_sizes[2] * input_sizes[3] +
         i2 * input_sizes[3] + i3];
  }
}

template <typename T>
__global__ void remove_padding_2(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  int input_offset = batch_id * input_sizes[1] * input_sizes[2];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / sizes_i[1];
    const int i1 = i % sizes_i[1];
    const int i0_offset = i0 * input_sizes[2];
    output[offset + i] = input[input_offset + i0_offset + i1];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / sizes_i[1];
    const int i1 = i % sizes_i[1];
    const int i0_offset = i0 * input_sizes[2];
    output[offset + i] = input[input_offset + i0_offset + i1];
  }
}

template <typename T>
__global__ void remove_padding(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
  int input_offset =
      batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
}

template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  dim3 grid;
  grid.x = batch_size;
  grid.y = GRID_DIM_Y;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  if (output_dim == 2) {
    remove_padding_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
  } else {
    remove_padding<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
  }
}

template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  dim3 grid;
  grid.x = batch_size;
  grid.y = GRID_DIM_Y;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(
      output_dim == 2,
      "remove padding transform0213 only support output dim == 2");

  remove_padding_transform0213_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
}

template void remove_padding_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template void remove_padding_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template void remove_padding_transform0213_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template void remove_padding_transform0213_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template <typename T>
__global__ void add_padding_1(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int batch_output_offset = batch_id * output_sizes_1;
  for (int ii = 0; ii < (output_sizes_1 / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int output_offset = batch_output_offset + i;
    if (batch_id < batch_size && i < sizes_i[0]) {
      const int batch_input_offset = offsets[batch_id];
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
  const int i = (output_sizes_1 / grainsize) * grainsize + tid;
  if (i < output_sizes_1) {
    const int output_offset = batch_output_offset + i;
    if (batch_id < batch_size && (i < sizes_i[0])) {
      const int batch_input_offset = offsets[batch_id];
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
}

template <typename T>
__global__ void add_padding_2(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    int output_sizes_2,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int output_offset = batch_id * output_sizes_1 * output_sizes_2;
  const int output_numel = output_sizes_1 * output_sizes_2;
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes_2);
    const int i1 = i - i0 * output_sizes_2;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int offset = offsets[batch_id];
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes_2);
    const int i1 = i - i0 * output_sizes_2;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int offset = offsets[batch_id];
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template <typename T>
__global__ void add_padding_3(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    int output_sizes_2,
    int output_sizes_3,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int output_offset =
      batch_id * output_sizes_1 * output_sizes_2 * output_sizes_3;
  const int output_numel = output_sizes_1 * output_sizes_2 * output_sizes_3;
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes_2 * output_sizes_3);
    const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
    const int i2 = i % output_sizes_3;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
        i2 < sizes_i[2]) {
      const int offset = offsets[batch_id];
      const int input_offset =
          offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes_2 * output_sizes_3);
    const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
    const int i2 = i % output_sizes_3;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
        i2 < sizes_i[2]) {
      const int offset = offsets[batch_id];
      const int input_offset =
          offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template <typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size) {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  dim3 grid;
  grid.x = output_batch_size;
  grid.y = GRID_DIM_Y;
  if (input_dim == 1) {
    add_padding_1<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        batch_size);
  }
  if (input_dim == 2) {
    add_padding_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        batch_size);
  }
  if (input_dim == 3) {
    add_padding_3<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        output_sizes[3],
        batch_size);
  }
}

template void add_padding_kernelLauncher<double>(
    double* input,
    double* output,
    double padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    float padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

template void add_padding_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    c10::Half padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

namespace {

#ifndef USE_ROCM
#ifndef _WIN32
template <
    typename scalar_t,
    unsigned int kPad,
    typename LayoutA,
    typename LayoutB,
    typename OpClass,
    typename Arch,
    typename ThreadBlockShape,
    typename WarpShape,
    typename InstructionShape>
void gemm_grouped_cuda_internal(
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    const std::vector<scalar_t*>& aptr,
    const std::vector<scalar_t*>& bptr,
    const std::vector<scalar_t*>& dptr,
    const std::vector<cutlass::gemm::GemmCoord>& gemm_sizes,
    const int problem_count,
    at::Device& device) {
  using Element = scalar_t;
  using ElementAcc = float;

  using GemmConfiguration =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          Arch,
          Element,
          Element,
          Element,
          ElementAcc>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      cutlass::layout::RowMajor,
      ElementAcc,
      OpClass,
      Arch,
      ThreadBlockShape,
      WarpShape,
      InstructionShape,
      typename GemmConfiguration::EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      GemmConfiguration::kStages>::GemmKernel;

  using GemmGrouped = typename cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using EpilogueOutputOp = typename GemmGrouped::GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(/*alpha*/ 1, /*beta*/ 0);

  const int64_t gemm_coord_size =
      problem_count * ((int64_t)sizeof(cutlass::gemm::GemmCoord));
  // Number of gmm args not including *problem_sizes
  at::Tensor gmm_args = at::empty(
      {problem_count * 6 + gemm_coord_size},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

  // Obtain pointers for each argument (on host)
  int64_t* lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  int64_t* ldb_data = lda_data + problem_count;
  int64_t* ldd_data = lda_data + 2 * problem_count;
  int64_t* ptr_a_data = lda_data + 3 * problem_count;
  int64_t* ptr_b_data = lda_data + 4 * problem_count;
  int64_t* ptr_d_data = lda_data + 5 * problem_count;
  cutlass::gemm::GemmCoord* problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(lda_data + 6 * problem_count);

  // Set arguments into gmm_args from input args
  for (int i = 0; i < problem_count; ++i) {
    problem_sizes_data[i] = gemm_sizes[i];
    lda_data[i] = lda[i];
    ldb_data[i] = ldb[i];
    ldd_data[i] = ldd[i];
    ptr_a_data[i] = reinterpret_cast<int64_t>(aptr[i]);
    ptr_b_data[i] = reinterpret_cast<int64_t>(bptr[i]);
    ptr_d_data[i] = reinterpret_cast<int64_t>(dptr[i]);
  }
  const int threadblock_count =
      GemmGrouped::sufficient(problem_sizes_data, problem_count);

  // Transfer arguments to GPU
  gmm_args = gmm_args.to(device, true);

  // Obtain pointers for each of arguments (on GPU)
  lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  ldb_data = lda_data + problem_count;
  ldd_data = lda_data + 2 * problem_count;
  ptr_a_data = lda_data + 3 * problem_count;
  ptr_b_data = lda_data + 4 * problem_count;
  ptr_d_data = lda_data + 5 * problem_count;
  problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(lda_data + 6 * problem_count);

  // Create GemmGrouped::Arguments using the arguments prepared above
  typename GemmGrouped::Arguments args(
      problem_sizes_data,
      problem_count,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<Element**>(ptr_a_data),
      reinterpret_cast<Element**>(ptr_b_data),
      reinterpret_cast<Element**>(ptr_d_data),
      reinterpret_cast<Element**>(ptr_d_data),
      lda_data,
      ldb_data,
      ldd_data,
      ldd_data);

  GemmGrouped gemm;
  cutlass::Status status =
      gemm.initialize(args, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status != cutlass::Status::kErrorWorkspaceNull,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to workspace.");
  TORCH_CHECK(
      status != cutlass::Status::kErrorInternal,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to internal error.");
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to initialize CUTLASS Grouped GEMM kernel.");

  // Run CUTLASS group GEMM
  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to run CUTLASS Grouped GEMM kernel.");

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
bool group_gemm_dispatch(
    at::Device device,
    std::vector<scalar_t*> aptr,
    std::vector<scalar_t*> bptr,
    std::vector<scalar_t*> dptr,
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    std::vector<cutlass::gemm::GemmCoord> gemm_sizes,
    int64_t ntensors) {
  return false;
}

template <>
bool group_gemm_dispatch(
    at::Device device,
    std::vector<float*> aptr,
    std::vector<float*> bptr,
    std::vector<float*> dptr,
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    std::vector<cutlass::gemm::GemmCoord> gemm_sizes,
    int64_t ntensors) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;

  if (is_sm8x) {
    gemm_grouped_cuda_internal<
        float,
        1,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 8>,
        cutlass::gemm::GemmShape<64, 32, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>>(
        lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
    return true;
  }

  // Did not perform GEMM
  return false;
}

template <>
bool group_gemm_dispatch(
    at::Device device,
    std::vector<c10::Half*> aptr_,
    std::vector<c10::Half*> bptr_,
    std::vector<c10::Half*> dptr_,
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    std::vector<cutlass::gemm::GemmCoord> gemm_sizes,
    int64_t ntensors) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;

  // Check alignment
  bool all_pad_8 = true;
  for (int i = 0; i < ntensors; i++) {
    all_pad_8 = all_pad_8 && (gemm_sizes[i].n() % 8 == 0);
    all_pad_8 = all_pad_8 && (gemm_sizes[i].k() % 8 == 0);

    // Not sure if this is a requirement, on the safe side
    all_pad_8 = all_pad_8 && (lda[i] % 8 == 0);
    all_pad_8 = all_pad_8 && (ldb[i] % 8 == 0);
    all_pad_8 = all_pad_8 && (ldd[i] % 8 == 0);
  }

  std::vector<cutlass::half_t*> aptr;
  std::vector<cutlass::half_t*> bptr;
  std::vector<cutlass::half_t*> dptr;
  for (int64_t i = 0; i < ntensors; i++) {
    aptr.push_back(reinterpret_cast<cutlass::half_t*>(aptr_[i]));
    bptr.push_back(reinterpret_cast<cutlass::half_t*>(bptr_[i]));
    dptr.push_back(reinterpret_cast<cutlass::half_t*>(dptr_[i]));
  }
  if (is_sm8x) {
    if (all_pad_8) {
      gemm_grouped_cuda_internal<
          cutlass::half_t,
          8,
          cutlass::layout::RowMajor,
          cutlass::layout::RowMajor,
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm80,
          cutlass::gemm::GemmShape<128, 128, 32>,
          cutlass::gemm::GemmShape<64, 64, 32>,
          cutlass::gemm::GemmShape<16, 8, 16>>(
          lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
      return true;
    } else {
      gemm_grouped_cuda_internal<
          cutlass::half_t,
          1,
          cutlass::layout::RowMajor,
          cutlass::layout::RowMajor,
          cutlass::arch::OpClassSimt,
          cutlass::arch::Sm80,
          cutlass::gemm::GemmShape<128, 128, 8>,
          cutlass::gemm::GemmShape<64, 32, 8>,
          cutlass::gemm::GemmShape<1, 1, 1>>(
          lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
      return true;
    }
  }
  // Did not perform GEMM
  return false;
}
#endif
#endif

} // namespace

Tensor bmm_nested_cuda(const Tensor& self, const Tensor& mat2) {
  if (self.is_nested() && !mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a nested self and non-nested other");
  } else if (!self.is_nested() && mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a non-nested self and nested other");
  }
  // dispatcher should have guaranteed that at least one is nested
  auto self_ptr = get_nested_tensor_impl(self);
  auto mat2_ptr = get_nested_tensor_impl(mat2);
  TORCH_CHECK(self_ptr->dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2_ptr->dim() == 3, "batch2 must be a 3D tensor");
  int64_t ntensors = self_ptr->size(0), ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(
      ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ",
      ntensors,
      " but got: ",
      ntensors2,
      ".");

  // create a contiguous output
  const Tensor& self_sizemat = self_ptr->get_nested_size_tensor();
  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();
  std::vector<int64_t> output_offsets;

  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr);
  std::vector<IntArrayRef> mat2_sizes = NestedTensor_get_sizes(mat2_ptr);

  int64_t out_numel = 0;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef &self_shape = self_sizes[i], &mat2_shape = mat2_sizes[i];
    const int64_t &self_size0 = self_shape[0], &self_size1 = self_shape[1],
                  &mat2_size0 = mat2_shape[0], &mat2_size1 = mat2_shape[1];
    TORCH_CHECK(
        self_size1 == mat2_size0,
        i,
        "-th nested matrices in batch cannot be multiplied (",
        self_size0,
        "x",
        self_size1,
        " and ",
        mat2_size0,
        "x",
        mat2_size1,
        ")");
    out_sizemat_ptr[0] = self_size0;
    out_sizemat_ptr[1] = mat2_size1;
    out_sizemat_ptr += 2;
    output_offsets.push_back(out_numel);
    out_numel += self_size0 * mat2_size1;
  }
  const Tensor &self_buffer = self_ptr->get_unsafe_storage_as_tensor();
  const Tensor &mat2_buffer = mat2_ptr->get_unsafe_storage_as_tensor();
  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = wrap_buffer(out_buffer, out_sizemat);
  auto out_ptr = get_nested_tensor_impl(output);

#ifndef USE_ROCM
#ifndef _WIN32
  std::vector<IntArrayRef> self_strides = NestedTensor_get_strides(self_ptr);
  std::vector<IntArrayRef> mat2_strides = NestedTensor_get_strides(mat2_ptr);
  const std::vector<int64_t>& self_offsets = self_ptr->get_storage_offsets();
  const std::vector<int64_t>& mat2_offsets = mat2_ptr->get_storage_offsets();
  const std::vector<int64_t>& out_offsets = out_ptr->get_storage_offsets();

  bool success = false;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "group_gemm_dispatch", [&] {
        std::vector<scalar_t*> aptr;
        std::vector<scalar_t*> bptr;
        std::vector<scalar_t*> dptr;
        std::vector<int64_t> lda;
        std::vector<int64_t> ldb;
        std::vector<int64_t> ldd;
        std::vector<cutlass::gemm::GemmCoord> gemm_sizes;
        bool all_row_major = true;
        for (int64_t i = 0; i < ntensors; i++) {
          const IntArrayRef& self_shape = self_sizes[i];
          const IntArrayRef& mat2_shape = mat2_sizes[i];
          const int64_t &self_size0 = self_shape[0];
          const int64_t &self_size1 = self_shape[1];
          const int64_t &mat2_size0 = mat2_shape[0];
          const int64_t &mat2_size1 = mat2_shape[1];
          gemm_sizes.push_back(
              cutlass::gemm::GemmCoord(self_size0, mat2_size1, self_size1));
          aptr.push_back(self_buffer.data_ptr<scalar_t>() + self_offsets[i]);
          bptr.push_back(mat2_buffer.data_ptr<scalar_t>() + mat2_offsets[i]);
          dptr.push_back( out_buffer.data_ptr<scalar_t>() + out_offsets[i]);
          all_row_major = all_row_major && (self_strides[i][1] == 1);
          all_row_major = all_row_major && (mat2_strides[i][1] == 1);
          lda.push_back(self_strides[i][0]);
          ldb.push_back(mat2_strides[i][0]);
          ldd.push_back(mat2_size1);
        }
        if (all_row_major) {
          success = group_gemm_dispatch<scalar_t>(
              output.device(),
              aptr,
              bptr,
              dptr,
              lda,
              ldb,
              ldd,
              gemm_sizes,
              ntensors);
        }
      });
  if (success) {
    return output;
  }
#endif
#endif
  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(
        output_unbind[i],
        self_buffer.as_strided(self_sizes[i], self_strides[i], self_offsets[i]),
        mat2_buffer.as_strided(
            mat2_sizes[i], mat2_strides[i], mat2_offsets[i]));
  }
  return output;
}

} // namespace native
} // namespace at
