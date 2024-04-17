#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
#include <ATen/native/sparse/cuda/ComputeSparseTile.h>
#include <ATen/native/sparse/cuda/SparseSemiStructuredPack.h>
#endif

namespace at::native {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
struct Params {
  uint64_t const* threads_masks;

  uint16_t const* input;
  int64_t input_stride;
  int64_t input_dim0;
  int64_t input_dim1;

  uint16_t* output;
  int64_t output_stride;

  __host__ dim3 getBlocksGrid() const {
    return dim3(
        cutlass::ceil_div(input_dim0, kWarpX),
        cutlass::ceil_div(input_dim1, kWarpY),
        1);
  }

  static CUTLASS_HOST_DEVICE dim3 getThreadsGrid() {
    return dim3(kWarpX / kThreadX, kWarpY / kThreadY, 1);
  }

  CUTLASS_DEVICE Tile8x8Masks* getCurrentThreadIndices() const {
    Tile8x8Masks* gmem_threads_masks = (Tile8x8Masks*)threads_masks;
    gmem_threads_masks += blockIdx.y * getThreadsGrid().y + threadIdx.y;
    int64_t strideX = gridDim.y * getThreadsGrid().y;
    gmem_threads_masks +=
        (blockIdx.x * getThreadsGrid().x + threadIdx.x) * strideX;
    return gmem_threads_masks;
  }
};

template <bool kInputRowMajor = true, bool kOutputRowMajor = true>
__global__ void __launch_bounds__(32 /* num_threads */, 32) sparse_semi_structured_apply_dense_k(Params p) {
  using Fragment = cutlass::Array<uint16_t, 8>;

  // Top-left of the 8x8 tile we own
  int warp_x = blockIdx.x * kWarpX;
  int warp_y = blockIdx.y * kWarpY;
  int x = warp_x + threadIdx.x * kThreadX;
  int y = warp_y + threadIdx.y * kThreadY;

  uint16_t* output = p.output + x * p.output_stride + y;
  Tile8x8Masks indices = *p.getCurrentThreadIndices();

  // Load dense
  Fragment lines[8];
  if (kInputRowMajor) {
    uint16_t const* input = p.input + x * p.input_stride + y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          lines[i], input + i * p.input_stride, true);
    }
  } else {
    uint16_t const* input = p.input + x + y * p.input_stride;
    Fragment columns[8];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          columns[i], input + i * p.input_stride, true);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        lines[i][j] = columns[j][i].get();
      }
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int row = 0; row < 2; ++row) {
    Indices4x4 masks[2];
    if (row == 0) {
      masks[0] = indices.a;
      masks[1] = indices.b;
    } else {
      masks[0] = indices.c;
      masks[1] = indices.d;
    }

    // Apply mask
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 4; ++r) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) {
          lines[4 * row + r][4 * m + c] = lines[4 * row + r][4 * m + c] *
              int((masks[m] >> (4 * r + c)) & 1);
        }
      }
    }
  }
  static_assert(kOutputRowMajor, "Transpose here for ColMajor output");
  // Save dense with zeros
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 8; ++i) {
    cutlass::arch::global_store<Fragment, sizeof(Fragment)>(
        lines[i], output + i * p.output_stride, true);
  }
}
#endif

Tensor _sparse_semi_structured_apply_dense(
    const Tensor& input,
    const Tensor& threads_masks) {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
  AT_ERROR("_sparse_semi_structured_apply_dense: not supported");
  return Tensor{};
#else
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "Unsupported `input` dtype");
  TORCH_CHECK(
      input.stride(0) == 1 || input.stride(1) == 1,
      "`input` should be either RowMajor or ColMajor. Invalid memory layout - try .contiguous()?");

  auto roundedx = cutlass::round_up(input.size(0), kWarpX);
  auto roundedy = cutlass::round_up(input.size(1), kWarpY);

  Params p;
  p.input = (uint16_t const*)input.data_ptr();
  p.input_dim0 = input.size(0);
  p.input_dim1 = input.size(1);
  p.threads_masks = (uint64_t const*)threads_masks.data_ptr();

  TORCH_CHECK(threads_masks.dim() == 3);
  TORCH_CHECK(threads_masks.size(0) == p.getBlocksGrid().x * p.getThreadsGrid().x);
  TORCH_CHECK(threads_masks.size(1) == p.getBlocksGrid().y * p.getThreadsGrid().y);
  TORCH_CHECK(threads_masks.stride(1) == sizeof(p.threads_masks[0]));
  TORCH_CHECK(threads_masks.size(2) == sizeof(p.threads_masks[0]));
  TORCH_CHECK(threads_masks.stride(2) == 1);
  TORCH_CHECK(threads_masks.scalar_type() == at::ScalarType::Byte);

  at::Tensor output = at::empty({p.input_dim0, p.input_dim1}, input.options());
  TORCH_INTERNAL_ASSERT(output.stride(-1) == 1, "expected RowMajor?");
  p.output = (uint16_t*)output.data_ptr();

  bool inputRowMajor = input.stride(-1) == 1;
  bool outputRowMajor = output.stride(-1) == 1;
  p.input_stride = input.stride(inputRowMajor ? 0 : 1);
  p.output_stride = output.stride(outputRowMajor ? 0 : 1);
  at::cuda::CUDAGuard device_guard(input.device());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  size_t smem_bytes = 0;
  if (inputRowMajor && outputRowMajor) {
    sparse_semi_structured_apply_dense_k<true, true>
        <<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  } else if (!inputRowMajor && outputRowMajor) {
    sparse_semi_structured_apply_dense_k<false, true>
        <<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  } else {
    TORCH_CHECK(
        false,
        "Unsupported configuration: `input` is ",
        inputRowMajor ? "RowMajor" : "ColMajor",
        ", and `output` is ",
        outputRowMajor ? "RowMajor" : "ColMajor");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
#endif
}

} // namespace
