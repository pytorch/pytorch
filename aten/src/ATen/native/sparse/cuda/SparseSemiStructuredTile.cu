#include <ATen/ScalarOps.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
#include <ATen/native/sparse/cuda/ComputeSparseTile.h>
#include <ATen/native/sparse/cuda/SparseSemiStructuredPack.h>
#include <cuda_runtime.h>
#endif

namespace at::native {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
struct MetadataCuSparseLt {
  // Format used by cuSparseLt
  // This is based on reverse-engineering, for a visual illustration:
  // https://docs.google.com/presentation/d/1DtmKThv8S5QAyBktuLRYzZhRzCvS1qSkBbrqNCjMPeA/edit#slide=id.g29afe95bda8_0_0
  static constexpr int kStrideBlock32x32 = (32 * 32) / (sizeof(ElementInputE) * 8);

  ElementInputE* _meta;
  ElementInputE* _meta_trans;
  int64_t _rows;
  int64_t _cols;

  static int64_t getMetadataSize(int rows, int cols)
  {
    TORCH_CHECK(rows % 128 == 0 && cols % 128 == 0, "Only supports rows/cols multiples of 128");
    // 1 bit per dense value
    return (rows * cols) / (8 * sizeof(ElementInputE));
  }

  // < return value of the function, packed, packed_meta >
  static std::tuple<Tensor, Tensor, Tensor> create_compressed_representation(int rows, int cols, at::Tensor const& like)
  {
    TORCH_CHECK(
        like.scalar_type() == at::ScalarType::Half ||
        like.scalar_type() == at::ScalarType::BFloat16);
    constexpr int kBytesPerScalar = 2;
    int64_t data_scalars = rows * cutlass::ceil_div(cols, 2);
    int64_t meta_scalars = getMetadataSize(rows, cols);

    at::Tensor storage = at::empty(
        {(data_scalars + meta_scalars)},
        at::TensorOptions().device(like.device()).dtype(like.dtype()));

    using at::indexing::Slice;
    using at::indexing::None;
    at::Tensor packed = storage.index({Slice(None, data_scalars)})
                            .view({rows, cutlass::ceil_div(cols, 2)});
    at::Tensor metadata = storage.index({Slice(data_scalars, None)});
    // TODO: Cast metadata to Short
    static_assert(kBytesPerScalar == 2, "or modify the last dim below");
    metadata = metadata.view({rows / 128, cols / 32, 256});
    return std::make_tuple(storage, packed, metadata);
  }

  MetadataCuSparseLt(at::Tensor metaN, at::Tensor metaT, int rows, int cols) {
    _meta = (ElementInputE*)metaN.data_ptr();
    _meta_trans = (ElementInputE*)metaT.data_ptr();
    _rows = rows;
    _cols = cols;
  }
  CUTLASS_HOST_DEVICE
  static int64_t _get_meta_offset(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col,
      int totalRows) {
    int64_t offset = 0;
    // warp-level: Find the 128x64 tile
    offset += (warp_row / 128) * (kStrideBlock32x32 * 8);
    offset += (warp_col / 64) * (kStrideBlock32x32 * 8) * (totalRows / 128);
    // Find the 32x32 tile inside
    offset += (((warp_row + thread_row) % 128) / 32) * kStrideBlock32x32;
    offset += (((warp_col + thread_col) % 64) / 32) * (kStrideBlock32x32 * 4);
    // Inside the 32x32 tile
    offset += (warp_row % 32) * 2;
    // Top/bottom 16x16 tile
    offset += ((thread_row % 32) / 16) * 4;
    // Left/right 16x16 tile
    offset += ((thread_col % 32) / 16) * 2;
    return offset;
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaN(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta +
        _get_meta_offset(warp_row, thread_row, warp_col, thread_col, _rows);
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaT(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta_trans +
        _get_meta_offset(warp_col, thread_col, warp_row, thread_row, _cols);
  }
};

struct MetadataCutlass {
  // Layout needed to run 2:4 gemms in CUTLASS
  // There is basically a hardware specific value for every
  // 32x32 dense tile (1024 bits). Then these tiles are
  // stored in a Column-Major fashion
  ElementInputE* _meta;
  ElementInputE* _meta_trans;
  int64_t _meta_reordered_sy;
  int64_t _meta_trans_reordered_sx;

  static std::tuple<
      at::Tensor, // return value of the function
      at::Tensor, // packed
      at::Tensor // packed_meta
      >
  create_compressed_representation(int rows, int cols, at::Tensor const& like) {
    TORCH_CHECK(
        like.scalar_type() == at::ScalarType::Half ||
        like.scalar_type() == at::ScalarType::BFloat16);
    auto roundedx = cutlass::round_up(rows, kWarpX);
    auto roundedy = cutlass::round_up(cols, kWarpY);

    // NB: Writing to `packed` tensors in transposed manner
    at::Tensor packed =
        at::empty({roundedx, cutlass::ceil_div(roundedy, 2)}, like.options());
    at::Tensor packed_meta = at::empty(
                                 {roundedx * roundedy / 16},
                                 like.options().dtype(at::ScalarType::Short))
                                 .view({roundedy / 32, roundedx, 2})
                                 .permute({1, 2, 0});
    return std::make_tuple(packed, packed, packed_meta);
  }
  MetadataCutlass(at::Tensor metaN, at::Tensor metaT, int rows, int cols) {
    _meta = (ElementInputE*)metaN.data_ptr();
    _meta_reordered_sy = metaN.stride(2);
    _meta_trans = (ElementInputE*)metaT.data_ptr();
    _meta_trans_reordered_sx = metaT.stride(2);
  }
  CUTLASS_HOST_DEVICE
  int64_t _get_meta_offset(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col,
      int64_t stride) const {
    int64_t offset = 0;
    offset += warp_row * 2 + (warp_col / 32) * stride;
    // A single warp is 32x64. The right 32x32 tile is at a different position
    offset += 64 * (thread_row / 32);
    offset += (thread_col / 32) * stride;
    // Top/bottom 16x16 tile
    offset += ((thread_row % 32) / 16) * 4;
    // Left/right 16x16 tile
    offset += ((thread_col % 32) / 16) * 2;
    return offset;
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaN(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta +
        _get_meta_offset(
               warp_row, thread_row, warp_col, thread_col, _meta_reordered_sy);
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaT(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta_trans +
        _get_meta_offset(
               warp_col,
               thread_col,
               warp_row,
               thread_row,
               _meta_trans_reordered_sx);
  }
};

template <typename KT, typename Metadata, typename Algorithm>
__global__ void __launch_bounds__(32 /* num_threads */, 20)
    sparse_semi_structured_tile_kernel(
        typename KT::Params p,
        Metadata metadata,
        Algorithm algo) {
  KT::sparse_semi_structured_tile_kernel(p, metadata, algo);
}

template <typename Element, typename MetadataFormat>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> sparse_semi_structured_tile_typed(
        const at::Tensor input,
        std::string algorithm)
{
  using KT = KernelTypes<Element>;
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!input.is_meta()) {
    device_guard.emplace(input.device());
  }

  TORCH_CHECK(input.dim() == 2, "Can only sparsify 2d tensors");
  TORCH_CHECK(
      input.stride(1) == 1,
      "Can only sparsify contiguous tensors. Sparsify the transpose otherwise.");

  auto rows = input.size(0);
  auto cols = input.size(1);

  auto [compressed, packed, packed_meta_reordered] =
      MetadataFormat::create_compressed_representation(rows, cols, input);
  auto [compressed_trans, packed_trans, packed_trans_meta_reordered] =
      MetadataFormat::create_compressed_representation(cols, rows, input);
  TORCH_CHECK(
      input.size(1) % 32 == 0, "Number of cols should be multiple of 32");

  typename KT::Params p;
  p.input = (Element const*)input.data_ptr();
  p.input_s0 = input.stride(0);
  p.input_dim0 = input.size(0);
  p.input_dim1 = input.size(1);

  p.packed = (Element*)packed.data_ptr();
  p.packed_stride = packed.stride(0);
  p.packed_trans = (Element*)packed_trans.data_ptr();
  p.packed_trans_stride = packed_trans.stride(0);

  MetadataFormat metadata = MetadataFormat(
      packed_meta_reordered, packed_trans_meta_reordered, rows, cols);
  at::Tensor threads_masks = at::empty(
      {p.getBlocksGrid().x * p.getThreadsGrid().x,
       p.getBlocksGrid().y * p.getThreadsGrid().y,
       sizeof(p.threads_masks[0])},
      input.options().dtype(at::ScalarType::Byte));
  p.threads_masks = (uint64_t*)threads_masks.data_ptr();

  bool kernel_launched = false;
  auto launchKernel = [&](auto algo, std::string const& algo_name) {
    if (algo_name == algorithm) {
      kernel_launched = true;
      if (input.is_meta()) {
        return;
      }
      size_t smem_bytes = 0;
      sparse_semi_structured_tile_kernel<KT>
          <<<p.getBlocksGrid(),
             p.getThreadsGrid(),
             smem_bytes,
             at::cuda::getCurrentCUDAStream()>>>(p, metadata, algo);
    }
  };
  named_algorithms(launchKernel);
  TORCH_CHECK(kernel_launched, "Unknown algorithm \"", algorithm, "\"");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      compressed,
      packed_meta_reordered,
      compressed_trans,
      packed_trans_meta_reordered,
      threads_masks);
}
#endif

// <packed, packed_meta_reordered, packed_trans, packed_trans_meta_reorderd, threads_masks>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _sparse_semi_structured_tile(
  const Tensor& input,
  std::string_view algorithm,
  bool use_cutlass)
{
#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
  TORCH_CHECK(false, "_sparse_semi_structured_tile: not supported");
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{});
#else
  std::string algo(algorithm.data(), algorithm.size());

  auto runTyped = [&](auto type)
  {
    using ElementT = decltype(type);
    if (use_cutlass) {
      return sparse_semi_structured_tile_typed<ElementT, MetadataCutlass>(input, algo);
    }
    else {
      return sparse_semi_structured_tile_typed<ElementT, MetadataCuSparseLt>(input, algo);
    }
  };

  if (input.scalar_type() == at::ScalarType::Half)
  {
    return runTyped(cutlass::half_t());
  } else {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Half ||
        input.scalar_type() == at::ScalarType::BFloat16, input.scalar_type());
    return runTyped(cutlass::bfloat16_t());
  }
#endif
}

} // namespace at::native
