#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/IndexKernel.h>
#include <ATen/native/IndexKernel.h>

#include <type_traits>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/quantized/IndexKernel.h>

#include <c10/core/Scalar.h>

namespace at::native {

static constexpr int launch_bound2 = 4;

static constexpr int launch_size_nd = 128;

template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel(const int64_t N, const func_t f) {
  const auto tid = threadIdx.x;
  const auto nv = nt * vt;
  auto idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_kernel(const int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  const auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_index_kernel(TensorIteratorBase& iter, const IntArrayRef index_size, const IntArrayRef index_stride, const func_t& f) {
  const auto num_indices = index_size.size();
  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(static_cast<int64_t>(num_indices) == iter.ntensors() - 2);

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_index_kernel(sub_iter, index_size, index_stride, f);
    }
    return;
  }

  auto sizes = at::detail::Array<int64_t, MAX_DIMS>(0);
  auto strides = at::detail::Array<int64_t, MAX_DIMS>(0);
  auto index_ptrs = at::detail::Array<char*, MAX_DIMS>(nullptr);
  for (unsigned i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  char* const out_ptr = static_cast<char*>(iter.data_ptr(0));
  char* const in_ptr = static_cast<char*>(iter.data_ptr(1));

  auto offset_calc = make_offset_calculator<3>(iter);
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int idx) {
    const auto offsets = offset_calc.get(idx);
    char* const out_data = out_ptr + offsets[0];
    const char* const in_data = in_ptr + offsets[1];

    int64_t offset = 0;
    #pragma unroll
    for (int i = 0; i < num_indices; i++) {
      int64_t index = *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
      CUDA_KERNEL_ASSERT(-sizes[i] <= index && index < sizes[i] && "index out of bounds");
      if (index < 0) {
        index += sizes[i];
      }
      offset += index * strides[i];
    }

    f(out_data, in_data, offset);
  });
}

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <int N> struct alignas(N) OpaqueType { char data[N]; };

template <typename scalar_t>
void index_fill_kernel_impl(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride,
  const scalar_t fill_val) {
  if (0 == iter.numel()) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_fill_kernel_impl(sub_iter, dim, self_dim_size, self_dim_stride, fill_val);
    }
    return;
  }

  char* const __restrict__ self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* const __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2>(iter);

  const auto loop = [=]C10_DEVICE(int i) {
    const auto offsets = offset_calc.get(i);

    auto* __restrict__ self_data = reinterpret_cast<scalar_t*>(self_ptr + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    CUDA_KERNEL_ASSERT(idx >= -self_dim_size && idx < self_dim_size && "index out of bounds");
    if (idx < 0) {
      idx += self_dim_size;
    }

    self_data[idx * self_dim_stride] = fill_val;
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}

template <typename scalar_t>
void index_copy_kernel_impl(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_copy_kernel_impl<scalar_t>(sub_iter, dim, self_dim_size, self_dim_stride);
    }
    return;
  }

  char* const __restrict__ self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* const __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* const __restrict__ source_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  const auto offset_calc = make_offset_calculator<3>(iter);

  const auto loop = [=]C10_DEVICE(int i) {
    const auto offsets = offset_calc.get(i);

    auto* const __restrict__ self_data = reinterpret_cast<scalar_t*>(self_ptr + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    const auto* const __restrict__ source_data = reinterpret_cast<scalar_t*>(source_ptr + offsets[2]);
    CUDA_KERNEL_ASSERT(idx >= 0 && idx < self_dim_size && "index_copy_(): index out of bounds");

    self_data[idx * self_dim_stride] = *source_data;
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}

template <typename scalar_t>
void index_kernel_impl(TensorIteratorBase& iter, const IntArrayRef index_size, const IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* const out_data, const char* const in_data, const int64_t offset) {
    *reinterpret_cast<scalar_t*>(out_data) = *reinterpret_cast<const scalar_t*>(in_data + offset);
  });
}

template <typename scalar_t>
void index_put_kernel_impl(TensorIterator& iter, const IntArrayRef index_size, const IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* const out_data, const char* const in_data, const int64_t offset) {
    *reinterpret_cast<scalar_t*>(out_data + offset) = *reinterpret_cast<const scalar_t*>(in_data);
  });
}

static void index_kernel(TensorIteratorBase& iter, const IntArrayRef index_size, const IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16, iter.dtype(), "index_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

static void index_fill_kernel(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride,
  const Scalar& source) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, kComplexHalf,
    iter.dtype(), "index_fill_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    const auto fill_val = source.to<scalar_t>();
    const auto fill_val_opaque = *reinterpret_cast<const dtype*>(&fill_val);
    index_fill_kernel_impl<dtype>(iter, dim, self_dim_size, self_dim_stride, fill_val_opaque);
  });
}

static void index_copy_kernel(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries
  // this kernel will not be called when torch.use_deterministic_algorithms(True)
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, kComplexHalf,
    iter.dtype(), "index_copy_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_copy_kernel_impl<dtype>(iter, dim, self_dim_size, self_dim_stride);
  });
}


static void index_put_kernel(TensorIterator& iter, const IntArrayRef index_size, const IntArrayRef index_stride, const bool accumulate) {
  TORCH_CHECK(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16, iter.dtype(), "index_put", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_put_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

void index_put_kernel_quantized_cuda(TensorIterator& iter, const IntArrayRef index_size, const IntArrayRef index_stride, const bool accumulate, const double scale, const int zero_point) {
  TORCH_CHECK(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(iter.dtype(), "index_put", [&] {
    constexpr int64_t qmin = std::numeric_limits<typename scalar_t::underlying>::min();
    constexpr int64_t qmax = std::numeric_limits<typename scalar_t::underlying>::max();
    const float inv_scale = 1.0f / static_cast<float>(scale);

    gpu_index_kernel(iter, index_size, index_stride, [inv_scale, zero_point, qmin, qmax]C10_DEVICE(char* const out_data, const char* const in_data, const int64_t offset) {
      int64_t qvalue = static_cast<int64_t>(zero_point + nearbyintf(*(float*)in_data * inv_scale));
      qvalue = std::clamp(qvalue, qmin, qmax);
      *(scalar_t*)(out_data + offset) = static_cast<scalar_t>(qvalue);
    });
  });
}

template <typename scalar_t, typename index_t, typename func_t>
void cuda_take_put_kernel(
  TensorIterator& iter,
  const TensorBase& indexed,
  const func_t& f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      cuda_take_put_kernel<scalar_t, index_t>(sub_iter, indexed, f);
    }
    return;
  }

  const auto numel = indexed.numel();
  const bool is_contiguous = indexed.is_contiguous();

  char* const __restrict__ iterated_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* const __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2>(iter);
  using uindex_t = std::make_unsigned_t<index_t>;

  // OffsetCalculator needs the sizes and strides reveresed
  const auto indexed_sizes = std::vector<int64_t>(indexed.sizes().rbegin(), indexed.sizes().rend());
  const auto indexed_strides = std::vector<int64_t>(indexed.strides().rbegin(), indexed.strides().rend());
  const auto* indexed_strides_data = indexed_strides.data();
  const auto offset_indexed = OffsetCalculator<1, uindex_t>(indexed.dim(),
                                                            indexed_sizes.data(),
                                                            &indexed_strides_data);

  const auto loop = [=]C10_DEVICE(int i) {
    const auto offsets = offset_calc.get(i);

    auto& iterated = *reinterpret_cast<scalar_t*>(iterated_ptr + offsets[0]);
    const auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    CUDA_KERNEL_ASSERT(idx < numel && idx >= -numel && "cuda_take_put_kernel() index out of bounds");
    index_t offset = static_cast<index_t>(idx);
    if (offset < 0) {
      offset += numel;
    }
    if (!is_contiguous) {
      offset = offset_indexed.get(offset)[0];
    }

    f(iterated, offset);
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}

void put_kernel(TensorIterator& iter, const TensorBase& output, const bool accumulate) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "put_cuda", [&] {
    // Cannot use `OpaqueType`, as we need the actual type for `fastSpecializedgpuAtomicAdd`
    AT_DISPATCH_INDEX_TYPES(cuda::detail::canUse32BitIndexMath(output) ? ScalarType::Int : ScalarType::Long,
        "put_cuda_index", [&] {
           auto* __restrict__ indexed_ptr = output.template data_ptr<scalar_t>();
           if (accumulate) {
             index_t numel = output.numel();
             cuda_take_put_kernel<scalar_t, index_t>(iter, output,
                 [numel, indexed_ptr] __device__(scalar_t& iterated, const index_t offset) {
                   fastSpecializedAtomicAdd(indexed_ptr, offset, numel, iterated);
                 });
           }
           else {
             cuda_take_put_kernel<scalar_t, index_t>(iter, output,
                 [indexed_ptr] __device__(scalar_t& iterated, const index_t offset) {
                   indexed_ptr[offset] = iterated;
                 });
           }
    });
  });
}

void take_kernel(
  TensorIterator& iter,
  const TensorBase& input) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "take_cuda", [&] {
    // Cannot use `OpaqueType`, as Tensor::data_ptr<OpaqueType<N>> is not implemented
    AT_DISPATCH_INDEX_TYPES(cuda::detail::canUse32BitIndexMath(input) ? ScalarType::Int : ScalarType::Long,
      "take_cuda_index", [&] {
         const auto* __restrict__ indexed_ptr = input.template data_ptr<scalar_t>();
         cuda_take_put_kernel<scalar_t, index_t>(iter, input,
            [indexed_ptr] __device__(scalar_t& iterated, const index_t offset) {
               iterated = indexed_ptr[offset];
             });
     });
  });
}

namespace {

__global__ void masked_scatter_size_check(const int64_t *mask_exclusive_sum, const bool *mask, int64_t srcSize) {
  // Convert exclusive sum to inclusive sum
  auto totalElements = *mask_exclusive_sum + *mask;
  CUDA_KERNEL_ASSERT(totalElements <= srcSize);
}

} // anonymous namespace

void launch_masked_scatter_kernel(
    const TensorBase &self, const TensorBase &mask,
    const TensorBase &maskPrefixSum, const TensorBase &source) {
  const auto srcSize = source.numel();
  const auto mask_cont = mask.contiguous();
  const auto mask_numel = mask.numel();

  // Use a prefix sum to determine the output locations of the masked elements
  auto maskPrefixSum_data = maskPrefixSum.mutable_data_ptr<int64_t>();
  auto mask_data = mask_cont.const_data_ptr<bool>();

  at::cuda::cub::mask_exclusive_sum(
      mask_data, maskPrefixSum_data, mask_numel);

  // Asynchronously check that the number of `1` elements present in the mask
  // must be <= the number of elements available in `src`.
  masked_scatter_size_check<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      &maskPrefixSum_data[mask_numel - 1], &mask_data[mask_numel - 1], srcSize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  auto source_contig = source.contiguous();

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_input(mask_cont)
      .add_input(maskPrefixSum)
      .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      self.scalar_type(),
      "masked_scatter_",
      [&]() {
        auto source_ptr = source_contig.const_data_ptr<scalar_t>();
        gpu_kernel(
            iter, [=] GPU_LAMBDA(const scalar_t a, const bool mask, const int64_t maskPrefixSum) -> scalar_t {
              if (mask) {
                return source_ptr[maskPrefixSum];
              }
              return a;
            });
        AT_CUDA_CHECK(cudaGetLastError());
      });
}

template <typename scalar_t>
void flip_kernel_impl(TensorIterator& iter) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      flip_kernel_impl<scalar_t>(sub_iter);
    }
    return;
  }

  char* const __restrict__ out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const __restrict__ in_ptr = reinterpret_cast<const char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2, /*signed_strides=*/true>(iter);

  const auto loop = [=]C10_DEVICE(const int i) {
    const auto offsets = offset_calc.get(i);
    // offsets can be negative here, but it's fine
    scalar_t* const __restrict__ out_data = reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
    const scalar_t* const __restrict__ in_data = reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
    *out_data = *in_data;
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}

void flip_kernel(TensorIterator& iter, const bool quantized) {
  if (quantized) {
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(iter.dtype(), "flip_quantized_cuda",
    [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      flip_kernel_impl<dtype>(iter);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
                                           iter.dtype(), "flip_cuda",
    [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      flip_kernel_impl<dtype>(iter);
    });
  }
}


REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_fill_stub, &index_fill_kernel);
REGISTER_DISPATCH(index_copy_stub, &index_copy_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
REGISTER_DISPATCH(put_stub, &put_kernel);
REGISTER_DISPATCH(take_stub, &take_kernel);
REGISTER_DISPATCH(flip_stub, &flip_kernel);

REGISTER_CUDA_DISPATCH(index_put_kernel_quantized_stub, &index_put_kernel_quantized_cuda);

} // namespace at::native
