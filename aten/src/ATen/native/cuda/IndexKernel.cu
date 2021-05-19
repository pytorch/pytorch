#include <ATen/native/TensorAdvancedIndexing.h>

#include <type_traits>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/MaybeOwned.h>
#include <THC/THCTensorInfo.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace at { namespace native {

static constexpr int launch_bound2 = 4;

static constexpr int launch_size_nd = 128;

template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, const func_t& f) {
  int num_indices = index_size.size();
  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(num_indices == iter.ntensors() - 2);

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_index_kernel(sub_iter, index_size, index_stride, f);
    }
    return;
  }

  auto sizes = at::detail::Array<int64_t, 25>(0);
  auto strides = at::detail::Array<int64_t, 25>(0);
  auto index_ptrs = at::detail::Array<char*, 25>(nullptr);
  for (int i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  char* out_ptr = (char*)iter.data_ptr(0);
  char* in_ptr = (char*)iter.data_ptr(1);

  auto offset_calc = make_offset_calculator<3>(iter);
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int idx) {
    auto offsets = offset_calc.get(idx);
    char* out_data = out_ptr + offsets[0];
    char* in_data = in_ptr + offsets[1];

    int64_t offset = 0;
    #pragma unroll
    for (int i = 0; i < num_indices; i++) {
      int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
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
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride,
  scalar_t fill_val) {
  if (0 == iter.numel()) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_fill_kernel_impl(sub_iter, dim, self_dim_size, self_dim_stride, fill_val);
    }
    return;
  }

  char* __restrict__ self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  auto offset_calc = make_offset_calculator<2>(iter);

  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calc.get(i);

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
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_copy_kernel_impl<scalar_t>(sub_iter, dim, self_dim_size, self_dim_stride);
    }
    return;
  }

  char* __restrict__ self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* __restrict__ source_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  auto offset_calc = make_offset_calculator<3>(iter);

  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calc.get(i);

    auto* __restrict__ self_data = reinterpret_cast<scalar_t*>(self_ptr + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    auto* __restrict__ source_data = reinterpret_cast<scalar_t*>(source_ptr + offsets[2]);
    CUDA_KERNEL_ASSERT(idx >= 0 && idx < self_dim_size && "index_copy_(): index out of bounds");

    self_data[idx * self_dim_stride] = *source_data;
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}

template <typename scalar_t>
void index_kernel_impl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* out_data, char* in_data, int64_t offset) {
    *(scalar_t*)out_data = *(scalar_t*)(in_data + offset);
  });
}

template <typename scalar_t>
void index_put_kernel_impl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* out_data, char* in_data, int64_t offset) {
    *(scalar_t*)(out_data + offset) = *(scalar_t*)in_data;
  });
}

static void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "index_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

static void index_fill_kernel(
  TensorIterator& iter,
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride,
  const Scalar& source) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(), "index_fill_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    auto fill_val = source.to<scalar_t>();
    auto fill_val_opaque = *reinterpret_cast<dtype*>(&fill_val);
    index_fill_kernel_impl<dtype>(iter, dim, self_dim_size, self_dim_stride, fill_val_opaque);
  });
}

static void index_copy_kernel(
  TensorIterator& iter,
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries
  // this kernel will not be called when torch.use_deterministic_algorithms(True)
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(), "index_copy_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_copy_kernel_impl<dtype>(iter, dim, self_dim_size, self_dim_stride);
  });
}


static void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  TORCH_CHECK(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "index_put", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_put_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

static Tensor & masked_select_out_cuda_impl(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor or ByteTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  auto mask_temp = (mask.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = expand_outplace(*mask_temp, *self_temp);
  at::native::index_out(result, *std::get<1>(mask_self_expanded), c10::List<c10::optional<at::Tensor>>({*std::get<0>(std::move(mask_self_expanded))}));

  return result;
}

Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_cuda_impl(result, self, mask);
}

Tensor & masked_select_out_cuda(const Tensor & self, const Tensor & mask, Tensor & result) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_cuda_impl(result, self, mask);
}

template <typename scalar_t, typename index_t, typename func_t>
void cuda_take_put_kernel(
  TensorIterator& iter,
  const Tensor& indexed,
  const func_t& f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      cuda_take_put_kernel<scalar_t, index_t>(sub_iter, indexed, f);
    }
    return;
  }

  const auto numel = indexed.numel();
  const bool is_contiguous = indexed.is_contiguous();

  char* __restrict__ iterated_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2>(iter);
  using uindex_t = std::make_unsigned_t<index_t>;

  // OffsetCalculator needs the sizes and strides reveresed
  const auto indexed_sizes = std::vector<int64_t>(indexed.sizes().rbegin(), indexed.sizes().rend());
  const auto indexed_strides = std::vector<int64_t>(indexed.strides().rbegin(), indexed.strides().rend());
  const auto* indexed_strides_data = indexed_strides.data();
  const auto offset_indexed = OffsetCalculator<1, uindex_t>(indexed.dim(),
                                                            indexed_sizes.data(),
                                                            &indexed_strides_data);

  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calc.get(i);

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

void put_kernel(TensorIterator& iter, const Tensor& output, const bool accumulate) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "put_cuda", [&] {
    // Cannot use `OpaqueType`, as we need the actual type for `fastSpecializedgpuAtomicAdd`
    AT_DISPATCH_INDEX_TYPES(cuda::detail::canUse32BitIndexMath(output) ? ScalarType::Int : ScalarType::Long,
        "put_cuda_index", [&] {
           auto* __restrict__ indexed_ptr = output.template data<scalar_t>();
           if (accumulate) {
             const auto numel = output.numel();
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
  const Tensor& input) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "take_cuda", [&] {
    // Cannot use `OpaqueType`, as Tensor::data_ptr<OpaqueType<N>> is not implemented
    AT_DISPATCH_INDEX_TYPES(cuda::detail::canUse32BitIndexMath(input) ? ScalarType::Int : ScalarType::Long,
      "take_cuda_index", [&] {
         const auto* __restrict__ indexed_ptr = input.template data<scalar_t>();
         cuda_take_put_kernel<scalar_t, index_t>(iter, input,
            [indexed_ptr] __device__(scalar_t& iterated, const index_t offset) {
               iterated = indexed_ptr[offset];
             });
     });
  });
}

namespace {

template <typename mask_t>
void masked_scatter_cuda_impl(Tensor& self, const Tensor& mask, const Tensor& source){
  auto srcSize = source.numel();

  // Determine our output size
  auto totalElements = mask.sum().item<int64_t>();

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  TORCH_CHECK(totalElements <= srcSize, "source nElements must be == mask `1` elements");

  auto mask_cont = mask.contiguous();

  // Use a prefix sum to determine the output locations of the masked elements
  auto maskPrefixSum = at::empty_like(mask, mask.options().dtype(kLong));

  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());

  thrust::device_ptr<mask_t> maskData(mask_cont.data_ptr<mask_t>());
  thrust::device_ptr<int64_t> maskPrefixSumData(
      maskPrefixSum.data_ptr<int64_t>());

  // Reference for using static_cast on `init_value`:
  // https://github.com/NVIDIA/thrust/issues/1379
  thrust::exclusive_scan(
      thrust::cuda::par(allocator).on(c10::cuda::getCurrentCUDAStream()),
      maskData,
      maskData + mask_cont.numel(),
      maskPrefixSumData,
      static_cast<int64_t>(0));

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
        auto source_ptr = source_contig.data_ptr<scalar_t>();
        gpu_kernel(
            iter, [=] GPU_LAMBDA(scalar_t a, mask_t mask, int64_t maskPrefixSum) -> scalar_t {
              if (mask) {
                return source_ptr[maskPrefixSum];
              }
              return a;
            });
        cudaGetLastError();
      });
}

} // anonymous namespace

Tensor & masked_scatter__cuda(Tensor& self, const Tensor& mask, const Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter: expected self and source to have same dtypes but got",
      self.scalar_type(),
      " and ",
      source.scalar_type());

  TensorArg self_arg{self, "self", 1};
  TensorArg mask_arg{mask, "mask", 2};
  TensorArg source_arg{source, "source", 3};
  checkAllSameGPU(__func__, {self_arg, mask_arg, source_arg});

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_scatter_");

  if (b_mask->dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  auto mask_dtype = b_mask->scalar_type();
  if (mask_dtype == ScalarType::Bool) {
    masked_scatter_cuda_impl<bool>(self, *b_mask, source);
  } else {
    masked_scatter_cuda_impl<uint8_t>(self, *b_mask, source);
  }

  return self;
}

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_fill_stub, &index_fill_kernel);
REGISTER_DISPATCH(index_copy_stub, &index_copy_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
REGISTER_DISPATCH(put_stub, &put_kernel);
REGISTER_DISPATCH(take_stub, &take_kernel);

}} // namespace at::native
