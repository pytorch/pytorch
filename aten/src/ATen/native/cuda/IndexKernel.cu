#include <ATen/native/TensorAdvancedIndexing.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/core/Array.h>

namespace at { namespace native {

template <int N>
static OffsetCalculator<N> index_make_offset_calculator(const TensorIterator& iter) {
  AT_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data());
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

  auto offset_calc = index_make_offset_calculator<3>(iter);
  legacy::launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int idx) {
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


static void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  AT_ASSERTM(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "index_put", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_put_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);

}} // namespace at::native
