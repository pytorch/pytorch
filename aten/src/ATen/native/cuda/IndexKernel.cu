#include <ATen/native/TensorAdvancedIndexing.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/core/Array.h>
#include <ATen/ExpandUtils.h>

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

template <typename func_t>
void gpu_masked_select_kernel(TensorIterator& iter, const func_t& f) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_masked_select_kernel(sub_iter, f);
    }
    return;
  }

  void* out_ptr = (void*)iter.data_ptr(0);
  void* in_ptr = (void*)iter.data_ptr(1);
  bool* mask_ptr = (bool*)iter.data_ptr(2);
  int64_t* mask_cumsum_ptr = (int64_t*)iter.data_ptr(3);

  legacy::launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int64_t idx) {
    void* out_data = out_ptr;
    void* in_data = in_ptr;
    bool mask = mask_ptr[idx];
    int64_t mask_cumsum = mask_cumsum_ptr[idx];

    f(idx, out_data, in_data, mask, mask_cumsum);
  });
}

template <typename scalar_t>
void masked_select_kernel_impl(TensorIterator& iter) {
  gpu_masked_select_kernel(iter, []C10_DEVICE(int64_t idx, void* out_data, void* in_data, bool mask, int64_t mask_cumsum) {
    if (mask) {
      scalar_t* in_ptr = (scalar_t*) in_data;
      scalar_t* out_ptr = (scalar_t*) out_data;
      out_ptr[mask_cumsum-1] = in_ptr[idx];
    }
  });
}

static void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "index_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}


static void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  AT_ASSERTM(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "index_put", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_put_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

static void masked_select_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "masked_select", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    masked_select_kernel_impl<dtype>(iter);
  });
}


static Tensor & masked_select_out_cuda_impl(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor or ByteTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  Tensor _mask, _self;
  std::tie(_mask, _self) = expand_outplace(mask, self);

  auto shape = _self.sizes().vec();

  Tensor mask_cumsum_flat = _mask.flatten().cumsum(0, c10::ScalarType::Long);
  int64_t mask_cumsum_flat_last_idx = mask_cumsum_flat.size(0)-1;

  int64_t* numel_ptr_dev = mask_cumsum_flat[mask_cumsum_flat_last_idx].data_ptr<int64_t>();
  int64_t numel;

  cudaMemcpy(&numel,numel_ptr_dev,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  Tensor mask_cumsum = mask_cumsum_flat.reshape(shape);

  result.resize_({numel});
  if (numel == 0) {
    return result;
  }

  // Create strided view of result before feeding into TensorIterator
  auto strides = DimVector(shape.size(), 0);
  auto result_strided = result.as_strided(shape, strides);

  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.dont_resize_outputs();
  iter.add_output(result_strided);
  iter.add_input(_self);
  iter.add_input(_mask);
  iter.add_input(mask_cumsum);
  iter.build();

  masked_select_kernel(iter);
  return result;
}


Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_cuda_impl(result, self, mask);
}

Tensor & masked_select_out_cuda(Tensor & result, const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_cuda_impl(result, self, mask);
}

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);

}} // namespace at::native
