#include <ATen/native/TensorAdvancedIndexing.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>

#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

namespace at { namespace native {

// Implement as functors since lambdas don't get optimized.
class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t * self_data, const scalar_t * src_data) const {
    gpuAtomicMul(self_data, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t * self_data, const scalar_t * src_data) const {
    gpuAtomicAdd(self_data, *src_data);
  }
};
static ReduceAdd reduce_add;

class TensorAssign {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t * self_data, const scalar_t * src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

// The kernels are implemented on an opaque,
// self-aligned type of the correct size,
// to avoid redundant kernels for different types
// of the same size.
template <int N> struct alignas(N) OpaqueType { char data[N]; };

// essentialy rewritten related to legacy::launch_kernel parts
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, vt)
__global__ void _scatter_gather_elementwise_kernel(int N, func_t f) {
  constexpr int nv = nt * vt;
  int idx = nv * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < vt; ++i) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <int nt, int vt, typename func_t>
static void _launch_scatter_gather_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  const auto stream = at::cuda::getCurrentCUDAStream();
  _scatter_gather_elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <bool is_scatter_like, typename scalar_t>
struct _cuda_scatter_gather_internal_kernel {
  template <typename func_t>
  void operator() (
    TensorIterator& iter,
    int64_t index_size,
    int64_t index_stride,
    const func_t& f
  ) {
    if (iter.numel() == 0) {
      return;
    }

    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        _cuda_scatter_gather_internal_kernel<is_scatter_like, scalar_t>()(
          sub_iter, index_size, index_stride, f
        );
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    auto offset_calc = make_offset_calculator<3>(iter);
    auto loop = [=]C10_DEVICE(int i) {
      auto offsets = offset_calc.get(i);

      int64_t idx_dim = *(int64_t*)(index_ptr + offsets[2]);
      CUDA_KERNEL_ASSERT(idx_dim >= 0 && idx_dim < index_size
        && "index out of bounds");

      char* self_data = self_ptr + offsets[0];
      char* src_data = src_ptr + offsets[1];

      f(
        (scalar_t*)self_data + (is_scatter_like ? idx_dim * index_stride : 0),
        (scalar_t*)src_data + (is_scatter_like ? 0 : idx_dim * index_stride)
      );

    };

    _launch_scatter_gather_kernel<num_threads, thread_work_size>(iter.numel(), loop);
  }
}; // struct _cuda_scatter_fill_internal_kernel

template <bool is_scatter_like = true, bool cast_to_opaque = true>
struct cuda_scatter_gather_base_kernel {
  template <typename func_t>
  void operator()(
    Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name,
    const func_t& f
  ) {
    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }
    at::assert_no_internal_overlap(self);

    dim = maybe_wrap_dim(dim, self.dim());

    scatter_gather_dtype_check(method_name, self, index, src);
    if (is_scatter_like) {
      scatter_shape_check(self, dim, index, src);
    }
    else {
      gather_shape_check(self, dim, index, src);
    }

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like ?
        restride_dim(self, dim, index_sizes)
      : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like ?
        src.as_strided(index_sizes, src_strides)
      : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self_restrided)
      .add_input(src_restrided)
      .add_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;


    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
      iter.dtype(),
      "cuda_scatter_gather_base_kernel_func", [&] {
        using dtype = typename std::conditional<cast_to_opaque,
          OpaqueType<sizeof(scalar_t)>, scalar_t>::type;

        _cuda_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
          iter, index_size, index_stride, f
        );
      }
    );
  }

  void operator()(
    Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name,
    const ReduceMultiply& f
  ) {
    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }
    at::assert_no_internal_overlap(self);

    dim = maybe_wrap_dim(dim, self.dim());

    scatter_gather_dtype_check(method_name, self, index, src);
    if (is_scatter_like) {
      scatter_shape_check(self, dim, index, src);
    }
    else {
      gather_shape_check(self, dim, index, src);
    }

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like ?
        restride_dim(self, dim, index_sizes)
      : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like ?
        src.as_strided(index_sizes, src_strides)
      : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self_restrided)
      .add_input(src_restrided)
      .add_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;


    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(),
      "cuda_scatter_gather_base_kernel_reduce_multiply", [&] {
        using dtype = typename std::conditional<cast_to_opaque,
          OpaqueType<sizeof(scalar_t)>, scalar_t>::type;

        _cuda_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
          iter, index_size, index_stride, f
        );
      }
    );
  }
}; // struct cuda_scatter_gather_base_kernel

template <typename scalar_t>
struct _cuda_scatter_fill_internal_kernel {
  template <typename func_t>
  void operator()(
    TensorIterator& iter,
    scalar_t src_val,
    int64_t index_size,
    int64_t index_stride,
    const func_t& f
  ) {
    if (iter.numel() == 0) {
      return;
    }

    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        _cuda_scatter_fill_internal_kernel<scalar_t>()(
          sub_iter, src_val, index_size, index_stride, f
        );
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* index_ptr = (char*)iter.data_ptr(1);

    auto offset_calc = make_offset_calculator<2>(iter);
    auto loop = [=]C10_DEVICE(int i) {
      auto offsets = offset_calc.get(i);

      int64_t idx_dim = *(int64_t*)(index_ptr + offsets[1]);
      CUDA_KERNEL_ASSERT(idx_dim >= 0 && idx_dim < index_size
        && "index out of bounds"
      );

      char* self_data = self_ptr + offsets[0];

      f(
        (scalar_t*)self_data + idx_dim * index_stride,
        (scalar_t*)&src_val
      );

    };

    _launch_scatter_gather_kernel<num_threads, thread_work_size>(iter.numel(), loop);
  }
}; // struct _cuda_scatter_fill_internal_kernel

template <bool cast_to_opaque = true>
struct cuda_scatter_fill_base_kernel {
  template <typename func_t>
  void operator()(
    Tensor& self, int64_t dim,
    const Tensor& index, Scalar src,
    const std::string& method_name,
    const func_t& f
  ) {
    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }
    at::assert_no_internal_overlap(self);

    dim = maybe_wrap_dim(dim, self.dim());

    scatter_gather_dtype_check(method_name, self, index);
    scatter_shape_check(self, dim, index);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self_restrided)
      .add_input(index)
      .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
      iter.dtype(),
      "cuda_scatter_fill_base_kernel_func", [&] {
        using dtype = typename std::conditional<cast_to_opaque,
          OpaqueType<sizeof(scalar_t)>, scalar_t>::type;

        auto src_scalar_val = src.to<scalar_t>();
        auto src_val = *(dtype*)&src_scalar_val;

        _cuda_scatter_fill_internal_kernel<dtype>()(
          iter, src_val, index_size, index_stride, f
        );
      }
    );
  }

  void operator()(
    Tensor& self, int64_t dim,
    const Tensor& index, Scalar src,
    const std::string& method_name,
    const ReduceMultiply& f
  ) {
    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }
    at::assert_no_internal_overlap(self);

    dim = maybe_wrap_dim(dim, self.dim());

    scatter_gather_dtype_check(method_name, self, index);
    scatter_shape_check(self, dim, index);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self_restrided)
      .add_input(index)
      .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(),
      "cuda_scatter_fill_base_kernel_reduce_multiply", [&] {
        using dtype = typename std::conditional<cast_to_opaque,
          OpaqueType<sizeof(scalar_t)>, scalar_t>::type;

        auto src_scalar_val = src.to<scalar_t>();
        auto src_val = *(dtype*)&src_scalar_val;

        _cuda_scatter_fill_internal_kernel<dtype>()(
          iter, src_val, index_size, index_stride, f
        );
      }
    );
  }
}; // struct cuda_scatter_fill_base_kernel

void gather_cuda_kernel(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  cuda_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
    result, dim, index, self,
    "gather_out_cuda", tensor_assign);
}

void scatter_cuda_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cuda_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_cuda_", tensor_assign);
}

void scatter_fill_cuda_kernel(Tensor& self, int64_t dim, const Tensor& index, const Scalar& src) {
  cuda_scatter_fill_base_kernel<>()(
    self, dim, index, src,
    "scatter_fill_cuda_", tensor_assign);
}

void scatter_add_cuda_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("scatter_add_cuda_kernel");
  cuda_scatter_gather_base_kernel</*is_scatter_like=*/true, /*cast_to_opaque=*/false>()(
    self, dim, index, src,
    "scatter_add_cuda_", reduce_add);
}

void scatter_reduce_cuda_kernel(Tensor& self, const int64_t dim, const Tensor& index,
                               const Tensor& src, const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
  case SCATTER_GATHER_OP::REDUCE_ADD :
    cuda_scatter_gather_base_kernel<true, false>()(self, dim, index, src,
                                       "scatter_reduce_cuda_add_", reduce_add);
    break;
  case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
    cuda_scatter_gather_base_kernel<true, false>()(self, dim, index, src,
                                       "scatter_reduce_cuda_multiply_", reduce_multiply);
    break;
  }
}

void scatter_scalar_reduce_cuda_kernel(Tensor& self, const int64_t dim, const Tensor& index,
                               const Scalar& value, const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
  case SCATTER_GATHER_OP::REDUCE_ADD :
    cuda_scatter_fill_base_kernel<false>()(self, dim, index, value,
                                      "scatter_fill_cuda_add_", reduce_add);
    break;
  case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
    cuda_scatter_fill_base_kernel<false>()(self, dim, index, value,
                                      "scatter_fill_cuda_multiply_", reduce_multiply);
    break;
  }
}


REGISTER_DISPATCH(gather_stub, &gather_cuda_kernel);
REGISTER_DISPATCH(scatter_stub, &scatter_cuda_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cuda_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cuda_kernel);
REGISTER_DISPATCH(scatter_reduce_stub, &scatter_reduce_cuda_kernel);
REGISTER_DISPATCH(scatter_scalar_reduce_stub, &scatter_scalar_reduce_cuda_kernel);

}} // namespace at::native
