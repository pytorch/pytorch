#include <ATen/native/TensorAdvancedIndexing.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/native/ScatterGatherShapeChecks.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/CUDALoops.cuh>

namespace at { namespace native {

// The kernels are implemented on an opaque,
// self-aligned type of the correct size,
// to avoid redundant kernels for different types
// of the same size.
template <int N> struct alignas(N) OpaqueType { char data[N]; };

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

    auto offset_calc = legacy::make_offset_calculator<3>(iter);
    legacy::launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int i) {
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

    });
  }
};

template <bool is_scatter_like = true>
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

    dim = maybe_wrap_dim(dim, self.dim());

    if (is_scatter_like) {
      scatter_shape_check(self, dim, index, src);
    }
    else {
      gather_shape_check(self, dim, index);
    }

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like ?
        restride_dim(self, dim, index_sizes)
      : self.as_strided(index.sizes(), self.strides());
    auto src_restrided = is_scatter_like ? 
        src.as_strided(index.sizes(), src.strides())
      : restride_dim(src, dim, index_sizes);

    auto iter = TensorIterator();
    iter.dont_compute_common_dtype();
    iter.dont_resize_outputs();
    iter.add_output(self_restrided);
    iter.add_input(src_restrided, src.device(), src.scalar_type());
    iter.add_input(index);
    iter.build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
      iter.dtype(),
      method_name, [&] {
        //using dtype = OpaqueType<sizeof(scalar_t)>;
        using dtype = scalar_t;
        _cuda_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
          iter, index_size, index_stride, f
        );
      }
    );
  }
}; // struct cuda_scatter_gather_base_kernel

void scatter_cuda_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cuda_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_cuda_", []C10_DEVICE(auto* lhs, const auto* rhs) {
      *lhs = *rhs;
    }
  );
}

void scatter_fill_cuda_kernel(Tensor& self, int64_t dim, const Tensor& index, Scalar src) {
  cuda_scatter_gather_base_kernel<>()(
    self, dim, index, self,
    "scatter_fill_cuda_", [src]C10_DEVICE(auto* lhs, const auto* rhs) {
      using scalar_t = typename std::remove_pointer<decltype(lhs)>::type;
      //*lhs = src.to<scalar_t>();
    }
  );
}

REGISTER_DISPATCH(scatter_stub, &scatter_cuda_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cuda_kernel);

}} // namespace at::native
