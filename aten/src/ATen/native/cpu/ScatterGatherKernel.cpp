#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/Parallel.h>

namespace at { namespace native {

namespace {

// Implement as functors since lambdas don't get optimized.
class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data *= *src_data;
  }

  constexpr void operator() (bool * self_data, bool * src_data) const {
    *self_data = *self_data && *src_data;
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static ReduceMultiply reduce_multiply;

class ReduceAdd {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data += *src_data;
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static ReduceAdd reduce_add;

class TensorAssign {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data = *src_data;
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static TensorAssign tensor_assign;

template <bool is_scatter_like = true>
struct _cpu_scatter_gather_dim_loop {
  template <typename scalar_t, typename func_t>
  void operator()(
    scalar_t* self_data, int64_t self_dim_stride,
    int64_t* index_data, int64_t index_dim_stride,
    scalar_t* src_data, int64_t src_dim_stride,
    int64_t dim, int64_t index_dim_size,
    int64_t index_upper_bound,
    func_t& f
  ) {

    for (int64_t i = 0; i < index_dim_size; ++i) {
      int64_t idx_dim = index_data[i * index_dim_stride];
      // we are not putting idx_dim in the error message because it disables
      // loop optimization in clang-7
      TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
        "index ", index_data[i * index_dim_stride],
        " is out of bounds for dimension ", dim,
        " with size ", index_upper_bound
      );

      f(
        self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
        src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
      );
    }
  }

  template <typename scalar_t, typename func_t>
  void operator()(
    scalar_t* self_data, int64_t self_dim_stride,
    int64_t* index_data, int64_t index_dim_stride,
    Scalar value,
    int64_t dim, int64_t index_dim_size,
    int64_t index_upper_bound,
    func_t& f
  ) {

    for (int64_t i = 0; i < index_dim_size; ++i) {
      int64_t idx_dim = index_data[i * index_dim_stride];
      // we are not putting idx_dim in the error message because it disables
      // loop optimization in clang-7
      TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
        "index ", index_data[i * index_dim_stride],
        " is out of bounds for dimension ", dim,
        " with size ", index_upper_bound
      );
      auto temp = value.to<scalar_t>();
      f(
        self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp
      );
    }
  }
};


template <bool is_scatter_like = true>
struct cpu_scatter_gather_base_kernel {
  template <typename func_t>
  void operator()(Tensor& self, int64_t dim,
    const Tensor& index, const Scalar& value,
    const std::string& method_name, func_t& kernel_func) {
    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }

    dim = maybe_wrap_dim(dim, self.dim());

    if (is_scatter_like) {
      scatter_shape_check(self, dim, index, self);
    }
    else {
      gather_shape_check(self, dim, index, self);
    }

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto index_strides = ensure_nonempty_vec(index.strides().vec());

    // `dim` is traversed in the kernel,
    // that is why index.stride(dim) = 0 and index.size(dim) = 1.
    // Also, index.size(dim) = 1 makes sure that TensorIterator.DimCounter
    // has the following form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
    index_sizes[dim] = 1;
    index_strides[dim] = 0;

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_borrowed_output(self)
      .add_borrowed_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto index_upper_bound = self_dim_size;

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Bool, ScalarType::Half, iter.dtype(),
      "scatter_gather_scalar_cpu", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 1;

        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension and/or
          // whether `n` is smaller than `index_dim_size`

          if ((dim== self.dim() - 1) || (n < index_dim_size)) {
            for (int64_t nelem = 0; nelem < n; ++nelem) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                (scalar_t*)self_data_bytes, self_dim_stride,
                (int64_t*)index_data_bytes, index_dim_stride,
                value, dim, index_dim_size, index_upper_bound,
                kernel_func);

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
            }
          }
          else {
            for (int64_t i = 0; i < index_dim_size; ++i) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              for (int64_t nelem = 0; nelem < n; ++nelem) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                auto temp = value.to<scalar_t>();
                kernel_func((scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop);
      }
    );
  }

  template <typename func_t>
  void operator()(Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, func_t& kernel_func) {

    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }

    dim = maybe_wrap_dim(dim, self.dim());

    scatter_gather_dtype_check(method_name, self, index, src);
    if (is_scatter_like) {
      scatter_shape_check(self, dim, index, src);
    }
    else {
      gather_shape_check(self, dim, index, src);
    }

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_borrowed_output(self)
      .add_borrowed_input(src)
      .add_borrowed_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Bool, ScalarType::Half, iter.dtype(),
      "scatter_gather_tensor_cpu", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension and/or
          // whether `n` is smaller than `index_dim_size`
          if ((dim== self.dim() - 1) || (n < index_dim_size)) {
            for (int64_t nelem = 0; nelem < n; ++nelem) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                 (scalar_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (int64_t i = 0; i < index_dim_size; ++i) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              for (int64_t nelem = 0; nelem < n; ++nelem) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop);
      }
    );
  }
};

void gather_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  cpu_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
    result, dim, index, self,
    "gather_out_cpu", tensor_assign);
}

void scatter_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src, "scatter_cpu_", tensor_assign);
}

void scatter_fill_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, value, "scatter_fill_cpu_", tensor_assign);
}

void scatter_add_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_add_", reduce_add);

}

void scatter_reduce_cpu_kernel(Tensor& self, const int64_t dim, const Tensor& index,
                               const Tensor& src, const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
  case SCATTER_GATHER_OP::REDUCE_ADD :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_add_", reduce_add);
    break;
  case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_multiply_", reduce_multiply);
    break;
  }
}

void scatter_scalar_reduce_cpu_kernel(Tensor& self, const int64_t dim, const Tensor& index,
                                      const Scalar& value, const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
  case SCATTER_GATHER_OP::REDUCE_ADD :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                       "scatter_scalar_reduce_add_", reduce_add);
    break;
  case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                       "scatter_scalar_reduce_multiply_", reduce_multiply);
    break;
  }
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(gather_stub, &gather_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(scatter_stub, &scatter_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(scatter_reduce_stub, &scatter_reduce_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(scatter_scalar_reduce_stub, &scatter_scalar_reduce_cpu_kernel);

}} // namespace at::native
