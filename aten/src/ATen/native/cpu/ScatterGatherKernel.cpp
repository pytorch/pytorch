#include <ATen/native/ScatterGatherShapeChecks.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/Parallel.h>
#include <unordered_map>


namespace at { namespace native {

namespace {

// Used for `gather`-like methods
// Test:
// 1. index.size(d) == self.size(d) for all d != dim
void gather_shape_check(const Tensor& self, int64_t dim, const Tensor& index) {
  auto self_dims = ensure_nonempty_dim(self.dim());

  TORCH_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as input tensor"
  );

  for (int64_t i = 0; i < self_dims; ++i) {
    if (i != dim) {
      TORCH_CHECK(
        ensure_nonempty_size(index, i) == ensure_nonempty_size(self, i),
        "Size does not match at dimension ", i,
        " get ", ensure_nonempty_size(self, i),
        " vs ", ensure_nonempty_size(index, i)
      );
    }
  }
}

// Used for `scatter`-like methods
// Tests:
//  1. index.size(d) <= self.size(d) for all d != dim
//  2. index.size(d) <= src.size(d) for all d if src is a Tensor
void scatter_shape_check(
  const Tensor& self, int64_t dim, const Tensor& index,
  const c10::optional<Tensor>& src_opt
) {
  bool is_wrong_shape = false;
  int64_t self_dims = ensure_nonempty_dim(self.dim());

  //  Check: index.size(d) <= self.size(d) for all d != dim
  for (int64_t d = 0; d < self_dims; ++d) {
    int64_t index_d_size = ensure_nonempty_size(index, d);
    if (d == dim) continue;
    if (index_d_size > ensure_nonempty_size(self, d)) {
      is_wrong_shape = true;
      break;
    }
  }

  //  Check: index.size(d) <= src.size(d) for all d if src is Tensor
  if (!is_wrong_shape && src_opt.has_value()) {
    auto src = src_opt.value();
    for (int64_t d = 0; d < self_dims; ++d) {
      int64_t index_d_size = ensure_nonempty_size(index, d);
      if (index_d_size > ensure_nonempty_size(src, d)) {
        is_wrong_shape = true;
        break;
      }
    }
  }

  if (src_opt.has_value()) {
    auto src = src_opt.value();
    TORCH_CHECK(!is_wrong_shape,
      "Expected index ", index.sizes(),
      " to be smaller than self ", self.sizes(),
      " apart from dimension ", dim,
      " and to be smaller size than src ", src.sizes()
    );
  }
  else {
    TORCH_CHECK(!is_wrong_shape,
      "Expected index ", index.sizes(),
      " to be smaller than self ", self.sizes(),
      " apart from dimension ", dim
    );
  }
}

template <bool is_scatter_like = true>
struct _cpu_scatter_gather_dim_loop {
  template <typename scalar_t, typename func_t>
  void operator()(
    scalar_t* self_data, int64_t self_dim_stride,
    int64_t* index_data, int64_t index_dim_stride,
    scalar_t* src_data, int64_t src_dim_stride,
    int64_t dim, int64_t index_dim_size,
    int64_t index_upper_bound,
    const func_t& f
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
    const func_t& f
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
        value
      );
    }
  }
};

// implement reduce_multiply as a class since the multiplication requires a type
// specialization for the boolean operation (which refuses to proceed the compilation
// on clang).
class ReduceMultiply {
public:
  ReduceMultiply() {};
  // don't use auto due to complaints from clang.
  template <typename scalar_t>
  void operator()(scalar_t * self_data, scalar_t * src_data) {
    *self_data *= *src_data;
  };

  void operator()(bool * self_data, bool * src_data) {
    *self_data = *self_data && *src_data;
  };
};
ReduceMultiply reduce_multiply;

auto reduce_add = [](auto * self_data, auto * src_data) {
                    *self_data += *src_data;
                  };
auto reduce_subtract = [](auto * self_data, auto * src_data) {
                         *self_data -= *src_data;
                       };
auto reduce_divide = [](auto * self_data, auto * src_data) {
                       *self_data /= *src_data;
                     };
auto tensor_assign = [](auto * self_data, auto * src_data) {
                *self_data = *src_data;
              };
auto scalar_assign = [](auto * self_data, Scalar src_data) {
                       using scalar_t = typename std::remove_pointer<decltype(self_data)>::type;
                       *self_data = src_data.to<scalar_t>();
                     };
auto scalar_reduce_add = [](auto * self_data, Scalar src_data) {
                       using scalar_t = typename std::remove_pointer<decltype(self_data)>::type;
                       *self_data += src_data.to<scalar_t>();
                     };
auto scalar_reduce_subtract = [](auto * self_data, Scalar src_data) {
                       using scalar_t = typename std::remove_pointer<decltype(self_data)>::type;
                       *self_data -= src_data.to<scalar_t>();
                     };
auto scalar_reduce_multiply = [](auto * self_data, Scalar src_data) {
                       using scalar_t = typename std::remove_pointer<decltype(self_data)>::type;
                       *self_data *= src_data.to<scalar_t>();
                     };
auto scalar_reduce_divide = [](auto * self_data, Scalar src_data) {
                       using scalar_t = typename std::remove_pointer<decltype(self_data)>::type;
                       *self_data /= src_data.to<scalar_t>();
                     };

  class Awesome {
  public:
    Awesome(int op) {
      
    }

    void operator()(auto * self_data, auto * src_data) {
      
    }
  };

template <bool is_scatter_like = true>
struct cpu_scatter_gather_base_kernel {
  void operator()(Tensor& self, int64_t dim,
    const Tensor& index, Scalar& value,
    const std::string& method_name,
    bool serial_exec, const SCATTER_GATHER_OP& func_enum) {
    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }

    dim = maybe_wrap_dim(dim, self.dim());

    if (is_scatter_like) {
      scatter_shape_check(self, dim, index, self);
    }
    else {
      gather_shape_check(self, dim, index);
    }

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto index_strides = ensure_nonempty_vec(index.strides().vec());

    // `dim` is traversed in the kernel,
    // that is why index.stride(dim) = 0 and index.size(dim) = 1.
    // Also, index.size(dim) = 1 makes sure that TensorIterator.DimCounter
    // has the following form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
    index_sizes[dim] = 1;
    index_strides[dim] = 0;

    // set self.shape = src.shape = index.shape,
    // this defines the number of elements to iterate over,
    // and set self.stride(dim) = src.stride(dim) = 0,
    // because `dim` is traversed in the kernel.
    auto self_restrided = restride_dim(self, dim, index_sizes);
    auto index_restrided = index.as_strided(index_sizes, index_strides);

    auto iter = TensorIterator();
    iter.dont_compute_common_dtype();
    iter.dont_resize_outputs();
    iter.add_output(self_restrided);
    iter.add_input(index_restrided);
    iter.build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);
    
    auto index_upper_bound = self_dim_size;

    AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, iter.dtype(),
      method_name, [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 1;

        using binary_func_t = std::function<void(scalar_t*, Scalar)>;
        std::unordered_map<const SCATTER_GATHER_OP, binary_func_t> binary_funcs;
        binary_funcs[SCATTER_GATHER_OP::SCALAR_ASSIGN] =  scalar_assign;
        binary_funcs[SCATTER_GATHER_OP::SCALAR_REDUCE_ADD] = scalar_reduce_add;
        binary_funcs[SCATTER_GATHER_OP::SCALAR_REDUCE_SUBTRACT] = scalar_reduce_subtract;
        binary_funcs[SCATTER_GATHER_OP::SCALAR_REDUCE_MULTIPLY] = scalar_reduce_multiply;
        binary_funcs[SCATTER_GATHER_OP::SCALAR_REDUCE_DIVIDE] = scalar_reduce_divide;

        auto run_loop = [&](const auto& kernel_func) {
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

                  kernel_func(
                              (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                              value);

                  self_data += strides[SELF_ITER_STRIDE_IDX];
                  index_data += strides[INDEX_ITER_STRIDE_IDX];
                }
              }
            }
          };
          
          if (serial_exec) {
            iter.serial_for_each(loop, {0, iter.numel()});
          }
          else {
            iter.for_each(loop);
          }
        };

        run_loop(binary_funcs[func_enum]);
      }
    );
  }
  
  void operator()(Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name,
    bool serial_exec,
    const SCATTER_GATHER_OP& func_enum) {
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
    auto index_strides = ensure_nonempty_vec(index.strides().vec());

    // `dim` is traversed in the kernel,
    // that is why index.stride(dim) = 0 and index.size(dim) = 1.
    // Also, index.size(dim) = 1 makes sure that TensorIterator.DimCounter
    // has the following form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
    index_sizes[dim] = 1;
    index_strides[dim] = 0;

    // set self.shape = src.shape = index.shape,
    // this defines the number of elements to iterate over,
    // and set self.stride(dim) = src.stride(dim) = 0,
    // because `dim` is traversed in the kernel.
    auto self_restrided = restride_dim(self, dim, index_sizes);
    auto index_restrided = index.as_strided(index_sizes, index_strides);
    auto src_restrided = restride_dim(src, dim, index_sizes);

    auto iter = TensorIterator();
    iter.dont_compute_common_dtype();
    iter.dont_resize_outputs();
    iter.add_output(self_restrided);
    iter.add_input(src_restrided, src.device(), src.scalar_type());
    iter.add_input(index_restrided);
    iter.build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);
    
    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;
    
    AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, iter.dtype(),
      method_name, [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;

        using binary_func_t = std::function<void(scalar_t*, scalar_t*)>;
        std::unordered_map<const SCATTER_GATHER_OP, binary_func_t> binary_funcs;
        binary_funcs[SCATTER_GATHER_OP::REDUCE_ADD] = reduce_add;
        binary_funcs[SCATTER_GATHER_OP::REDUCE_SUBTRACT] = reduce_subtract;
        binary_funcs[SCATTER_GATHER_OP::REDUCE_MULTIPLY] = reduce_multiply;
        binary_funcs[SCATTER_GATHER_OP::REDUCE_DIVIDE] = reduce_divide;
        binary_funcs[SCATTER_GATHER_OP::TENSOR_ASSIGN] = tensor_assign;

        auto run_loop = [&](const auto& kernel_func) {
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
          if (serial_exec) {
            iter.serial_for_each(loop, {0, iter.numel()});
          }
          else {
            iter.for_each(loop);
          }
        };

        run_loop(binary_funcs[func_enum]);
      }
    );
  }
};

void gather_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  cpu_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
    result, dim, index, self,
    "gather_out_cpu", /*serial_exec=*/false, SCATTER_GATHER_OP::TENSOR_ASSIGN);
}

void scatter_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src, "scatter_cpu_", false, SCATTER_GATHER_OP::TENSOR_ASSIGN);
}

void scatter_fill_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, value, "scatter_fill_cpu_", /*serial_exec=*/false,
    SCATTER_GATHER_OP::SCALAR_ASSIGN);
}

void scatter_add_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_add_", /*serial_exec=*/true, SCATTER_GATHER_OP::REDUCE_ADD);
}

void scatter_reduce_cpu_kernel(Tensor& self, const int64_t dim, const Tensor& index,
                               const Tensor& src, const SCATTER_GATHER_OP& reduce) {
  cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                    "scatter_reduce_", true, reduce);
}

void scatter_scalar_reduce_cpu_kernel(Tensor& self, const int64_t dim, const Tensor& index,
                                      Scalar& value, const SCATTER_GATHER_OP& reduce) {
  cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                     "scatter_scalar_reduce_", true, reduce);
}

} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_cpu_kernel);
REGISTER_DISPATCH(scatter_stub, &scatter_cpu_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cpu_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel);
REGISTER_DISPATCH(scatter_reduce_stub, &scatter_reduce_cpu_kernel);
REGISTER_DISPATCH(scatter_scalar_reduce_stub, &scatter_scalar_reduce_cpu_kernel);

}} // namespace at::native
