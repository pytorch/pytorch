#include <ATen/native/ScatterGatherShapeChecks.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>

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

// Used for `scatter` and `scatter_add`
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

template <typename func_t>
void cpu_scatter_gather_base_kernel(
  Tensor& self, int64_t dim,
  const Tensor& index, const Tensor& src,
  const std::string& method_name,
  const func_t& f,
  bool serial_exec = true
) {
  auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
  auto index_strides = ensure_nonempty_vec(index.strides().vec());

  // `dim` is traversed in a kernel function `f`,
  // that is why index.stride(dim) = 0 and index.size(dim) = 1.
  // Also, index.size(dim) = 1 makes sure that TensorIterator.DimCounter
  // has the following form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
  index_sizes[dim] = 1;
  index_strides[dim] = 0;

  // set self.shape = src.shape = index.shape,
  // this defines the number of elements to iterate over,
  // and set self.stride(dim) = src.stride(dim) = 0,
  // because `dim` is traversed in a kernel function `f`.
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
  auto index_dim_stride = ensure_nonempty_stride(index, dim);
  auto src_dim_stride = ensure_nonempty_stride(src, dim);

  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Bool, ScalarType::Half, iter.dtype(),
    method_name, [&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* self_data_bytes = data[0];
        const auto* index_data_bytes = data[2];
        const auto* src_data_bytes = data[1];

        for (int64_t i = 0; i < n; ++i) {
          f(
            (scalar_t*)self_data_bytes, self_dim_stride,
            (int64_t*)index_data_bytes, index_dim_stride,
            (scalar_t*)src_data_bytes, src_dim_stride
          );

          self_data_bytes += strides[0];
          index_data_bytes += strides[2];
          src_data_bytes += strides[1];
        }
      };
      if (serial_exec) {
        iter.serial_for_each(loop, {0, iter.numel()});
      } else {
        iter.for_each(loop);
      }
    }
  );
}

void gather_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  if (index.numel() == 0) {
    return;
  }

  dim = maybe_wrap_dim(dim, self.dim());

  gather_shape_check(self, dim, index);

  int64_t index_dim_size = ensure_nonempty_size(index, dim);
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  cpu_scatter_gather_base_kernel(
    result, dim, index, self,
    "gather_out_cpu", [&] (
      auto* result_data, auto result_dim_stride,
      const auto* index_data, auto index_dim_stride,
      const auto* self_data, auto self_dim_stride
    ) {
      for (int64_t i = 0; i < index_dim_size; ++i) {
        int64_t idx_dim = index_data[i * index_dim_stride];
        // we are not putting idx_dim in the error message because it disables
        // loop optimization in clang-7
        TORCH_CHECK(idx_dim >= 0 && idx_dim < self_dim_size,
                    "index ", index_data[i * index_dim_stride], " is out of bounds for dimension ", dim,
                    " with size ", self_dim_size);
        result_data[i * result_dim_stride] = self_data[idx_dim * self_dim_stride];
      }
    }, /*serial_exec=*/false
  );
}

void scatter_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  if (index.numel() == 0) {
    return;
  }

  dim = maybe_wrap_dim(dim, self.dim());
  
  scatter_shape_check(self, dim, index, src);

  int64_t index_dim_size = ensure_nonempty_size(index, dim);
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  cpu_scatter_gather_base_kernel(
    self, dim, index, src,
    "scatter_cpu_", [&] (
      auto* self_data, auto self_dim_stride,
      const auto* index_data, auto index_dim_stride,
      const auto* src_data, auto src_dim_stride
    ) {
      for (int64_t i = 0; i < index_dim_size; ++i) {
        int64_t idx_dim = index_data[i * index_dim_stride];
        // we are not putting idx_dim in the error message because it disables
        // loop optimization in clang-7
        TORCH_CHECK(idx_dim >= 0 && idx_dim < self_dim_size,
          "index ", index_data[i * index_dim_stride],
          " is out of bounds for dimension ", dim,
          " with size ", self_dim_size);
        self_data[idx_dim * self_dim_stride] = src_data[i * src_dim_stride];
      }
    }, /*serial_exec=*/false
  );
}

void scatter_fill_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, Scalar src) {
  if (index.numel() == 0) {
    return;
  }

  dim = maybe_wrap_dim(dim, self.dim());
  
  scatter_shape_check(self, dim, index);

  int64_t index_dim_size = ensure_nonempty_size(index, dim);
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  cpu_scatter_gather_base_kernel(
    self, dim, index, self,
    "scatter_fill_cpu_", [&] (
      auto* self_data, auto self_dim_stride,
      const auto* index_data, auto index_dim_stride,
      const auto* src_data, auto src_dim_stride
    ) {
      for (int64_t i = 0; i < index_dim_size; ++i) {
        int64_t idx_dim = index_data[i * index_dim_stride];
        // we are not putting idx_dim in the error message because it disables
        // loop optimization in clang-7
        TORCH_CHECK(idx_dim >= 0 && idx_dim < self_dim_size,
          "index ", index_data[i * index_dim_stride],
          " is out of bounds for dimension ", dim,
          " with size ", self_dim_size);
        using scalar_t = typename std::remove_pointer<decltype(self_data)>::type;
        self_data[idx_dim * self_dim_stride] = src.to<scalar_t>();
      }
    }, /*serial_exec=*/false
  );
}

void scatter_add_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  if (index.numel() == 0) {
    return;
  }

  dim = maybe_wrap_dim(dim, self.dim());
  
  scatter_shape_check(self, dim, index, src);

  int64_t index_dim_size = ensure_nonempty_size(index, dim);
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  cpu_scatter_gather_base_kernel(
    self, dim, index, src,
    "scatter_add_", [&] (
      auto* self_data, auto self_dim_stride,
      const auto* index_data, auto index_dim_stride,
      const auto* src_data, auto src_dim_stride
    ) {
      for (int64_t i = 0; i < index_dim_size; ++i) {
        int64_t idx_dim = index_data[i * index_dim_stride];
        // we are not putting idx_dim in the error message because it disables
        // loop optimizations in clang-7
        TORCH_CHECK(idx_dim >= 0 && idx_dim < self_dim_size,
                    "index ", index_data[i * index_dim_stride], " is out of bounds for dimension ", dim,
                    " with size ", self_dim_size);
        self_data[idx_dim * self_dim_stride] += src_data[i * src_dim_stride];
      }
    },
      /*serial_exec=*/true);
}

} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_cpu_kernel);
REGISTER_DISPATCH(scatter_stub, &scatter_cpu_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cpu_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel);

}} // namespace at::native
