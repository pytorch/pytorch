#include <ATen/native/TensorCompare.h>

#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <c10/util/Optional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native { namespace {

template <typename scalar_t, typename func_t>
static inline void compare_base_kernel(Tensor& result, Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const func_t& f) {
  const int64_t input_ndim = self.dim();
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  self_sizes[dim] = 1;

  // result and indice may be a empty tensor, if not,
  // reshape them as self dims
  if (0 == result.numel()) {
    result.resize_(self_sizes);
  } else {
    //error out if result cannot be viewed as desired size
    auto result_view = result.view(self_sizes);
    result.set_(result_view);
  }
  if (0 == indice.numel()) {
    indice.resize_(self_sizes);
  } else {
    auto indices_view = indice.view(self_sizes);
    indice.set_(indices_view);
  }

  Tensor self_restrided = restride_dim(self, dim, self_sizes);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.dont_resize_outputs();
  iter.add_output(result);
  iter.add_output(indice);
  iter.add_input(self_restrided);
  iter.build();

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    auto* indice_data_bytes = data[1];
    const auto* self_data_bytes = data[2];
    for (int64_t i = 0; i < n; ++i) {
      f(
        (scalar_t*)result_data_bytes, (int64_t*)indice_data_bytes,
        (scalar_t*)self_data_bytes, self_dim_stride
      );
      result_data_bytes += strides[0];
      indice_data_bytes += strides[1];
      self_data_bytes += strides[2];
    }
  };
  iter.for_each(loop, /* grain_size */ 1);

  if (!keepdim) {
    result.squeeze_(dim);
    indice.squeeze_(dim);
  }
}

static void min_kernel_impl(
    Tensor& result,
    Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool, self.scalar_type(), "min_cpu", [&] {
    compare_base_kernel<scalar_t>(result, indice, self, wrap_dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename ztype<scalar_t>::value_t;
        value_t (*zabs_)(scalar_t) = zabs<scalar_t, value_t>;
        scalar_t min_number = self_data[0];
        int64_t index = 0;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          scalar_t value = self_data[i * self_dim_stride];
          if (!(zabs_(value) >= zabs_(min_number))) {
            min_number = value;
            index = i;
            if (_isnan<scalar_t>(value)) {
              break;
            }
          }
        }
        *result_data = min_number;
        *indice_data = index;
      }
    );
  });
}

static void max_kernel_impl(
    Tensor& result,
    Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool, self.scalar_type(), "max_cpu", [&] {
    compare_base_kernel<scalar_t>(result, indice, self, wrap_dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename ztype<scalar_t>::value_t;
        value_t (*zabs_)(scalar_t) = zabs<scalar_t, value_t>;
        scalar_t max_number = self_data[0];
        int64_t index = 0;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          scalar_t value = self_data[i * self_dim_stride];
          if (!(zabs_(value) <= zabs_(max_number))) {
            max_number = value;
            index = i;
            if (_isnan<scalar_t>(value)) {
              break;
            }
          }
        }
        *result_data = max_number;
        *indice_data = index;
      }
    );
  });
}

static void where_kernel_impl(TensorIterator &iter, ScalarType condition_type) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "where_cpu", [&] {
    if (condition_type == at::ScalarType::Byte) {
      cpu_kernel(
        iter,
        [=](uint8_t cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
    } else {
      cpu_kernel(
        iter,
        [=](bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(max_stub, &max_kernel_impl);
REGISTER_DISPATCH(min_stub, &min_kernel_impl);
REGISTER_DISPATCH(where_kernel, &where_kernel_impl);

}} // namespace at::native
