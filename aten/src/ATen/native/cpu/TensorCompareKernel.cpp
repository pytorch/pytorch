#include <ATen/native/TensorCompare.h>

#include <numeric>
#include <iterator>
#include <algorithm>
#include <utility>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <c10/util/Optional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native { namespace {

template <typename scalar_t, typename scalar_t_2 = int64_t, typename loop1d_t>
static inline void compare_base_kernel_core(
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const loop1d_t& loop) {
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  self_sizes[dim] = 1;

  // result1 and result2 may be a empty tensor, if not,
  // reshape them as self dims
  if (!keepdim) {
    if (result1.ndimension() >= dim) {
      result1.unsqueeze_(dim);
    }
    if (result2.ndimension() >= dim) {
      result2.unsqueeze_(dim);
    }
  }

  at::native::resize_output(result1, self_sizes);
  at::native::resize_output(result2, self_sizes);

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dims=*/dim)
    .add_output(result1)
    .add_output(result2)
    .add_input(self)
    .build();

  iter.for_each(loop, /* grain_size */ 1);

  if (!keepdim) {
    result1.squeeze_(dim);
    result2.squeeze_(dim);
  }
}

template <typename scalar_t, typename scalar_t_2=int64_t, typename func_t>
static inline void compare_base_kernel(Tensor& result1, Tensor& result2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const func_t& f) {

  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result1_data_bytes = data[0];
    auto* result2_data_bytes = data[1];
    const auto* self_data_bytes = data[2];
    for (int64_t i = 0; i < n; ++i) {
      f((scalar_t*)result1_data_bytes,
        (scalar_t_2*)result2_data_bytes,
        (scalar_t*)self_data_bytes,
        self_dim_stride);
      result1_data_bytes += strides[0];
      result2_data_bytes += strides[1];
      self_data_bytes += strides[2];
    }
  };

  compare_base_kernel_core<scalar_t, scalar_t_2>(
      result1, result2, self, dim, keepdim, loop);
}

static void min_kernel_impl(
    Tensor& result,
    Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  TORCH_CHECK(result.scalar_type() == self.scalar_type() && indice.scalar_type() == kLong,
    "Expect dtype ", self.scalar_type(), "and torch.long, but got ", result.scalar_type(), "and", indice.scalar_type());

  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "min_cpu", [&] {
    compare_base_kernel<scalar_t>(result, indice, self, wrap_dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
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

  TORCH_CHECK(result.scalar_type() == self.scalar_type() && indice.scalar_type() == kLong,
    "Expect dtype ", self.scalar_type(), "and torch.long, but got ", result.scalar_type(), "and", indice.scalar_type());

  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "max_cpu", [&] {
    compare_base_kernel<scalar_t>(result, indice, self, wrap_dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
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

static void _aminmax_kernel_impl(
    Tensor& min_result,
    Tensor& max_result,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  TORCH_CHECK(min_result.scalar_type() == self.scalar_type() && max_result.scalar_type() == self.scalar_type(),
    "Expect min and max dtype ", self.scalar_type(),
    " but got ", min_result.scalar_type(), " and ", max_result.scalar_type());

  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "_aminmax_cpu", [&] {
    compare_base_kernel<scalar_t, scalar_t>(min_result, max_result, self, wrap_dim, keepdim, [&] (
      scalar_t* min_result_data, scalar_t* max_result_data,
      const scalar_t* self_data, auto self_dim_stride) {
        scalar_t min_number = self_data[0];
        scalar_t max_number = self_data[0];
        for (int64_t i = 0; i < self_dim_size; ++i) {
          scalar_t value = self_data[i * self_dim_stride];
          // note: comparison is written this way to handle NaN correctly
          if (!(value >= min_number)) {
            min_number = value;
            if (_isnan<scalar_t>(value)) {
              max_number = value;
              break;
            }
          } else if (!(value <= max_number)) {
            max_number = value;
          }
        }
        *min_result_data = min_number;
        *max_result_data = max_number;
      }
    );
  });
}

static void where_kernel_impl(TensorIterator &iter, ScalarType condition_type) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    iter.dtype(), "where_cpu", [&] {
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

static void isposinf_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); });
  });
}

static void isneginf_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); });
  });
}

static void mode_kernel_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto self_dim_size = ensure_nonempty_size(self, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, self.scalar_type(), "mode_cpu", [&] {
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* values_data_bytes = data[0];
          auto* indices_data_bytes = data[1];
          const auto* self_data_bytes = data[2];

          std::vector<std::pair<scalar_t, int64_t>> elements(self_dim_size);

          for (int64_t k = 0; k < n; ++k) {
            scalar_t* values_data = (scalar_t*)values_data_bytes;
            int64_t* indices_data = (int64_t*)indices_data_bytes;
            const scalar_t* self_data = (scalar_t*)self_data_bytes;

            scalar_t mode = 0;
            int64_t modei = 0;
            int64_t temp_freq = 0;
            int64_t max_freq = 0;

            for (int64_t i = 0; i < self_dim_size; i++) {
              elements[i] = std::make_pair(self_data[i * self_dim_stride], i);
            }

            // Even though, theoretically, we don't need to specify this lambda
            // (it's basically the same as std::less), doing so degrades
            // performance. That is because its implementation for std::pair
            // uses 3 comparisons.
            std::sort(
                elements.begin(),
                elements.end(),
                [=](const auto& i, const auto& j) {
                  return i.first < j.first;
                });

            for (int64_t i = 0; i < self_dim_size; i++) {
              temp_freq++;
              if ((i == self_dim_size - 1) ||
                  (elements[i].first != elements[i + 1].first)) {
                if (temp_freq > max_freq) {
                  mode = elements[i].first;
                  modei = elements[i].second;
                  max_freq = temp_freq;
                }
                temp_freq = 0;
              }
            }

            *values_data = mode;
            *indices_data = modei;

            values_data_bytes += strides[0];
            indices_data_bytes += strides[1];
            self_data_bytes += strides[2];
          }
        };

        compare_base_kernel_core<scalar_t>(
            values, indices, self, dim, keepdim, loop);
      });
}

} // anonymous namespace

REGISTER_DISPATCH(max_stub, &max_kernel_impl);
REGISTER_DISPATCH(min_stub, &min_kernel_impl);
REGISTER_DISPATCH(_aminmax_stub, &_aminmax_kernel_impl);
REGISTER_DISPATCH(where_kernel, &where_kernel_impl);
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl);
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl);
REGISTER_DISPATCH(mode_stub, &mode_kernel_impl);

}} // namespace at::native
