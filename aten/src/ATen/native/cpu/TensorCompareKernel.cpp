#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorCompare.h>

#include <numeric>
#include <iterator>
#include <algorithm>
#include <utility>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/irange.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/Loops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/result_type.h>
#endif

namespace at::native { namespace {

template <typename scalar_t, typename scalar_t_2 = int64_t, typename loop1d_t>
static inline void compare_base_kernel_core(
    const Tensor& result1,
    const Tensor& result2,
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
    .add_const_input(self)
    .build();

  iter.for_each(loop, /* grain_size */ 1);

  if (!keepdim) {
    result1.squeeze_(dim);
    result2.squeeze_(dim);
  }
}

template <typename scalar_t, typename scalar_t_2=int64_t, typename func_t>
static inline void compare_base_kernel(const Tensor& result1, const Tensor& result2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const func_t& f) {

  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result1_data_bytes = data[0];
    auto* result2_data_bytes = data[1];
    const auto* self_data_bytes = data[2];
    for ([[maybe_unused]] const auto i : c10::irange(n)) {
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
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "min_cpu", [&] {
    compare_base_kernel<scalar_t>(result, indice, self, dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
        value_t (*zabs_)(scalar_t) = zabs<scalar_t, value_t>;
        scalar_t min_number = c10::load(self_data);
        int64_t index = 0;
        for (const auto i : c10::irange(self_dim_size)) {
          scalar_t value = c10::load(&self_data[i * self_dim_stride]);
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
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "max_cpu", [&] {
    compare_base_kernel<scalar_t>(result, indice, self, dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
        value_t (*zabs_)(scalar_t) = zabs<scalar_t, value_t>;
        scalar_t max_number = c10::load(self_data);
        int64_t index = 0;
        for (const auto i : c10::irange(self_dim_size)) {
          scalar_t value = c10::load(&self_data[i * self_dim_stride]);
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

static void aminmax_kernel(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& min_result,
    Tensor& max_result) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  TORCH_CHECK(min_result.scalar_type() == self.scalar_type() && max_result.scalar_type() == self.scalar_type(),
    "Expect min and max dtype ", self.scalar_type(),
    " but got ", min_result.scalar_type(), " and ", max_result.scalar_type());

  if (self.numel() == 1 && self.ndimension() == 0) {
    TORCH_CHECK(!self.is_complex(), "aminmax not implemented for ", self.scalar_type());
    min_result.resize_({});
    max_result.resize_({});
    min_result.fill_(self);
    max_result.fill_(self);
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "aminmax_cpu", [&] {
    compare_base_kernel<scalar_t, scalar_t>(min_result, max_result, self, wrap_dim, keepdim, [&] (
      scalar_t* min_result_data, scalar_t* max_result_data,
      const scalar_t* self_data, auto self_dim_stride) {
        scalar_t min_number = c10::load(self_data);
        scalar_t max_number = min_number;
        for (const auto i : c10::irange(self_dim_size)) {
          scalar_t value = c10::load(&self_data[i * self_dim_stride]);
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

static void where_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_V2(
    iter.dtype(), "where_cpu", [&] {
      cpu_kernel(
        iter,
        [=](bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
  },
  kComplexHalf, kHalf, kBFloat16, kBool, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_FLOAT8_TYPES));
}

static void isposinf_kernel_impl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); });
  });
}

static void isneginf_kernel_impl(TensorIteratorBase& iter) {
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

          for ([[maybe_unused]] const auto k : c10::irange(n)) {
            scalar_t* values_data = (scalar_t*)values_data_bytes;
            int64_t* indices_data = (int64_t*)indices_data_bytes;
            const scalar_t* self_data = (scalar_t*)self_data_bytes;

            scalar_t mode = 0;
            int64_t modei = 0;
            int64_t temp_freq = 0;
            int64_t max_freq = 0;

            for (const auto i : c10::irange(self_dim_size)) {
              elements[i] = std::make_pair(c10::load(&self_data[i * self_dim_stride]), i);
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

            for (const auto i : c10::irange(self_dim_size)) {
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

// Default brute force implementation of isin(). Used when the number of test elements is small.
// Iterates through each element and checks it against each test element.
static void isin_default_kernel_cpu(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out) {
  // Since test elements is not an input of the TensorIterator, type promotion
  // must be done manually.
  ScalarType common_type = at::result_type(elements, test_elements);
  Tensor promoted_elements = elements.to(common_type);
  Tensor test_elements_flat = test_elements.to(common_type).view(-1);
  auto test_elements_stride = test_elements_flat.stride(0);

  auto iter = TensorIteratorConfig()
    .add_output(out)
    .add_const_input(promoted_elements)
    .check_all_same_dtype(false)
    .build();
  // Dispatch based on promoted type.
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(1), "isin_default_cpu", [&]() {
    cpu_kernel(iter, [&](scalar_t element_val) -> bool {
      const auto* test_element_data = test_elements_flat.const_data_ptr<scalar_t>();
      for (const auto j : c10::irange(test_elements_flat.numel())) {
        if (element_val == *(test_element_data + test_elements_stride * j)) {
          return !invert;
        }
      }
      return invert;
    });
  });
}

static void clamp_kernel_impl(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_cpu", [&]() {
    cpu_kernel_vec(iter,
      [](scalar_t a, scalar_t min, scalar_t max) -> scalar_t {
        if (min != min || max != max) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        } else {
            return std::min(std::max(a, min), max);
        }
      },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> min, Vectorized<scalar_t> max) {
        return vec::minimum(vec::maximum(a, min), max);
      });
  });
}

static void clamp_scalar_kernel_impl(TensorIteratorBase& iter, const Scalar& min_, const Scalar& max_) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_scalar_cpu", [&]() {
    const auto min = min_.to<scalar_t>();
    const auto max = max_.to<scalar_t>();
    const Vectorized<scalar_t> min_vec(min);
    const Vectorized<scalar_t> max_vec(max);
      cpu_kernel_vec(iter,
        [=](scalar_t a) -> scalar_t {
          return std::min(std::max(a, min), max);
        },
        [=](Vectorized<scalar_t> a) {
          return vec::clamp(a, min_vec, max_vec);
        });
  });
}

static void clamp_max_scalar_kernel_impl(TensorIteratorBase& iter, Scalar max_) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_max_scalar_cpu", [&]() {
    const auto max = max_.to<scalar_t>();
    const Vectorized<scalar_t> max_vec(max);
    cpu_kernel_vec(iter,
      [=](scalar_t a) -> scalar_t {
        return std::min(a, max);
      },
      [=](Vectorized<scalar_t> a) {
        return vec::clamp_max(a, max_vec);
      });
  });
}

static void clamp_min_scalar_kernel_impl(TensorIteratorBase& iter, Scalar min_) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_min_scalar_cpu", [&]() {
    const auto min = min_.to<scalar_t>();
    const Vectorized<scalar_t> min_vec(min);
    cpu_kernel_vec(iter,
        [=](scalar_t a) -> scalar_t {
          return std::max(a, min);
        },
        [=](Vectorized<scalar_t> a) {
          return vec::clamp_min(a, min_vec);
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(max_stub, &max_kernel_impl)
REGISTER_DISPATCH(min_stub, &min_kernel_impl)
REGISTER_DISPATCH(aminmax_stub, &aminmax_kernel)
REGISTER_DISPATCH(where_kernel, &where_kernel_impl)
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl)
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl)
REGISTER_DISPATCH(mode_stub, &mode_kernel_impl)
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl)
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl)
REGISTER_DISPATCH(isin_default_stub, &isin_default_kernel_cpu)

} // namespace at::native
