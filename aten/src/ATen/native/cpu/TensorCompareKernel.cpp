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
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native { namespace {

template <typename scalar_t, typename func_t>
static inline void compare_base_kernel(Tensor& result, Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const func_t& f) {
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  self_sizes[dim] = 1;

  // result and indice may be a empty tensor, if not,
  // reshape them as self dims
  if (!keepdim) {
    if (result.ndimension() >= dim) {
      result.unsqueeze_(dim);
    }
    if (indices.ndimension() >= dim) {
      indices.unsqueeze_(dim);
    }
  }
  result.resize_(self_sizes);
  indices.resize_(self_sizes);

  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dim=*/dim)
    .add_output(result)
    .add_output(indices)
    .add_input(self)
    .build();

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
    indices.squeeze_(dim);
  }
}

template <typename scalar_t, typename func_t>
static inline void compare_base_kernel_2_outputs(
    Tensor& result1,
    Tensor& indices1,
    Tensor& result2,
    Tensor& indices2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const func_t& f) {
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  self_sizes[dim] = 1;

  // result and indice may be a empty tensor, if not,
  // reshape them as self dims
  if (!keepdim) {
    if (result1.ndimension() >= dim) {
      result1.unsqueeze_(dim);
    }
    if (indices1.ndimension() >= dim) {
      indices1.unsqueeze_(dim);
    }
    if (result2.ndimension() >= dim) {
      result2.unsqueeze_(dim);
    }
    if (indices2.ndimension() >= dim) {
      indices2.unsqueeze_(dim);
    }
  }
  result1.resize_(self_sizes);
  indices1.resize_(self_sizes);
  result2.resize_(self_sizes);
  indices2.resize_(self_sizes);

  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dim=*/dim)
    .add_output(result1)
    .add_output(indices1)
    .add_output(result2)
    .add_output(indices2)
    .add_input(self)
    .build();

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result1_data_bytes = data[0];
    auto* indice1_data_bytes = data[1];
    auto* result2_data_bytes = data[2];
    auto* indice2_data_bytes = data[3];
    const auto* self_data_bytes = data[4];
    for (int64_t i = 0; i < n; ++i) {
      f(
        (scalar_t*)result1_data_bytes, (int64_t*)indice1_data_bytes,
        (scalar_t*)result2_data_bytes, (int64_t*)indice2_data_bytes,
        (scalar_t*)self_data_bytes, self_dim_stride
      );
      result1_data_bytes += strides[0];
      indice1_data_bytes += strides[1];
      result2_data_bytes += strides[2];
      indice2_data_bytes += strides[3];
      self_data_bytes += strides[4];
    }
  };
  iter.for_each(loop, /* grain_size */ 1);

  if (!keepdim) {
    result1.squeeze_(dim);
    indices1.squeeze_(dim);
    result2.squeeze_(dim);
    indices2.squeeze_(dim);
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

  TORCH_CHECK(result.scalar_type() == self.scalar_type() && indice.scalar_type() == kLong,
    "Expect dtype ", self.scalar_type(), "and torch.long, but got ", result.scalar_type(), "and", indice.scalar_type());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool, self.scalar_type(), "min_cpu", [&] {
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

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool, self.scalar_type(), "max_cpu", [&] {
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

static void _min_max_kernel_impl(
    Tensor& min_result,
    Tensor& min_indice,
    Tensor& max_result,
    Tensor& max_indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  TORCH_CHECK(min_result.scalar_type() == self.scalar_type() && min_indice.scalar_type() == kLong,
    "Expect min dtype ", self.scalar_type(), "and torch.long, but got ", min_result.scalar_type(), "and", min_indice.scalar_type());
  TORCH_CHECK(max_result.scalar_type() == self.scalar_type() && max_indice.scalar_type() == kLong,
    "Expect max dtype ", self.scalar_type(), "and torch.long, but got ", max_result.scalar_type(), "and", max_indice.scalar_type());

  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "_min_max_cpu", [&] {
    compare_base_kernel_2_outputs<scalar_t>(min_result, min_indice, max_result, max_indice, self, wrap_dim, keepdim, [&] (
      scalar_t* min_result_data, int64_t* min_indice_data,
      scalar_t* max_result_data, int64_t* max_indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
        scalar_t min_number = self_data[0];
        scalar_t max_number = self_data[0];
        int64_t min_index = 0;
        int64_t max_index = 0;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          scalar_t value = self_data[i * self_dim_stride];
          // note: comparison is written this way to handle NaN correctly
          if (!(value >= min_number)) {
            min_number = value;
            min_index = i;
            if (_isnan<scalar_t>(value)) {
              max_number = value;
              max_index = i;
              break;
            }
          } else if (!(value <= max_number)) {
            max_number = value;
            max_index = i;
            if (_isnan<scalar_t>(value)) {
              min_number = value;
              min_index = i;
              break;
            }
          }
        }
        *min_result_data = min_number;
        *min_indice_data = min_index;
        *max_result_data = max_number;
        *max_indice_data = max_index;
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

} // anonymous namespace

REGISTER_DISPATCH(max_stub, &max_kernel_impl);
REGISTER_DISPATCH(min_stub, &min_kernel_impl);
REGISTER_DISPATCH(_min_max_stub, &_min_max_kernel_impl);
REGISTER_DISPATCH(where_kernel, &where_kernel_impl);
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl);
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl);

}} // namespace at::native
