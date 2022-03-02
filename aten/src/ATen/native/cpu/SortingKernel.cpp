#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Sorting.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/TopKImpl.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

template <typename func_t>
void _dim_apply(
    const TensorBase &values,
    const TensorBase &indices,
    int64_t dim,
    const std::string& method_name,
    const func_t& f) {
  dim = maybe_wrap_dim(dim, values.dim());
  TORCH_CHECK(
    dim >= 0 && dim < values.dim(),
    method_name, "(): invalid dimension parameter ", dim
  );

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(values.sizes(), /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .build();

  auto values_dim_stride = values.stride(dim);
  auto indices_dim_stride = indices.stride(dim);
  auto dim_size = values.size(dim);

  AT_DISPATCH_ALL_TYPES_AND3(
    ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
    "sorting_kernel_method_name", [&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* values_data_bytes = data[0];
        auto* indices_data_bytes = data[1];

        for (const auto i : c10::irange(n)) {
          (void)i; //Suppress unused variable warning
          f(
            reinterpret_cast<scalar_t*>(values_data_bytes),
            values_dim_stride,
            reinterpret_cast<int64_t*>(indices_data_bytes),
            indices_dim_stride,
            dim_size
          );

          values_data_bytes += strides[0];
          indices_data_bytes += strides[1];
        }
      };

      iter.for_each(loop);
    }
  );
}

template <typename scalar_t>
struct KeyValueCompAsc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (!_isnan<scalar_t>(get<0>(lhs)) && _isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) < get<0>(rhs));
  }
};

template <typename scalar_t>
struct KeyValueCompDesc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (_isnan<scalar_t>(get<0>(lhs)) && !_isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) > get<0>(rhs));
  }
};

static void sort_kernel(
    const TensorBase &values,
    const TensorBase &indices,
    int64_t dim,
    bool descending,
    bool stable) {
  dim = maybe_wrap_dim(dim, values.dim());
  _fill_indices(indices, dim);
  _dim_apply(
    values, indices, dim,
    "sort_cpu", [&](
      auto* values, int64_t values_dim_stride,
      auto* indices, int64_t indices_dim_stride,
      int64_t dim_size
    ) {
      using scalar_t = typename std::remove_pointer<decltype(values)>::type;
      auto values_accessor = StridedRandomAccessor<scalar_t>(
        values, values_dim_stride);
      auto indices_accessor = StridedRandomAccessor<int64_t>(
        indices, indices_dim_stride);
      auto composite_accessor = CompositeRandomAccessorCPU<
        decltype(values_accessor), decltype(indices_accessor)
      >(values_accessor, indices_accessor);

      if (descending) {
        if (stable) {
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
        else {
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
      }
      else {
        if (stable) {
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
        else {
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
      }
    }
  );
}

static void topk_kernel(
    const TensorBase &values,
    const TensorBase &indices,
    const TensorBase &self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto sizes = self.sizes();
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .add_input(self)
    .build();

  auto mode_values_stride = values.strides()[dim];
  auto mode_indices_stride = indices.strides()[dim];
  auto tmp_values_stride = self.strides()[dim];

  AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, self.scalar_type(), "topk_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      if (self.scalar_type() == ScalarType::BFloat16) {
        return topk_impl_loop<scalar_t, float>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n);
      } else {
        return topk_impl_loop<scalar_t, scalar_t>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n);
      }
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    iter.for_each(loop, /*grain_size=*/grain_size);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(sort_stub, &sort_kernel);
REGISTER_DISPATCH(topk_stub, &topk_kernel);

}} //at::native
