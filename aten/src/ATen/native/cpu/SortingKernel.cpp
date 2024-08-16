#define TORCH_ASSERT_NO_OPERATORS

#include <limits>

#include <ATen/native/Sorting.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/TopKImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#ifdef USE_FBGEMM
#include <fbgemm/Utils.h>
#endif

namespace at::native {

namespace {

template <typename func_t>
void _dim_apply(
    const TensorBase &values,
    const TensorBase &indices,
    int64_t dim,
    const std::string& method_name,
    const func_t& f) {
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

  AT_DISPATCH_V2(
    iter.dtype(), "sorting_kernel_method_name", AT_WRAP([&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* values_data_bytes = data[0];
        auto* indices_data_bytes = data[1];

        if(values_data_bytes==nullptr || indices_data_bytes==nullptr){
          return;
        }

        for (const auto i C10_UNUSED : c10::irange(n)) {
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

      int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, dim_size);
      iter.for_each(loop, /*grain_size=*/grain_size);
    }), kBool, kHalf, kBFloat16, AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
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

#ifdef USE_FBGEMM
static bool can_use_radix_sort(const TensorBase& values, const bool descending) {
  // radix_sort can be used only for 1D data
  if (values.dim() != 1) return false;
  // radix_sort sorts in ascending order
  if (descending) return false;
  // radix_sort works for integer values
  if (!at::isIntegralType(values.scalar_type(), /*includeBool=*/false)) return false;
  // performance improvements are visible for bigger tensor sizes, when radix_sort
  // is accelerated with OpenMP
  if (values.numel() < at::internal::GRAIN_SIZE || !fbgemm::is_radix_sort_accelerated_with_openmp()) return false;
  // TODO(DamianSzwichtenberg): radix_sort is a stable sorting algorithm,
  // should we check here, whether stable is set to true?

  return true;
}

static void parallel_sort1d_kernel(
    const TensorBase& values,
    const TensorBase& indices) {
  AT_DISPATCH_INTEGRAL_TYPES(values.scalar_type(), "parallel_sort1d_kernel", [&] {
    const auto elements = values.numel();
    auto* const keys = values.data_ptr<scalar_t>();
    auto* const vals = indices.data_ptr<int64_t>();
    std::vector<scalar_t> tmp_keys(elements);
    std::vector<int64_t> tmp_vals(elements);
    const scalar_t* sorted_keys = nullptr;
    const int64_t* sorted_vals = nullptr;
    std::tie(sorted_keys, sorted_vals) = fbgemm::radix_sort_parallel(
        keys,
        vals,
        tmp_keys.data(),
        tmp_vals.data(),
        elements,
        std::numeric_limits<scalar_t>::max(),
        values.scalar_type() != ScalarType::Byte);

    const bool sorted_in_place = keys == sorted_keys;
    if (!sorted_in_place) {
      const auto num_threads = at::get_num_threads();
      at::parallel_for(0, elements, elements / num_threads, [&](int64_t begin, int64_t end) {
        const auto job_size = end - begin;
        vec::map([](vec::Vectorized<scalar_t> x) -> vec::Vectorized<scalar_t> { return x; }, keys + begin, sorted_keys + begin, job_size);
        vec::map([](vec::Vectorized<int64_t> x) -> vec::Vectorized<int64_t> { return x; }, vals + begin, sorted_vals + begin, job_size);
      });
    }
  });
}
#endif

template <typename scalar_t, typename value_accessor_t, typename indices_accessor_t>
static inline void sort_kernel_impl(const value_accessor_t& value_accessor,
            const indices_accessor_t& indices_accessor,
            int64_t dim_size, bool descending, bool stable) {
  auto composite_accessor = CompositeRandomAccessorCPU<
    value_accessor_t, indices_accessor_t
  >(value_accessor, indices_accessor);
  if (descending) {
    if (stable) {
      std::stable_sort(composite_accessor, composite_accessor + dim_size,
        KeyValueCompDesc<scalar_t>());
    } else {
      std::sort(composite_accessor, composite_accessor + dim_size,
        KeyValueCompDesc<scalar_t>());
    }
  } else {
    if (stable) {
      std::stable_sort(composite_accessor, composite_accessor + dim_size,
        KeyValueCompAsc<scalar_t>());
    } else {
      std::sort(composite_accessor, composite_accessor + dim_size,
        KeyValueCompAsc<scalar_t>());
    }
  }
}

static void sort_kernel(
    const TensorBase& self,
    const TensorBase& values,
    const TensorBase& indices,
    int64_t dim,
    bool descending,
    bool stable) {
  dim = maybe_wrap_dim(dim, values.dim());
  _fill_indices(indices, dim);
  if (self.stride(dim) == 0) {
    // check if stride is zero
    // https://github.com/pytorch/pytorch/issues/91420
    return;
  }
#ifdef USE_FBGEMM
  if (can_use_radix_sort(values, descending)) {
    parallel_sort1d_kernel(values, indices);
    return;
  }
#endif
  _dim_apply(
    values, indices, dim,
    "sort_cpu", [&](
      auto* values, int64_t values_dim_stride,
      auto* indices, int64_t indices_dim_stride,
      int64_t dim_size
    ) {
      using scalar_t = std::remove_pointer_t<decltype(values)>;
      if (values_dim_stride == 1 && indices_dim_stride == 1) {
        sort_kernel_impl<
          scalar_t, decltype(values), decltype(indices)
        >(values, indices, dim_size, descending, stable);
      } else if (values_dim_stride == 1 && indices_dim_stride != 1) {
        auto indices_accessor = StridedRandomAccessor<int64_t>(
          indices, indices_dim_stride);
        sort_kernel_impl<
          scalar_t, decltype(values), decltype(indices_accessor)
        >(values, indices_accessor, dim_size, descending, stable);
      } else if (values_dim_stride != 1 && indices_dim_stride == 1) {
        auto values_accessor = StridedRandomAccessor<scalar_t>(
          values, values_dim_stride);
        sort_kernel_impl<
          scalar_t, decltype(values_accessor), decltype(indices)
        >(values_accessor, indices, dim_size, descending, stable);
      } else {
        auto values_accessor = StridedRandomAccessor<scalar_t>(
          values, values_dim_stride);
        auto indices_accessor = StridedRandomAccessor<int64_t>(
          indices, indices_dim_stride);
        sort_kernel_impl<
          scalar_t, decltype(values_accessor), decltype(indices_accessor)
        >(values_accessor, indices_accessor, dim_size, descending, stable);
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
    .add_const_input(self)
    .build();

  auto mode_values_stride = values.strides()[dim];
  auto mode_indices_stride = indices.strides()[dim];
  auto tmp_values_stride = self.strides()[dim];

  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "topk_cpu", [&] {
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

} //at::native
