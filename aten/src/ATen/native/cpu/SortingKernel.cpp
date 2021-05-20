#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/SortingUtils.h>

namespace at { namespace native {

namespace {

void _fill_indices(Tensor& indices, int64_t dim) {
  auto dim_size = indices.size(dim);
  auto idx_dim = at::arange(0, dim_size, indices.options().dtype(at::kLong));
  auto idx_dim_sizes = std::vector<int64_t>(indices.dim(), 1);
  auto idx_dim_strides = std::vector<int64_t>(indices.dim(), 0);
  idx_dim_sizes[dim] = dim_size;
  idx_dim_strides[dim] = 1;
  auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
  indices.copy_(idx_dim_restrided);
}

template <typename func_t>
void _dim_apply(
    Tensor& values,
    Tensor& indices,
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

  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Bool, ScalarType::Half, iter.dtype(),
    "sorting_kernel_method_name", [&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* values_data_bytes = data[0];
        auto* indices_data_bytes = data[1];

        for (int64_t i = 0; i < n; ++i) {
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
    Tensor& values,
    Tensor& indices,
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
    const Tensor& values,
    const Tensor& indices,
    const Tensor& self,
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

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "topk_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      for (int64_t i = 0; i < n; ++i) {
        TensorAccessor<scalar_t, 1> mode_values(
            reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
            &sizes[dim], &mode_values_stride);
        TensorAccessor<int64_t, 1> mode_indices(
            reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
            &sizes[dim], &mode_indices_stride);
        TensorAccessor<scalar_t, 1> tmp_values(
            reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
            &sizes[dim], &tmp_values_stride);

        auto n = tmp_values.size(0);
        auto use_partial_sort = k * 64 <= n;

        using elem_t = std::pair<scalar_t, int64_t>;
        std::vector<elem_t> queue(n);
        for (int64_t j = 0; j < n; j++) {
          queue[j].first = tmp_values[j];
          queue[j].second = j;
        }

        // we want NaN to be sorted as top for numpy compatibility
        if (use_partial_sort) {
          if (largest) {
            std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((_isnan<scalar_t>(x.first) && !_isnan<scalar_t>(y.first)) || (x.first > y.first));
              });
          } else {
            std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((!_isnan<scalar_t>(x.first) && _isnan<scalar_t>(y.first)) || (x.first < y.first));
              });
          }
        } else {
          if (largest) {
            std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((_isnan<scalar_t>(x.first) && !_isnan<scalar_t>(y.first)) || (x.first > y.first));
              });
            if (sorted) {
              std::sort(queue.begin(), queue.begin() + k - 1,
                [](const elem_t& x, const elem_t& y) -> bool {
                  return ((_isnan<scalar_t>(x.first) && !_isnan<scalar_t>(y.first)) || (x.first > y.first));
                });
            }
          } else {
            std::nth_element(queue.begin(), queue.begin() + k -1, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((!_isnan<scalar_t>(x.first) && _isnan<scalar_t>(y.first)) || (x.first < y.first));
              });
            if (sorted) {
              std::sort(queue.begin(), queue.begin() + k -1,
                [](const elem_t& x, const elem_t& y) -> bool {
                  return ((!_isnan<scalar_t>(x.first) && _isnan<scalar_t>(y.first)) || (x.first < y.first));
                });
            }
          }
        }

        for (int64_t j = 0; j < k; j++) {
          mode_values[j] = queue[j].first;
          mode_indices[j] = queue[j].second;
        }
      }
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    iter.for_each(loop, /*grain_size=*/grain_size);
  });
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(sort_stub, &sort_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(topk_stub, &topk_kernel);

}} //at::native
