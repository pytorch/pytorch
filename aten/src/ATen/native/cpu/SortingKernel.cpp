#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/SortingUtils.h>

namespace at { namespace native {

namespace {

static void topk_kernel(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "topk_cpu", [&] {
    dim_apply(
        {self, values, indices},
        dim,
        [&](int64_t i, TensorList tl) {
          auto tmp_values = tl[0].accessor<scalar_t, 1>();
          auto mode_values = tl[1].accessor<scalar_t, 1>();
          auto mode_indices = tl[2].accessor<int64_t, 1>();

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
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(topk_stub, &topk_kernel);

}} //at::native
