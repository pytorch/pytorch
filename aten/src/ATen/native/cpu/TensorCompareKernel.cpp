#include <ATen/native/cpu/TensorCompareKernel.h>

#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <c10/util/Optional.h>

namespace at { namespace native { namespace {

template <typename scalar_t, typename index_t>
struct Reduction {
  static void apply(
      Tensor& res,
      Tensor& res_indices,
      const Tensor& self,
      c10::optional<int64_t> dim,
      bool greater) {
    auto out_ = res.data_ptr<scalar_t>();
    auto indices_ = res_indices.data_ptr<index_t>();
    auto data_ = self.data_ptr<scalar_t>();
    auto numel = self.numel();

    int64_t n = self.size(*dim);
    int64_t stride = self.stride(*dim);

    if (n == 1) {
      stride = 1;
      for (int64_t i = self.ndimension() - 1; i > *dim; i--) {
        stride *= self.size(i);
      }
    }
    int64_t batch = numel / (n * stride);
    if (stride == 1) {
      parallel_for(0, batch, 1, [=](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; b++) {
          const scalar_t* data = &data_[b * n];
          scalar_t result = data[0];
          index_t result_index = 0;
          for (int64_t k = 0; k < n; k++) {
            scalar_t value = data[k];
            bool cmp = greater ? (result > value) : (result < value);
            result = cmp ? result : value;
            result_index = cmp ? result_index : k;
            if (_isnan<scalar_t>(result)) {
              break;
            }
          }
          out_[b] = result;
          indices_[b] = result_index;
        }
      });
    } else {
      parallel_for(0, batch * stride, 1, [=](int64_t begin, int64_t end) {
        for (int64_t bi = begin; bi < end; bi++) {
          int64_t b = bi / stride;
          int64_t i = bi % stride;
          const scalar_t* data = &data_[b * n * stride + i];
          scalar_t result = data[0];
          index_t result_index = 0;
          for (int64_t k = 0; k < n; k++) {
            scalar_t value = data[k * stride];
            bool cmp = greater ? (result > value) : (result < value);
            result = cmp ? result : value;
            result_index = cmp ? result_index : k;
            if (_isnan<scalar_t>(result)) {
              break;
            }
          }
          out_[b * stride + i] = result;
          indices_[b * stride + i] = result_index;
        }
      });
    }
  }
};

static void max_kernel_impl(
    Tensor& max,
    Tensor& max_indices,
    const Tensor& self,
    c10::optional<int64_t> dim) {
  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "max", [&] {
    Reduction<scalar_t, int64_t>::apply(max, max_indices, self, dim, true);
  });
}

static void min_kernel_impl(
    Tensor& min,
    Tensor& min_indices,
    const Tensor& self,
    c10::optional<int64_t> dim) {
  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "min", [&] {
    Reduction<scalar_t, int64_t>::apply(min, min_indices, self, dim, false);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(max_kernel, &max_kernel_impl);
REGISTER_DISPATCH(min_kernel, &min_kernel_impl);

}} // namespace at::native
