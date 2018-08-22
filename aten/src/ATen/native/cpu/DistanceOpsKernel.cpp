#include "ATen/native/cpu/ReduceOpsKernel.h"

#include <numeric>
#include <iterator>
#include <algorithm>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/core/optional.h"
#include "ATen/cpu/vec256/vec256.h"

namespace at { namespace native { namespace {

template<typename scalar_t>
struct PDist {

  static inline scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }

  static scalar_t zdist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result += *a != *b;
    }
    return result;
  }

  static scalar_t odist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result += std::abs(*a - *b);
    }
    return result;
  }

  static scalar_t tdist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      scalar_t diff = *a - *b;
      result += diff * diff;
    }
    return std::sqrt(result);
  }

  static scalar_t pdist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result += std::pow(std::abs(*a - *b), p);
    }
    return std::pow(result, 1.0 / p);
  }

  static scalar_t idist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result = std::max(result, std::abs(*a - *b));
    }
    return result;
  }

  template <scalar_t (*F)(const scalar_t *, const scalar_t *, const int64_t, const int64_t, const double)>
  static void run_parallel(Tensor& result, const Tensor& self, const double p) {
    auto res_ = result.data<scalar_t>();
    auto self_ = self.data<scalar_t>();
    int64_t n = self.size(0);
    int64_t m = self.size(1);
    int64_t ns = self.stride(0);
    int64_t ms = self.stride(1);

    int64_t combs = n * (n - 1) / 2;
    parallel_for(0, combs, 1, [=](int64_t k, int64_t end) {
      float n2 = n - .5;
      // The -1 accounts for floating point truncation issues
      int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
      for (; k < end; ++k) {
        res_[k] = F(self_ + i * ns, self_ + j * ns, m, ms, p);
        ++j;
        if (j == n) {
          ++i;
          j = i + 1;
        }
      }
    });
  }

  // Assumes self is nonempty and 2D
  static void apply(Tensor& result, const Tensor& self, const double p) {
    if (p == 0.0) {
      run_parallel<zdist_calc>(result, self, p);
    } else if (p == 1.0) {
      run_parallel<odist_calc>(result, self, p);
    } else if (p == 2.0) {
      run_parallel<tdist_calc>(result, self, p);
    } else if (std::isinf(p)) {
      run_parallel<idist_calc>(result, self, p);
    } else {
      run_parallel<pdist_calc>(result, self, p);
    }
  }

  static scalar_t one_backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const double p) {
    return grad * sign(diff);
  }

  static scalar_t lt_two_backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const double p) {
    return dist == 0.0 ? 0 : sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1);
  }

  static scalar_t two_backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const double p) {
    return dist == 0.0 ? 0 : grad * diff / dist;
  }

  static scalar_t gt_two_backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const double p) {
    return dist == 0.0 ? 0 : diff * std::pow(std::abs(diff), p - 2) * grad / std::pow(dist, p - 1);
  }

  static scalar_t inf_backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const double p) {
    return grad * sign(diff) * (std::abs(diff) == dist);
  }

  template <scalar_t (*F)(const scalar_t, const scalar_t, const scalar_t, const double)>
  static void run_backward_parallel(Tensor& result, const Tensor & grad, const Tensor & self, const double p, const Tensor& dist) {
    const int64_t n = self.size(0);
    const int64_t ns = self.stride(0);
    const int64_t m = self.size(1);
    const int64_t ms = self.stride(1);
    const int64_t gs = grad.stride(0);
    const int64_t ds = dist.stride(0);

    const scalar_t * const grad_ = grad.data<scalar_t>();
    const scalar_t * const dist_ = dist.data<scalar_t>();
    const scalar_t * const self_ = self.data<scalar_t>();
    scalar_t * const res_ = result.data<scalar_t>();

    at::parallel_for(0, m, 1, [=](int64_t l, int64_t end) {
      const scalar_t * grad_k = grad_;
      const scalar_t * dist_k = dist_;
      const scalar_t * self_l = self_ + l * ms;
      scalar_t * res_l = res_ + l;
      for (; l != end; l += 1, self_l += ms, res_l += 1) {
        const scalar_t * self_i = self_l;
        scalar_t * res_i = res_l;
        for (int64_t i = 0, k = 0; i != n - 1; i += 1, self_i += ns, res_i += m) {
          const scalar_t * self_j = self_i + ns;
          scalar_t * res_j = res_i + m;
          for (int64_t j = i + 1; j != n; j += 1, k += 1, self_j += ns, res_j += m, grad_k += gs, dist_k += ds) {
            const scalar_t res = F(*self_i - *self_j, *grad_k, *dist_k, p);
            *res_i += res;
            *res_j -= res;
          }
        }
      }
    });
  }

  static void apply_backward(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
    if (p == 0.0) {
    } else if (p == 1.0) {
      run_backward_parallel<one_backward>(result, grad, self, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel<lt_two_backward>(result, grad, self, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel<two_backward>(result, grad, self, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel<inf_backward>(result, grad, self, p, dist);
    } else {
      run_backward_parallel<gt_two_backward>(result, grad, self, p, dist);
    }
  }

};

static void pdist_kernel_impl(Tensor& result, const Tensor& self, double p) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist", [&] {
    PDist<scalar_t>::apply(result, self, p);
  });
}

static void pdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_backward", [&] {
    PDist<scalar_t>::apply_backward(result, grad, self, p, dist);
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(pdist_kernel, &pdist_kernel_impl);
REGISTER_DISPATCH(pdist_backward_kernel, &pdist_backward_kernel_impl);

}}  // namespace at::native
