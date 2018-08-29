#include "DistanceOpsKernel.h"

#include <numeric>
#include <iterator>
#include <algorithm>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"

namespace at { namespace native { namespace {

template<typename scalar_t>
struct PDist {

  static inline scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }

  // Zero norm
  struct zdist_calc {
    static inline void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff != 0.0; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
  };

  // One norm
  struct odist_calc {
    static inline void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static inline scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return grad * sign(diff); }
  };

  // Special general pnorm derivative if p is less than two
  struct lttdist_calc {
    static inline scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1); }
  };

  // Two norm
  struct tdist_calc {
    static inline void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff * diff; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::sqrt(agg); }
    static inline scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : grad * diff / dist; }

  };

  // General p norm
  struct pdist_calc {
    static inline void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += std::pow(diff, p); }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::pow(agg, 1.0 / p); }
    static inline scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : diff * std::pow(std::abs(diff), p - 2) * grad / std::pow(dist, p - 1); }
  };

  // Info norm
  struct idist_calc {
    static inline void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg = std::max(agg, diff); }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static inline scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return grad * sign(diff) * (std::abs(diff) == dist); }
  };

  template <typename F>
  static void run_parallel(Tensor& result, const Tensor& self, const scalar_t p) {
    auto res_ = result.data<scalar_t>();
    auto self_ = self.data<scalar_t>();
    int64_t n = self.size(0);
    int64_t m = self.size(1);

    int64_t combs = n * (n - 1) / 2;
    parallel_for(0, combs, 1, [=](int64_t k, int64_t end) {
      float n2 = n - .5;
      // The -1 accounts for floating point truncation issues
      int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
      for (; k < end; ++k) {
        const scalar_t * a = self_ + i * m;
        const scalar_t * b = self_ + j * m;
        const scalar_t * const stop = a + m;
        scalar_t agg = 0.0;
        for (; a != stop; ++a, ++b) {
          F::inc(agg, std::abs(*a - *b), p);
        }
        res_[k] = F::finish(agg, p);

        ++j;
        if (j == n) {
          ++i;
          j = i + 1;
        }
      }
    });
  }

  // Assumes self is nonempty and 2D
  static void apply(Tensor& result, const Tensor& self, const scalar_t p) {
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

  template <typename F>
  static void run_backward_parallel(Tensor& result, const Tensor & grad, const Tensor & self, const scalar_t p, const Tensor& dist) {
    const int64_t n = self.size(0);
    const int64_t m = self.size(1);
    const int64_t gs = grad.stride(0);

    const scalar_t * const grad_ = grad.data<scalar_t>();
    const scalar_t * const dist_ = dist.data<scalar_t>();
    const scalar_t * const self_ = self.data<scalar_t>();
    scalar_t * const res_ = result.data<scalar_t>();

    at::parallel_for(0, m, 1, [=](int64_t l, int64_t end) {
      const scalar_t * grad_k = grad_;
      const scalar_t * dist_k = dist_;
      const scalar_t * self_l = self_ + l;
      scalar_t * res_l = res_ + l;
      for (; l != end; l += 1, self_l += 1, res_l += 1) {
        const scalar_t * self_i = self_l;
        scalar_t * res_i = res_l;
        for (int64_t i = 0, k = 0; i != n - 1; i += 1, self_i += m, res_i += m) {
          const scalar_t * self_j = self_i + m;
          scalar_t * res_j = res_i + m;
          for (int64_t j = i + 1; j != n; j += 1, k += 1, self_j += m, res_j += m, grad_k += gs, dist_k += 1) {
            const scalar_t res = F::backward(*self_i - *self_j, *grad_k, *dist_k, p);
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
      run_backward_parallel<odist_calc>(result, grad, self, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel<lttdist_calc>(result, grad, self, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel<tdist_calc>(result, grad, self, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel<idist_calc>(result, grad, self, p, dist);
    } else {
      run_backward_parallel<pdist_calc>(result, grad, self, p, dist);
    }
  }

};

}  // anonymous namespace

void pdist_kernel(Tensor& result, const Tensor& self, double p) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist", [&] {
    PDist<scalar_t>::apply(result, self, p);
  });
}

void pdist_backward_kernel(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_backward", [&] {
    PDist<scalar_t>::apply_backward(result, grad, self, p, dist);
  });
}

}}  // namespace at::native
