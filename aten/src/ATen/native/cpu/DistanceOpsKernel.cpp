#include <ATen/native/Distance.h>

#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vml.h>

namespace at { namespace native { namespace {

template<typename scalar_t>
struct PDist {
  using Vec = vec256::Vec256<scalar_t>;

  // Depending on the value of the pnorm, there are specific implementations
  // that are much faster than std::pow(std::abs(a - b), p), but have the same
  // standard loop code for how to process the input vector. To reuse the main
  // outside loop while still guaranteeing that the compiler inlines every
  // different function on p, we break the inner norm logic into structs with
  // static functions that represent what's done differently, and template the
  // outer loop on those structs.
  //
  // The four functions are:
  //     map :      This tells how to modify (a - b) to form the component that
  //                gets summed.
  //     red :      This tells how to sum the result of map up. This is
  //                separate because the inf norm actuall uses max instead of
  //                sum.
  //     finish :   This tells what to do with the aggregated value to compute
  //                the norm. Generally this is the result of val ^ (1 / p).
  //     backward : This is the gradient for that norm. Arguments are pretty
  //                self explanitory.
  //
  // There are a few cases where these aren't used. The 0 norm has no backward,
  // because it's always 0, so that's shortcircuited earlier. There's a special
  // implementation of the general backward pass when p is less than two, so
  // there's a struct with only a backward pass for this case.

  // TODO This is an inefficient way to compite sign, and can be much faster
  // using native SSE instructions that should be added to Vec256.
  static inline Vec sign(Vec val) {
    return vec256::minimum(vec256::maximum(Vec(0), val.ceil()), Vec(1)) +
      vec256::minimum(vec256::maximum(Vec(-1), val.floor()), Vec(0));
  }

  // Zero norm
  struct zdist_calc {
    static inline Vec map(const Vec& diff, const Vec& p) { return vec256::minimum(diff.abs().ceil(), Vec(1)); }
    static inline Vec red(const Vec& agg, const Vec& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
  };

  // One norm
  struct odist_calc {
    static inline Vec map(const Vec& diff, const Vec& p) { return diff; }
    static inline Vec red(const Vec& agg, const Vec& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return Vec(grad) * sign(diff); }
  };

  // Special general pnorm derivative if p is less than two
  struct lttdist_calc {
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return dist == 0.0 ? Vec(0) : sign(diff) * diff.abs().pow(p - Vec(1)) * Vec(grad) / Vec(dist).pow(p - Vec(1)); }
  };

  // Two norm
  struct tdist_calc {
    // TODO This can probably use fused add multiply to get better perf
    static inline Vec map(const Vec& diff, const Vec& p) { return diff * diff; }
    static inline Vec red(const Vec& agg, const Vec& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::sqrt(agg); }
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return dist == 0.0 ? Vec(0) : Vec(grad) * diff / Vec(dist); }
  };

  // General p norm
  struct pdist_calc {
    static inline Vec map(const Vec& diff, const Vec& p) { return diff.pow(p); }
    static inline Vec red(const Vec& agg, const Vec& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::pow(agg, 1.0 / p); }
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return dist == 0.0 ? Vec(0) : diff * diff.abs().pow(p - Vec(2)) * Vec(grad) / Vec(dist).pow(p - Vec(1)); }
  };

  // Info norm
  struct idist_calc {
    static inline Vec map(const Vec& diff, const Vec& p) { return diff; }
    static inline Vec red(const Vec& agg, const Vec& up) { return vec256::maximum(agg, up); }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    // TODO This backward pass uses a very complext expression to compute (diff
    // == dist) that could be much faster if using SSE instructions.
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return Vec(grad) * sign(diff) * (Vec(1) - vec256::minimum(Vec(1), (diff.abs() - Vec(dist)).abs().ceil())); }
  };

  template <typename F>
  static void run_parallel(Tensor& result, const Tensor& self, const scalar_t p) {
    const scalar_t * const self_start = self.data<scalar_t>();
    const scalar_t * const self_end = self_start + self.numel();
    int64_t n = self.size(0);
    int64_t m = self.size(1);

    scalar_t * const res_start = result.data<scalar_t>();
    int64_t combs = result.numel(); // n * (n - 1) / 2
    const Vec pvec(p);

    // We conceptually iterate over tuples of (i, j, k) where i is the first
    // vector from the input, j is the second, and k is the result index. This
    // parallelizes over the range of k and infers what i and j are from the
    // value of k.
    parallel_for(0, combs, internal::GRAIN_SIZE / (16 * m), [=, &pvec](int64_t k, int64_t end) {
      float n2 = n - .5;
      // The -1 accounts for floating point truncation issues
      int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

      const scalar_t * self_i = self_start + i * m;
      const scalar_t * self_j = self_start + j * m;
      scalar_t * res = res_start + k;
      const scalar_t * const res_end = res_start + end;

      while (res != res_end) {
        *res = F::finish(vec256::map2_reduce_all<scalar_t>(
          [&pvec](Vec a, Vec b) { return F::map((a - b).abs(), pvec); },
          F::red, self_i, self_j, m), p);

        res += 1;
        self_j += m;
        if (self_j == self_end) {
          self_i += m;
          self_j = self_i + m;
        }
      }
    });
  }

  // Assumes self is nonempty, contiguous, and 2D
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
  inline static void backward_down_column(const scalar_t * self_i, scalar_t * res_i, const scalar_t * grad_k, const scalar_t * dist_k, const Vec& pvec, int64_t n, int64_t m, int64_t gs, int64_t count = Vec::size()) {
    for (const scalar_t * const self_end = self_i + m * n; self_i != self_end - m; self_i += m, res_i += m) {

      const Vec self_vec_i = Vec::loadu(self_i, count);
      Vec res_vec_i = Vec::loadu(res_i, count);

      const scalar_t * self_j = self_i + m;
      scalar_t * res_j = res_i + m;
      for (; self_j != self_end; self_j += m, res_j += m, grad_k += gs, dist_k += 1) {
        const Vec self_vec_j = Vec::loadu(self_j, count);
        Vec res_vec_j = Vec::loadu(res_j, count);

        Vec res = F::backward(self_vec_i - self_vec_j, *grad_k, *dist_k, pvec);
        res_vec_i = res_vec_i + res;
        res_vec_j = res_vec_j - res;

        res_vec_j.store(res_j, count);
      }

      res_vec_i.store(res_i, count);
    }
  }

  template <typename F>
  static void run_backward_parallel(Tensor& result, const Tensor & grad, const Tensor & self, const scalar_t p, const Tensor& dist) {
    const int64_t n = self.size(0);
    const int64_t m = self.size(1);
    const int64_t gs = grad.stride(0);
    const Vec pvec(p);

    const scalar_t * const grad_start = grad.data<scalar_t>();
    const scalar_t * const dist_start = dist.data<scalar_t>();
    const scalar_t * const self_start = self.data<scalar_t>();
    scalar_t * const res_start = result.data<scalar_t>();

    // The only way to parallelize and avoid locking requires parallelizing
    // over the columns of the input, i.e. we compute the gradient for the
    // first section of each vector independentaly of the second section, etc.
    at::parallel_for(0, m / Vec::size(), internal::GRAIN_SIZE / (8 * n * n), [=, &pvec](int64_t l, int64_t end) {
      const scalar_t * self_l = self_start + l * Vec::size();
      scalar_t * res_l = res_start + l * Vec::size();

      for (const scalar_t * const res_end = res_start + end * Vec::size(); res_l != res_end; self_l += Vec::size(), res_l += Vec::size()) {
        backward_down_column<F>(self_l, res_l, grad_start, dist_start, pvec, n, m, gs);
      }
    });
    const int64_t remainder = m % Vec::size();
    if (remainder) {
      backward_down_column<F>(self_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, pvec, n, m, gs, remainder);
    }
  }

  // Assumes self is nonempty, contiguous, and 2D and dist is also contiguous
  static void apply_backward(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
    result.fill_(0);
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

void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, const double p) {
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

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl);

}}  // namespace at::native
