#include <ATen/native/Distance.h>

#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vml.h>

namespace at { namespace native { namespace {

template<typename scalar_t>
struct Dist {
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
  static void run_parallel_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    const scalar_t * const self_start = self.data<scalar_t>();
    const scalar_t * const self_end = self_start + self.numel();
    int64_t n = self.size(0);
    int64_t m = self.size(1);

    scalar_t * const res_start = result.data<scalar_t>();
    int64_t combs = result.numel(); // n * (n - 1) / 2

    // We conceptually iterate over tuples of (i, j, k) where i is the first
    // vector from the input, j is the second, and k is the result index. This
    // parallelizes over the range of k and infers what i and j are from the
    // value of k.
    parallel_for(0, combs, internal::GRAIN_SIZE / (16 * m), [p, self_start, self_end, n, m, res_start, combs](int64_t k, int64_t end) {
      const Vec pvec(p);
      double n2 = n - .5;
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
  static void apply_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    if (p == 0.0) {
      run_parallel_pdist<zdist_calc>(result, self, p);
    } else if (p == 1.0) {
      run_parallel_pdist<odist_calc>(result, self, p);
    } else if (p == 2.0) {
      run_parallel_pdist<tdist_calc>(result, self, p);
    } else if (std::isinf(p)) {
      run_parallel_pdist<idist_calc>(result, self, p);
    } else {
      run_parallel_pdist<pdist_calc>(result, self, p);
    }
  }

  template <typename F>
  static void run_parallel_cdist(Tensor& result, const Tensor& t1, const Tensor& t2, const scalar_t p) {
    const scalar_t * const t1_start = t1.data<scalar_t>();
    const scalar_t * const t2_start = t2.data<scalar_t>();
    int64_t r1 = t1.size(-2);
    int64_t r2 = t2.size(-2);
    int64_t m = t1.size(-1);

    scalar_t * const res_start = result.data<scalar_t>();
    int64_t total = r1 * r2;

    parallel_for(0, total, internal::GRAIN_SIZE / (16 * m), [=](int64_t start, int64_t end) {
      const Vec pvec(p);
      scalar_t * res = res_start + start;
      const scalar_t * const res_end = res_start + end;

      int64_t k = start;
      while (res != res_end) {
        int64_t i = k / r2;
        int64_t j = k % r2;
        const scalar_t * self_i = t1_start + i * m;
        const scalar_t * self_j = t2_start + j * m;

        *res = F::finish(vec256::map2_reduce_all<scalar_t>(
                [&pvec](Vec a, Vec b) { return F::map((a - b).abs(), pvec); },
                F::red, self_i, self_j, m), p);

        res += 1;
        k++;
      }
    });
  }

  static void apply_cdist(Tensor& result, const Tensor& x1, const Tensor& x2, const scalar_t p) {
    if (p == 0.0) {
      run_parallel_cdist<zdist_calc>(result, x1, x2, p);
    } else if (p == 1.0) {
      run_parallel_cdist<odist_calc>(result, x1, x2, p);
    } else if (p == 2.0) {
      run_parallel_cdist<tdist_calc>(result, x1, x2, p);
    } else if (std::isinf(p)) {
      run_parallel_cdist<idist_calc>(result, x1, x2, p);
    } else {
      run_parallel_cdist<pdist_calc>(result, x1, x2, p);
    }
  }

  // This does a backward pass down a Vec column of the input
  template <typename F>
  inline static void backward_down_column_pdist(const scalar_t * self_i, scalar_t * res_i, const scalar_t * grad_k, const scalar_t * dist_k, const Vec& pvec, int64_t n, int64_t m, int64_t gs, int64_t count = Vec::size()) {
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
  static void run_backward_parallel_pdist(Tensor& result, const Tensor & grad, const Tensor & self, const scalar_t p, const Tensor& dist) {
    const int64_t n = self.size(0);
    const int64_t m = self.size(1);
    const int64_t gs = grad.stride(0);

    const scalar_t * const grad_start = grad.data<scalar_t>();
    const scalar_t * const dist_start = dist.data<scalar_t>();
    const scalar_t * const self_start = self.data<scalar_t>();
    scalar_t * const res_start = result.data<scalar_t>();

    // The only way to parallelize and avoid locking requires parallelizing
    // over the columns of the input, i.e. we compute the gradient for the
    // first section of each vector independentaly of the second section, etc.
    at::parallel_for(0, m / Vec::size(), internal::GRAIN_SIZE / (8 * n * n), [p, n, m, gs, grad_start, dist_start, self_start, res_start](int64_t l, int64_t end) {
      const Vec pvec(p);

      const scalar_t * self_l = self_start + l * Vec::size();
      scalar_t * res_l = res_start + l * Vec::size();

      for (const scalar_t * const res_end = res_start + end * Vec::size(); res_l != res_end; self_l += Vec::size(), res_l += Vec::size()) {
        backward_down_column_pdist<F>(self_l, res_l, grad_start, dist_start, pvec, n, m, gs);
      }
    });
    const int64_t remainder = m % Vec::size();
    if (remainder) {
      backward_down_column_pdist<F>(self_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, Vec(p), n, m, gs, remainder);
    }
  }

  // Assumes self is nonempty, contiguous, and 2D and dist is also contiguous
  static void apply_backward_pdist(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
    result.fill_(0);
    if (p == 0.0) {
    } else if (p == 1.0) {
      run_backward_parallel_pdist<odist_calc>(result, grad, self, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel_pdist<lttdist_calc>(result, grad, self, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel_pdist<tdist_calc>(result, grad, self, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel_pdist<idist_calc>(result, grad, self, p, dist);
    } else {
      run_backward_parallel_pdist<pdist_calc>(result, grad, self, p, dist);
    }
  }

  static void apply_backward_cdist(Tensor& result, const Tensor& grad, const Tensor& x1, const Tensor& x2, const double p, const Tensor& dist) {
    result.fill_(0);
    if (p == 0.0) {
    } else if (p == 1.0) {
      run_backward_parallel_cdist<odist_calc>(result, grad, x1, x2, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel_cdist<lttdist_calc>(result, grad, x1, x2, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel_cdist<tdist_calc>(result, grad, x1, x2, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel_cdist<idist_calc>(result, grad, x1, x2, p, dist);
    } else {
      run_backward_parallel_cdist<pdist_calc>(result, grad, x1, x2, p, dist);
    }
  }


  template <typename F>
  static void run_backward_parallel_cdist(Tensor& result, const Tensor & grad, const Tensor & t1, const Tensor & t2, const scalar_t p, const Tensor& dist) {
    const int64_t r1 = t1.size(-2);
    const int64_t r2 = t2.size(-2);
    const int64_t m = t1.size(-1);
    const int64_t gs = grad.stride(1);

    const scalar_t * const grad_start = grad.data<scalar_t>();
    const scalar_t * const dist_start = dist.data<scalar_t>();
    const scalar_t * const t1_start = t1.data<scalar_t>();
    const scalar_t * const t2_start = t2.data<scalar_t>();
    scalar_t * const res_start = result.data<scalar_t>();

    at::parallel_for(0, m / Vec::size(), internal::GRAIN_SIZE / (16 * r1), [=](int64_t l, int64_t end) {
      const Vec pvec(p);

      const scalar_t * i = t1_start + l * Vec::size();
      const scalar_t * j = t2_start + l * Vec::size();
      scalar_t * res_l = res_start + l * Vec::size();

      for (const scalar_t * const res_end = res_start + end * Vec::size(); res_l != res_end; i += Vec::size(), j += Vec::size(), res_l += Vec::size()) {
        backward_down_column_cdist<F>(i, j, res_l, grad_start, dist_start, pvec, r1, r2, m, gs);
      }
    });
    const int64_t remainder = m % Vec::size();
    if (remainder) {
      backward_down_column_cdist<F>(t1_start + (m - remainder), t2_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, Vec(p), r1, r2, m, gs, remainder);
    }
  }

  template <typename F>
  inline static void backward_down_column_cdist(const scalar_t * t1, const scalar_t * t2, scalar_t * res, const scalar_t * grad_k, const scalar_t * dist_k, const Vec& pvec, int64_t r1, int64_t r2, int64_t m, int64_t gs, int64_t count = Vec::size()) {
    const scalar_t * const t1_end = t1 + m * r1;
    const scalar_t * const t2_end = t2 + m * r2;

    for (; t1 != t1_end; t1 += m, res += m) {
      const Vec vec_t1 = Vec::loadu(t1, count);
      Vec res_vec = Vec::loadu(res, count);

      for (const scalar_t * t2_curr = t2; t2_curr != t2_end; t2_curr += m, grad_k += gs, dist_k += 1) {
        const Vec vec_t2 = Vec::loadu(t2_curr, count);
        Vec res = F::backward(vec_t1 - vec_t2, *grad_k, *dist_k, pvec);
        res_vec = res_vec + res;
      }

      res_vec.store(res, count);
    }
  }

};

void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, const double p) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist", [&] {
    Dist<scalar_t>::apply_pdist(result, self, p);
  });
}

static void pdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_backward", [&] {
    Dist<scalar_t>::apply_backward_pdist(result, grad, self, p, dist);
  });
}

static void cdist_kernel_impl(Tensor& result, const Tensor& x1, const Tensor& x2, const double p) {
  AT_DISPATCH_FLOATING_TYPES(result.type(), "cdist", [&] {
    Dist<scalar_t>::apply_cdist(result, x1, x2, p);
  });
}

static void cdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& x1, const Tensor& x2, const double p, const Tensor& dist) {
  AT_DISPATCH_FLOATING_TYPES(result.type(), "cdist_backward", [&] {
    Dist<scalar_t>::apply_backward_cdist(result, grad, x1, x2, p, dist);
  });
}


}  // anonymous namespace

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl);
REGISTER_DISPATCH(cdist_stub, &cdist_kernel_impl);
REGISTER_DISPATCH(cdist_backward_stub, &cdist_backward_kernel_impl);

}}  // namespace at::native
