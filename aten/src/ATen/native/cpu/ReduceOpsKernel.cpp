#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <c10/util/Optional.h>

namespace at { namespace native { namespace {

using namespace vec256;

static void sum_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "sum", [&] {
    binary_kernel_reduce_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a + b; });
  });
}

static void mean_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "mean", [&] {
    scalar_t factor = scalar_t(iter.num_output_elements()) / iter.numel();
    binary_kernel_reduce(
      iter,
      MeanOps<scalar_t, scalar_t> {factor},
      scalar_t(0)
    );
  });
}

static void std_kernel_impl(TensorIterator &iter, bool unbiased) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.type(), "std", [&] {
    binary_kernel_reduce(
      iter,
      WelfordOps<scalar_t, double> { unbiased },
      WelfordData<double>()
    );
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "prod", [&] {
    binary_kernel_reduce_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a * b; },
      /*identity=*/1);
  });
}

static inline int64_t round_down(int64_t a, int64_t m) {
  return a - (a % m);
}

template<typename scalar_t>
struct NormReduction {
  // reduction width in number of scalar elements
  static constexpr int WIDTH = 128 / sizeof(scalar_t);
  using Vec = Vec256<scalar_t>;

  static void apply(
      Tensor& res,
      const Tensor& self,
      Scalar p,
      c10::optional<int64_t> dim) {
    auto out_ = res.data<scalar_t>();
    auto data_ = self.data<scalar_t>();
    auto numel = self.numel();
    float pval = 0.0;
    if (p.isIntegral()){
      pval = p.to<int64_t>();
    } else if (p.isFloatingPoint()) {
      pval = p.to<float>();
    }
    if (!dim.has_value()) {
      *out_ = reduce_all(data_, numel,  pval);
      return;
    }
    int64_t n = self.size(*dim);
    int64_t stride = self.stride(*dim);
    // A contiguous tensor does not need to hold a meaningful stride
    // if the corresponding size is 1
    if (n == 1) {
      stride = 1;
      for (int64_t i = self.ndimension() - 1; i > *dim; i--) {
        stride *= self.size(i);
      }
    }
    int64_t batch = numel / n;
    parallel_for(0, batch, 1, [=](int64_t begin, int64_t end) {
      for (int64_t bi = begin; bi < end; bi++) {
        int64_t b = bi / stride;
        int64_t i = bi % stride;
        const scalar_t* data = &data_[b * n * stride + i];
        out_[bi] = norm_reduce(data, n, stride, pval);
      }
    });
  }

  static scalar_t reduce_all(const scalar_t* data_, int64_t size,  float pval) {
    scalar_t sum = parallel_reduce(
      0,
      size,
      internal::GRAIN_SIZE,
      (scalar_t)0,
      [=](int64_t begin, int64_t end, scalar_t init) {
        const scalar_t* data = &data_[begin];
        int64_t n = end - begin;
        scalar_t result = norm_reduce(data, n, 1, pval);
        return result;
      },
      std::plus<scalar_t>());
    return sum;
  }

  static scalar_t norm_reduce(const scalar_t* data, int64_t n, int64_t stride, float pval) {
    scalar_t result = 0.0;
    if (stride == 1 && (pval == 1 || pval == 2 || pval == 3) && n >= WIDTH) {
      int64_t n_rounded = round_down(n, WIDTH);
      scalar_t result1 = norm_reduce128(data, n_rounded, pval);
      scalar_t result2 = norm_reduce_sequential(data + n_rounded, n - n_rounded, stride, pval);
      result = std::pow(std::pow(result1, pval) + std::pow(result2, pval), 1.0/pval);
    } else {
      result = norm_reduce_sequential(data, n, stride, pval);
    }
    return result;
  }

  static scalar_t norm_reduce_sequential(const scalar_t* data, int64_t n, int64_t stride, float pval) {
    scalar_t result = 0.0;
    if (pval == 0) {
      for (int64_t k = 0; k < n; k++) {
        result += (data[k * stride] != 0.0);
      }
    } else if (pval == 1) {
      for (int64_t k = 0; k < n; k++) {
        result += std::abs(data[k * stride]);
      }
    } else if (pval == 2) {
      for (int64_t k = 0; k < n; k++) {
        result += data[k * stride] * data[k * stride];
      }
      result = std::sqrt(result);
    } else if (pval == 3) {
      for (int64_t k = 0; k < n; k++) {
        result += std::abs(data[k * stride] * data[k * stride] * data[k * stride]);
      }
      result = std::pow(result, 1.0/3);
    } else if (pval == INFINITY) {
      for (int64_t k = 0; k < n; k++) {
        result = std::abs(data[k * stride]) > result ? std::abs(data[k * stride]) : result;
      }
    } else if (pval == -INFINITY) {
      result = INFINITY;
      for (int64_t k = 0; k < n; k++) {
        result = std::abs(data[k * stride]) < result ? std::abs(data[k * stride]) : result;
      }
    } else {
      for (int64_t k = 0; k < n; k++) {
        result += std::pow(std::abs(data[k * stride]), pval);
      }
      result = std::pow(result, 1.0/pval);
    }
    return result;
  }

  // Reduce down a column of WIDTH elements (128 bytes) with the given number n
  // n is already rounded by 128
  static scalar_t norm_reduce128(const scalar_t* data, int64_t n, float pval) {
    scalar_t result = 0.0;
    Vec acc[4] = {0.0, 0.0, 0.0, 0.0};  // 128 bytes (two cache lines)
    static_assert(sizeof(acc) == 128, "accumulator should be 128 bytes");
    int64_t rows = n / WIDTH;
    if (pval == 1){
      for (int row = 0; row < rows; row ++) {
        for (int j = 0; j != 4; j++) {
          auto val = Vec::loadu(&data[row * WIDTH + j * Vec::size()]);
          acc[j] = acc[j] + val.abs();
        }
      }
    }
    else if (pval == 2) {
      for (int row = 0; row < rows; row ++) {
        for (int j = 0; j != 4; j++) {
          auto val = Vec::loadu(&data[row * WIDTH + j * Vec::size()]);
          acc[j] = acc[j] + val * val;
        }
      }
    }
    else if (pval == 3) {
      for (int row = 0; row < rows; row ++) {
        for (int j = 0; j != 4; j++) {
          auto val = Vec::loadu(&data[row * WIDTH + j * Vec::size()]);
          acc[j] = acc[j] + (val * val * val).abs();
        }
      }
    }
    scalar_t buf[WIDTH] = {0};
    for (int j = 0; j != 4; j++) {
      acc[j].store(&buf[j * Vec::size()]);
    }
    for (int i = 0; i < WIDTH; i++) {
      result += buf[i];
    }
    result = std::pow(result, 1.0/pval);
    return result;
  }
};

static void norm_kernel_impl(
    Tensor& result,
    const Tensor& self,
    Scalar p,
    c10::optional<int64_t> dim) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "norm", [&] {
    NormReduction<scalar_t>::apply(result, self, p, dim);
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);
REGISTER_DISPATCH(std_stub, &std_kernel_impl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);
REGISTER_DISPATCH(norm_kernel, &norm_kernel_impl);
REGISTER_DISPATCH(mean_stub, &mean_kernel_impl);

}}  // namespace at::native
