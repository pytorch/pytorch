#include "ATen/native/cpu/ReduceOpsKernel.h"

#include <numeric>
#include <iterator>
#include <algorithm>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/core/optional.h"
#include "ATen/cpu/vec256/vec256.h"

namespace at { namespace native { namespace {

using namespace vec256;

static inline int64_t round_down(int64_t a, int64_t m) {
  return a - (a % m);
}

template <typename F>
static void _parallel_for(int64_t size, int64_t step, bool parallelize, F func) {
  if (parallelize) {
    parallel_for(0, size / step, 1, [func, step](int64_t begin, int64_t end) {
      int64_t k = begin * step;
      for (int64_t i = begin; i < end; i++, k += step) {
        func(k);
      }
    });
  } else {
    for (int64_t i = 0; i != size; i += step) {
      func(i);
    }
  }
}

// Vectorized reduction defined by reduce operation `Op` with identity `ident`.
// The reduction is built on top of reduce128, which reduces down a column
// 128 bytes wide (WIDTH scalar elements). The width of 128 bytes is chosen
// because of the "adjacent cache line prefetch" behavior on x86 CPUs.
template<typename scalar_t, template <class> class Op, int ident>
struct Reduction {
  // reduction width in number of scalar elements
  static constexpr int WIDTH = 128 / sizeof(scalar_t);

  using Vec = Vec256<scalar_t>;
  using Reduce = Op<Vec>;
  using ReduceScalar = Op<scalar_t>;

  static void apply(Tensor& res, const Tensor& self, at::optional<int64_t> dim) {
    auto out_ = res.data<scalar_t>();
    auto data_ = self.data<scalar_t>();
    auto numel = self.numel();
    if (!dim.has_value()) {
      *out_ = reduce_all(data_, numel);
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
    int64_t batch = numel / (n * stride);
    bool paralellize = batch * n > internal::GRAIN_SIZE;
    if (stride == 1) {
      parallel_for(0, batch, 1, [=](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; b++) {
          const scalar_t* data = &data_[b * n];
          scalar_t* out = &out_[b];
          scalar_t buf[WIDTH] = {0};
          std::fill(buf, buf + WIDTH, ident);
          int64_t cols_rounded = n / WIDTH;
          reduce128(data, buf, cols_rounded, WIDTH);
          scalar_t result = ident;
          for (int64_t i = 0; i < WIDTH; i++) {
            result = ReduceScalar()(result, buf[i]);
          }
          for (int64_t col = cols_rounded * WIDTH; col != n; col++) {
            result = ReduceScalar()(result, data[col]);
          }
          out_[b] = result;
        }
      });
    } else {
      int64_t rows = n;
      int64_t cols = stride;
      int64_t cols_rounded = round_down(cols, WIDTH);
      int64_t size = cols_rounded;
      parallel_for(
          0,
          batch * (size / WIDTH),
          1,
          [out_, data_, n, stride, rows, cols, cols_rounded, size](
              int64_t begin, int64_t end) {
            for (int64_t bi = begin; bi < end; bi++) {
              int64_t b = bi / (size / WIDTH);
              int64_t i = bi % (size / WIDTH);
              int64_t k = i * WIDTH;
              reduce128(
                  &data_[b * n * stride + k],
                  &out_[b * stride + k],
                  rows,
                  stride);
            }
          });

      _parallel_for(batch, 1, paralellize, [=](int64_t b) {
        const scalar_t* data = &data_[b * n * stride];
        scalar_t* out = &out_[b * stride];
        int64_t rows = n;
        int64_t cols = stride;

        int64_t cols_rounded = round_down(cols, WIDTH);
        if (cols_rounded != cols) {
          scalar_t buf[WIDTH] = {0};
          std::fill(buf, buf + WIDTH, ident);
          for (int64_t row = 0; row != rows; row++) {
            for (int64_t j = 0; j != cols - cols_rounded; j++) {
              auto val = data[row * stride + j + cols_rounded];
              buf[j] = ReduceScalar()(buf[j], val);
            }
          }
          for (int64_t j = 0; j != cols - cols_rounded; j++) {
            out[j + cols_rounded] = buf[j];
          }
        }
      });
    }
  }

  static scalar_t reduce_all(const scalar_t* data, int64_t size) {
    int64_t k = size / WIDTH;

    scalar_t sum = parallel_reduce(
        0,
        k,
        internal::GRAIN_SIZE / WIDTH,
        (scalar_t)ident,
        [data](int64_t begin, int64_t end, scalar_t init) {
          scalar_t buf[WIDTH];
          reduce128(&data[begin * WIDTH], buf, end - begin, WIDTH);
          return std::accumulate(buf, buf + WIDTH, init, ReduceScalar());
        },
        ReduceScalar());

    for (int64_t i = k * WIDTH; i != size; i++) {
      sum = ReduceScalar()(sum, data[i]);
    }
    return sum;
  }

  // Reduce down a column of WIDTH elements (128 bytes) with the given number
  // of rows. Stores the results in out[0 ... WIDTH-1].
  static void reduce128(const scalar_t* data, scalar_t* out, int64_t rows, int64_t stride) {
    Vec acc[4] = {ident, ident, ident, ident};  // 128 bytes (two cache lines)
    static_assert(sizeof(acc) == 128, "accumulator should be 128 bytes");
    for (int64_t row = 0; row != rows; row++) {
      for (int j = 0; j != 4; j++) {
        auto val = Vec::loadu(&data[row * stride + j * Vec::size]);
        acc[j] = Reduce()(acc[j], val);
      }
    }
    for (int j = 0; j != 4; j++) {
      acc[j].store(&out[j * Vec::size]);
    }
  }
};

static void sum_kernel_impl(Tensor& result, const Tensor& self, at::optional<int64_t> dim) {
  AT_DISPATCH_ALL_TYPES(self.type(), "sum", [&] {
    Reduction<scalar_t, std::plus, 0>::apply(result, self, dim);
  });
}

static void prod_kernel_impl(Tensor& result, const Tensor& self, at::optional<int64_t> dim) {
  AT_DISPATCH_ALL_TYPES(self.type(), "prod", [&] {
    Reduction<scalar_t, std::multiplies, 1>::apply(result, self, dim);
  });
}

template<typename scalar_t>
struct NormReduction {
  static void apply(Tensor& res, const Tensor& self, Scalar p, at::optional<int64_t> dim) {
    auto out_ = res.data<scalar_t>();
    auto data_ = self.data<scalar_t>();
    auto numel = self.numel();
    auto pval = 0.0;
    if (p.isIntegral()){
      pval = p.to<int64_t>();
    } else if (p.isFloatingPoint()) {
      pval = p.to<float>();
    }
    if (!dim.has_value()) {
      *out_ = reduce_all(data_, numel,  p);
      //std::cout << "dim has no value, not supported yet, TODO" << std::endl;
      return;
    }

    int64_t n = self.size(*dim);
    int64_t stride = self.stride(*dim);
    //std::cout << "NormReduction called, p = " << p << ", pval = " << pval << ", stride = " << stride << std::endl;
    // A contiguous tensor does not need to hold a meaningful stride
    // if the corresponding size is 1
    if (n == 1) {
      stride = 1;
      for (int64_t i = self.ndimension() - 1; i > *dim; i--) {
        stride *= self.size(i);
      }
    }
    int64_t batch = numel / n;

#if 1
    if (stride == 1) {
      parallel_for(0, batch, 1, [=](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; b++) {
          const scalar_t* data = &data_[b * n];
          scalar_t result = 0.0;
          if (pval == 0) {
            for (int64_t k = 0; k < n; k++) {
              result += (data[k] != 0.0);
            }
            out_[b] = result;
          } else if (pval == 1) {
            for (int64_t k = 0; k < n; k++) {
              result += std::abs(data[k]);
            }
            out_[b] = result;
          } else if (pval == 2) {
            for (int64_t k = 0; k < n; k++) {
              result += data[k] * data[k];
            }
            out_[b] = std::sqrt(result);
          } else if (pval == 3) {
            for (int64_t k = 0; k < n; k++) {
              result += std::abs(data[k] * data[k] * data[k]);
            }
            out_[b] = std::pow(result, 1.0/3);
          } else if (std::isinf(pval)) {
            for (int64_t k = 0; k < n; k++) {
              result = std::abs(data[k]) > result ? std::abs(data[k]) : result;
            }
            out_[b] = result;
          } else {
            for (int64_t k = 0; k < n; k++) {
              result += std::pow(std::abs(data[k]), pval);
            }
            out_[b] = std::pow(result, 1.0/pval);
          }
        }
      });
    } else {
      parallel_for(0, batch, 1, [=](int64_t begin, int64_t end) {
        for (int64_t bi = begin; bi < end; bi++) {
          int64_t b = bi / stride;
          int64_t i = bi % stride;
          const scalar_t* data = &data_[b * n * stride + i];
          scalar_t result = 0.0;
          if (pval == 0) {
            for (int64_t k = 0; k < n; k++) {
              result += (data[k * stride] != 0.0);
            }
            out_[bi] = result;
          } else if (pval == 1) {
            for (int64_t k = 0; k < n; k++) {
              result += std::abs(data[k * stride]);
            }
            out_[bi] = result;
          } else if (pval == 2) {
            for (int64_t k = 0; k < n; k++) {
              result += data[k * stride] * data[k * stride];
            }
            out_[bi] = std::sqrt(result);
          } else if (pval == 3) {
            for (int64_t k = 0; k < n; k++) {
              result += std::abs(data[k * stride] * data[k * stride] * data[k * stride]);
            }
            out_[bi] = std::pow(result, 1.0/3);
          } else if (std::isinf(pval)) {
            for (int64_t k = 0; k < n; k++) {
              result = std::abs(data[k * stride]) > result ? std::abs(data[k * stride]) : result;
            }
            out_[bi] = result;
          } else {
            for (int64_t k = 0; k < n; k++) {
              result += std::pow(std::abs(data[k * stride]), pval);
            }
            out_[bi] = std::pow(result, 1.0/pval);
          }
        }
      });
    }
#endif
  }

  static scalar_t reduce_all(const scalar_t* data_, int64_t size,  Scalar p) {
    auto pval = 0.0;
    if (p.isIntegral()){
      pval = p.to<int64_t>();
    } else if (p.isFloatingPoint()) {
      pval = p.to<float>();
    }
    scalar_t sum = parallel_reduce(
      0,
      size,
      internal::GRAIN_SIZE,
      (scalar_t)0,
      [=](int64_t begin, int64_t end, scalar_t init) {
        const scalar_t* data = &data_[begin];
        int64_t n = end - begin;
        scalar_t result = 0.0;
        if (pval == 0) {
          for (int64_t k = 0; k < n; k++) {
            result += (data[k] != 0.0);
          }
        } else if (pval == 1) {
          for (int64_t k = 0; k < n; k++) {
            result += std::abs(data[k]);
          }
        } else if (pval == 2) {
          for (int64_t k = 0; k < n; k++) {
            result += data[k] * data[k];
          }
          result = std::sqrt(result);
        } else if (pval == 3) {
          for (int64_t k = 0; k < n; k++) {
            result += std::abs(data[k] * data[k] * data[k]);
          }
          result = std::pow(result, 1.0/3);
        } else if (std::isinf(pval)) {
          for (int64_t k = 0; k < n; k++) {
            result = std::abs(data[k]) > result ? std::abs(data[k]) : result;
          }
        } else {
          for (int64_t k = 0; k < n; k++) {
            result += std::pow(std::abs(data[k]), pval);
          }
          result = std::pow(result, 1.0/pval);
        }
        return result;
      },
      std::plus<scalar_t>());
    return sum;
  }

};

static void norm_kernel_impl(Tensor& result, const Tensor& self, Scalar p, at::optional<int64_t> dim) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "norm", [&] {
    NormReduction<scalar_t>::apply(result, self, p, dim);
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(sum_kernel, &sum_kernel_impl);
REGISTER_DISPATCH(prod_kernel, &prod_kernel_impl);
REGISTER_DISPATCH(norm_kernel, &norm_kernel_impl);

}}  // namespace at::native
