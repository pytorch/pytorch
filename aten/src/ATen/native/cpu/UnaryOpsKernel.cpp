#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/functional.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/native/cpu/CapabilityDispatch.h"

// [Note AVX-SSE transitions] In general we avoid calls into cmath for code
// compiled with AVX/AVX2 This is because of SSE-AVX transitions and a bug in
// Glibc2.23 See https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280
// Calling zeroupper when using AVX/AVX2 code resolves this.
#if defined(__AVX__) && defined(__GLIBC__) && __GLIBC_MINOR__ == 23
#define DL_RUNTIME_BUG(opfn, type) \
  volatile type x = (type)(1);     \
  x = opfn(x);                     \
  _mm256_zeroall();
#else
#define DL_RUNTIME_BUG(opfn, type)
#endif

namespace at {
namespace native {
namespace {

template <typename scalar_t, typename VecOp, typename ScalarOp>
inline void _parallel_vector_map_(
    VecOp vec_op,
    ScalarOp scalar_op,
    Tensor& self) {
  DL_RUNTIME_BUG(scalar_op, scalar_t);
  CPU_tensor_parallel_kernel_apply1<scalar_t>(
      self, [vec_op, scalar_op](int64_t size, scalar_t* x, int64_t stridex) {
        if (stridex == 1) {
          vec256::map_(vec_op, x, size);
        } else {
          for (int64_t i = 0; i < size; i++) {
            x[stridex * i] = scalar_op(x[i * stridex]);
          }
        }
      });
}

template <typename scalar_t, typename VecOp, typename ScalarOp>
inline void _parallel_vector_map(
    VecOp vec_op,
    ScalarOp scalar_op,
    Tensor& result,
    const Tensor& self) {
  DL_RUNTIME_BUG(scalar_op, scalar_t);
  CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
      result,
      self,
      [vec_op, scalar_op](
          int64_t size,
          scalar_t* x,
          scalar_t* y,
          int64_t stridex,
          int64_t stridey) {
        if (stridex == 1 && stridey == 1) {
          vec256::map(vec_op, x, y, size);
        } else {
          for (int64_t i = 0; i < size; i++) {
            x[stridex * i] = scalar_op(y[i * stridey]);
          }
        }
      });
}

static void clamp_max__kernel(Tensor& self, Scalar& max_) {
  AT_DISPATCH_ALL_TYPES(self.type(), "clamp_max_", [&] {
    const scalar_t max_val = max_.to<scalar_t>();
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map_<scalar_t>(
        [max_val](const Vec& x) { return vec256::min(Vec(max_val), x); },
        [max_val](const scalar_t x) { return std::min(max_val, x); },
        self);
  });
}

static void clamp_min__kernel(Tensor& self, Scalar& min_) {
  AT_DISPATCH_ALL_TYPES(self.type(), "clamp_min_", [&] {
    const scalar_t min_val = min_.to<scalar_t>();
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map_<scalar_t>(
        [min_val](const Vec& x) { return vec256::max(Vec(min_val), x); },
        [min_val](const scalar_t y) { return std::max(min_val, y); },
        self);
  });
}

static void clamp_max_kernel(Tensor& result, const Tensor& self, Scalar& max_) {
  AT_DISPATCH_ALL_TYPES(self.type(), "clamp_max", [&] {
    const scalar_t max_val = max_.to<scalar_t>();
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map<scalar_t>(
        [max_val](const Vec& x) { return vec256::min(Vec(max_val), x); },
        [max_val](const scalar_t y) { return std::min(max_val, y); },
        result,
        self);
  });
}

static void clamp_min_kernel(Tensor& result, const Tensor& self, Scalar& min_) {
  AT_DISPATCH_ALL_TYPES(self.type(), "clamp_min", [&] {
    const scalar_t min_val = min_.to<scalar_t>();
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map<scalar_t>(
        [min_val](const Vec& x) { return vec256::max(Vec(min_val), x); },
        [min_val](const scalar_t y) { return std::max(min_val, y); },
        result,
        self);
  });
}

static void sigmoid__kernel(Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sigmoid_", [&] {
    DL_RUNTIME_BUG(std::exp, scalar_t);
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map_<scalar_t>(
        [](const Vec& x) { return vec256::sigmoid(x); },
        [](const scalar_t y_) {
          scalar_t y = -y_;
          y = std::exp(y);
          y = ((scalar_t)1) + y;
          return ((scalar_t)1.0) / y;
        },
        self);
  });
}

static void sigmoid_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sigmoid", [&] {
    DL_RUNTIME_BUG(std::exp, scalar_t);
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map<scalar_t>(
        [](const Vec& x) { return vec256::sigmoid(x); },
        [](const scalar_t y_) {
          scalar_t y = -y_;
          y = std::exp(y);
          y = ((scalar_t)1) + y;
          return ((scalar_t)1.0) / y;
        },
        result,
        self);
  });
}

static void frac__kernel(Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "frac_", [&] {
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map_<scalar_t>(
        [](const Vec& x) { return vec256::frac(x); },
        [](const scalar_t y) { return y - std::trunc(y); },
        self);
  });
}

static void frac_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "frac", [&] {
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map<scalar_t>(
        [](const Vec& x) { return vec256::frac(x); },
        [](const scalar_t y) { return y - std::trunc(y); },
        result,
        self);
  });
}

static void clamp__kernel(Tensor& self, Scalar& min_, Scalar& max_) {
  AT_DISPATCH_ALL_TYPES(self.type(), "clamp_", [&] {
    const scalar_t min_val = min_.to<scalar_t>();
    const scalar_t max_val = max_.to<scalar_t>();
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map_<scalar_t>(
        [min_val, max_val](const Vec& x) {
          Vec max_vec = Vec(max_val);
          Vec min_vec = Vec(min_val);
          return at::vec256::max(min_vec, at::vec256::min(max_vec, x));
        },
        [min_val, max_val](const scalar_t x) {
          return std::max(min_val, std::min(max_val, x));
        },
        self);
  });
}

static void clamp_kernel(
    Tensor& result,
    const Tensor& self,
    Scalar& min_,
    Scalar& max_) {
  AT_DISPATCH_ALL_TYPES(self.type(), "clamp", [&] {
    const scalar_t min_val = min_.to<scalar_t>();
    const scalar_t max_val = max_.to<scalar_t>();
    using Vec = vec256::Vec256<scalar_t>;
    _parallel_vector_map<scalar_t>(
        [min_val, max_val](const Vec& x) {
          Vec max_vec = Vec(max_val);
          Vec min_vec = Vec(min_val);
          return at::vec256::max(min_vec, at::vec256::min(max_vec, x));
        },
        [min_val, max_val](const scalar_t y) {
          return std::max(min_val, std::min(max_val, y));
        },
        result,
        self);
  });
}

#define IMPLEMENT_COMPUTEBOUND_KERNEL(types, op, opfn)                     \
  static void op##_kernel(Tensor& result, const Tensor& self) {            \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                    \
      static constexpr int WIDTH = 128 / sizeof(scalar_t);                 \
      DL_RUNTIME_BUG(opfn, scalar_t);                                      \
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(               \
          result,                                                          \
          self,                                                            \
          [](int64_t size,                                                 \
             scalar_t* x,                                                  \
             scalar_t* y,                                                  \
             int64_t stridex,                                              \
             int64_t stridey) {                                            \
            using Vec = vec256::Vec256<scalar_t>;                          \
            if (stridex == 1 && stridey == 1) {                            \
              vec256::map(                                                 \
                  [](const Vec& x) { return vec256::op(x); }, x, y, size); \
            } else {                                                       \
              int64_t i = 0;                                               \
              if (size > WIDTH) {                                          \
                for (; i < size - size % WIDTH; i += WIDTH) {              \
                  scalar_t buffer[WIDTH];                                  \
                  for (int64_t j = 0; j < WIDTH; j++)                      \
                    buffer[j] = y[stridey * (j + i)];                      \
                  vec256::map_(                                            \
                      [](const Vec& x) { return vec256::op(x); },          \
                      buffer,                                              \
                      WIDTH);                                              \
                  for (int64_t j = 0; j < WIDTH; j++)                      \
                    x[stridex * (j + i)] = buffer[j];                      \
                }                                                          \
              }                                                            \
              for (; i < size; i++) {                                      \
                x[stridex * i] = opfn(y[stridey * i]);                     \
              }                                                            \
            }                                                              \
          });                                                              \
    });                                                                    \
  }                                                                        \
  static void op##__kernel(Tensor& self) {                                 \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                    \
      static constexpr int WIDTH = 128 / sizeof(scalar_t);                 \
      DL_RUNTIME_BUG(opfn, scalar_t);                                      \
      CPU_tensor_parallel_kernel_apply1<scalar_t>(                         \
          self, [](int64_t size, scalar_t* x, int64_t stridex) {           \
            using Vec = vec256::Vec256<scalar_t>;                          \
            if (stridex == 1) {                                            \
              vec256::map_(                                                \
                  [](const Vec& x) { return vec256::op(x); }, x, size);    \
            } else {                                                       \
              int64_t i = 0;                                               \
              if (size > WIDTH) {                                          \
                for (; i < size - size % WIDTH; i += WIDTH) {              \
                  scalar_t buffer[WIDTH];                                  \
                  for (int64_t j = 0; j < WIDTH; j++)                      \
                    buffer[j] = x[stridex * (j + i)];                      \
                  vec256::map_(                                            \
                      [](const Vec& x) { return vec256::op(x); },          \
                      buffer,                                              \
                      WIDTH);                                              \
                  for (int64_t j = 0; j < WIDTH; j++)                      \
                    x[stridex * (j + i)] = buffer[j];                      \
                }                                                          \
              }                                                            \
              for (; i < size; i++) {                                      \
                x[stridex * i] = opfn(x[stridex * i]);                     \
              }                                                            \
            }                                                              \
          });                                                              \
    });                                                                    \
  }                                                                        \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)                                \
  REGISTER_DISPATCH(op##_Impl, &op##__kernel)

#define IMPLEMENT_KERNEL(types, op, opfn)                       \
  static void op##_kernel(Tensor& result, const Tensor& self) { \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {         \
      using Vec = vec256::Vec256<scalar_t>;                     \
      _parallel_vector_map<scalar_t>(                           \
          [](const Vec& x) { return vec256::op(x); },           \
          [](const scalar_t y) { return opfn(y); },             \
          result,                                               \
          self);                                                \
    });                                                         \
  }                                                             \
  static void op##__kernel(Tensor& self) {                      \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {         \
      using Vec = vec256::Vec256<scalar_t>;                     \
      _parallel_vector_map_<scalar_t>(                          \
          [](const Vec& x) { return vec256::op(x); },           \
          [](const scalar_t y) { return opfn(y); },             \
          self);                                                \
    });                                                         \
  }                                                             \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)                     \
  REGISTER_DISPATCH(op##_Impl, &op##__kernel)

#define IMPLEMENT_KERNEL_LOOP(types, op, opfn)                          \
  static void op##_kernel(Tensor& result, const Tensor& self) {         \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                 \
      DL_RUNTIME_BUG(opfn, scalar_t);                                   \
      CPU_tensor_parallel_apply2<scalar_t, scalar_t>(                   \
          result, self, [](scalar_t& x, scalar_t& y) { x = opfn(y); }); \
    });                                                                 \
  }                                                                     \
  static void op##__kernel(Tensor& self) {                              \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                 \
      DL_RUNTIME_BUG(opfn, scalar_t);                                   \
      CPU_tensor_parallel_apply1<scalar_t>(                             \
          self, [](scalar_t& x) { x = opfn(x); });                      \
    });                                                                 \
  }                                                                     \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)                             \
  REGISTER_DISPATCH(op##_Impl, &op##__kernel)

} // anonymous namespace

REGISTER_DISPATCH(clamp_Impl, &clamp__kernel);
REGISTER_DISPATCH(clampMax_Impl, &clamp_max__kernel);
REGISTER_DISPATCH(clampMin_Impl, &clamp_min__kernel);
REGISTER_DISPATCH(clampImpl, &clamp_kernel);
REGISTER_DISPATCH(clampMaxImpl, &clamp_max_kernel);
REGISTER_DISPATCH(clampMinImpl, &clamp_min_kernel);
REGISTER_DISPATCH(frac_Impl, &frac__kernel);
REGISTER_DISPATCH(fracImpl, &frac_kernel);
REGISTER_DISPATCH(sigmoid_Impl, &sigmoid__kernel);
REGISTER_DISPATCH(sigmoidImpl, &sigmoid_kernel);

IMPLEMENT_KERNEL(ALL, abs, std::abs)
IMPLEMENT_KERNEL(FLOATING, acos, std::acos)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, asin, std::asin)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, atan, std::atan)
IMPLEMENT_KERNEL(FLOATING, ceil, std::ceil)
IMPLEMENT_KERNEL(FLOATING, erf, std::erf)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, exp, std::exp)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, expm1, std::expm1)
IMPLEMENT_KERNEL(FLOATING, floor, std::floor)
IMPLEMENT_KERNEL(FLOATING, log, std::log)
IMPLEMENT_KERNEL(FLOATING, log10, std::log10)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, log1p, std::log1p)
IMPLEMENT_KERNEL(FLOATING, log2, std::log2)
IMPLEMENT_KERNEL(ALL, neg, -)
IMPLEMENT_KERNEL(FLOATING, reciprocal, 1 /)
IMPLEMENT_KERNEL(FLOATING, round, std::round)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, rsqrt, 1 / std::sqrt)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, sqrt, std::sqrt)
IMPLEMENT_COMPUTEBOUND_KERNEL(FLOATING, tanh, std::tanh)
IMPLEMENT_KERNEL(FLOATING, trunc, std::trunc)

IMPLEMENT_KERNEL_LOOP(FLOATING, cos, std::cos)
IMPLEMENT_KERNEL_LOOP(FLOATING, cosh, std::cosh)
IMPLEMENT_KERNEL_LOOP(FLOATING, sin, std::sin)
IMPLEMENT_KERNEL_LOOP(FLOATING, sinh, std::sinh)
IMPLEMENT_KERNEL_LOOP(FLOATING, tan, std::tan)

} // namespace native
} // namespace at
