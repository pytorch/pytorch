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
#define ZEROUPPER _mm256_zeroupper();
#else
#define ZEROUPPER
#endif

namespace at {
namespace native {
namespace {

static void clamp_max_kernel(Tensor& result, const Tensor& self, Scalar& max_) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "clamp_max", [&] {
    const scalar_t max_val = max_.to<scalar_t>();
    CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
        result,
        self,
        [max_val](
            int64_t size,
            scalar_t* x,
            scalar_t* y,
            int64_t stridex,
            int64_t stridey) {
          if (stridex == 1 && stridey == 1) {
            using Vec = vec256::Vec256<scalar_t>;
            vec256::map(
                [max_val](const Vec& x) {
                  return vec256::min(Vec(max_val), x);
                },
                x,
                y,
                size);
          } else {
            for (int64_t i = 0; i < size; i++) {
              ZEROUPPER
              x[stridex * i] = std::min(max_val, y[stridey * i]);
            }
          }
        });
  });
}

static void clamp_min_kernel(Tensor& result, const Tensor& self, Scalar& min_) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "clamp_min", [&] {
    const scalar_t min_val = min_.to<scalar_t>();
    CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
        result,
        self,
        [min_val](
            int64_t size,
            scalar_t* x,
            scalar_t* y,
            int64_t stridex,
            int64_t stridey) {
          if (stridex == 1 && stridey == 1) {
            using Vec = vec256::Vec256<scalar_t>;
            vec256::map(
                [min_val](const Vec& x) {
                  return vec256::max(Vec(min_val), x);
                },
                x,
                y,
                size);
          } else {
            for (int64_t i = 0; i < size; i++) {
              ZEROUPPER
              x[stridex * i] = std::max(min_val, y[stridey * i]);
            }
          }
        });
  });
}

static void clamp_kernel(
    Tensor& result,
    const Tensor& self,
    Scalar& min_,
    Scalar& max_) {
  if (at::isFloatingType(self.type().scalarType())) {
std::cerr << "HERE" << std::endl;
    AT_DISPATCH_FLOATING_TYPES(self.type(), "clamp", [&] {
      const scalar_t min_val = min_.to<scalar_t>();
      const scalar_t max_val = max_.to<scalar_t>();
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
          result,
          self,
          [min_val, max_val](
              int64_t size,
              scalar_t* x,
              scalar_t* y,
              int64_t stridex,
              int64_t stridey) {
            if (stridex == 1 && stridey == 1) {
              using Vec = vec256::Vec256<scalar_t>;
              vec256::map(
                  [min_val, max_val](const Vec& x) {
                    Vec max_vec = Vec(max_val);
                    Vec min_vec = Vec(min_val);
                    return at::vec256::max(
                        min_vec, at::vec256::min(max_vec, x));
                  },
                  x,
                  y,
                  size);
            } else {
              for (int64_t i = 0; i < size; i++) {
                ZEROUPPER
                x[stridex * i] =
                    std::max(min_val, std::min(max_val, y[stridey * i]));
              }
            }
          });
    });
  } else {
    AT_DISPATCH_ALL_TYPES(self.type(), "clamp", [&] {
      const scalar_t min_val = min_.to<scalar_t>();
      const scalar_t max_val = max_.to<scalar_t>();
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
          result,
          self,
          [min_val, max_val](
              int64_t size,
              scalar_t* x,
              scalar_t* y,
              int64_t stridex,
              int64_t stridey) {
            for (int64_t i = 0; i < size; i++) {
              ZEROUPPER
              x[stridex * i] =
                  std::max(min_val, std::min(max_val, y[stridey * i]));
            }
          });
    });
  }
}

static void fill_kernel(Tensor& self, Scalar& value_) {
  AT_DISPATCH_ALL_TYPES(self.type(), "fill", [&] {
    const scalar_t value = value_.to<scalar_t>();
    CPU_tensor_parallel_kernel_apply1<scalar_t>(
        self, [value](int64_t size, scalar_t* x, int64_t stridex) {
          if (stridex == 1) {
            using Vec = vec256::Vec256<scalar_t>;
            int64_t d = 0;
            Vec output_vec(value);
            for (; d < size - (size % Vec::size); d += Vec::size) {
              output_vec.storeu(x + d);
            }
            if (size - d > 0) {
              output_vec.storeu(x + d, size - d);
            }
          } else {
            for (int64_t i = 0; i < size; i++) {
              x[stridex * i] = value;
            }
          }
        });
  });
}

#define IMPLEMENT_COMPUTEBOUND_KERNEL(types, op, opfn)                      \
  static void op##_kernel(Tensor& result, const Tensor& self) {             \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                     \
      static constexpr int WIDTH = 128 / sizeof(scalar_t);                  \
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(                \
          result,                                                           \
          self,                                                             \
          [](int64_t size,                                                  \
             scalar_t* x,                                                   \
             scalar_t* y,                                                   \
             int64_t stridex,                                               \
             int64_t stridey) {                                             \
            using Vec = vec256::Vec256<scalar_t>;                           \
            if (stridex == 1 && stridey == 1) {                             \
              vec256::map([](const Vec& x) { return x.op(); }, x, y, size); \
            } else {                                                        \
              int64_t i = 0;                                                \
              if (size > WIDTH) {                                           \
                for (; i < size - size % WIDTH; i += WIDTH) {               \
                  scalar_t buffer[WIDTH];                                   \
                  for (int64_t j = 0; j < WIDTH; j++)                       \
                    buffer[j] = y[stridey * (j + i)];                       \
                  vec256::map_(                                             \
                      [](const Vec& x) { return x.op(); }, buffer, WIDTH);  \
                  for (int64_t j = 0; j < WIDTH; j++)                       \
                    x[stridex * (j + i)] = buffer[j];                       \
                }                                                           \
              }                                                             \
              for (; i < size; i++) {                                       \
                ZEROUPPER                                                   \
                x[stridex * i] = opfn(y[stridey * i]);                      \
              }                                                             \
            }                                                               \
          });                                                               \
    });                                                                     \
  }                                                                         \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

#define IMPLEMENT_KERNEL(types, op, opfn)                                   \
  static void op##_kernel(Tensor& result, const Tensor& self) {             \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                     \
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(                \
          result,                                                           \
          self,                                                             \
          [](int64_t size,                                                  \
             scalar_t* x,                                                   \
             scalar_t* y,                                                   \
             int64_t stridex,                                               \
             int64_t stridey) {                                             \
            using Vec = vec256::Vec256<scalar_t>;                           \
            if (stridex == 1 && stridey == 1) {                             \
              vec256::map([](const Vec& x) { return x.op(); }, x, y, size); \
            } else {                                                        \
              for (int64_t i = 0; i < size; i++) {                          \
                ZEROUPPER                                                   \
                x[stridex * i] = opfn(y[stridey * i]);                      \
              }                                                             \
            }                                                               \
          });                                                               \
    });                                                                     \
  }                                                                         \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

#define IMPLEMENT_KERNEL_LOOP(types, op, opfn)                   \
  static void op##_kernel(Tensor& result, const Tensor& self) { \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {          \
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(    \
          result,                                               \
          self,                                                 \
          [](int64_t size,                                      \
             scalar_t* x,                                       \
             scalar_t* y,                                       \
             int64_t stridex,                                   \
             int64_t stridey) {                                 \
            for (int64_t i = 0; i < size; i++) {                \
              ZEROUPPER                                         \
              x[stridex * i] = opfn(y[stridey * i]);            \
            }                                                   \
          });                                                   \
    });                                                         \
  }                                                             \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

} // anonymous namespace

// REGISTER_DISPATCH(absImpl, &abs_kernel);
REGISTER_DISPATCH(clampImpl, &clamp_kernel);
REGISTER_DISPATCH(clampMaxImpl, &clamp_max_kernel);
REGISTER_DISPATCH(clampMinImpl, &clamp_min_kernel);
REGISTER_DISPATCH(fillImpl, &fill_kernel);

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

}} // namespace at::native
