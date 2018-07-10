#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include "ATen/Dispatch.h"
#include "ATen/cpu/vml.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/native/cpu/CapabilityDispatch.h"
#ifdef __AVX2__
#include "ATen/native/cpu/avx_mathfun.h"
#endif

namespace at { namespace native {
namespace {

template <typename scalar_t, typename VecOp, typename ScalarOp>
inline void _parallel_vector_map_(
    VecOp vec_op,
    ScalarOp scalar_op,
    Tensor& self) {
  CPU_tensor_parallel_kernel_apply1<scalar_t>(
      self, [vec_op, scalar_op](int64_t size, scalar_t* x, int64_t stridex) {
        vec256::map_(vec_op, x, size, stridex);
      });
}

template <typename scalar_t, typename VecOp, typename ScalarOp>
inline void _parallel_vector_map(
    VecOp vec_op,
    ScalarOp scalar_op,
    Tensor& result,
    const Tensor& self) {
  CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
      result,
      self,
      [vec_op, scalar_op](
          int64_t size,
          scalar_t* x,
          scalar_t* y,
          int64_t stridex,
          int64_t stridey) {
          vec256::map(vec_op, x, y, size, stridex, stridey);
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

using namespace vec256;

template <typename scalar_t>
static int64_t _sigmoid(scalar_t* x, scalar_t* y, int64_t size);

// This should be a temporary solution until we understand why SLEEF is slower
// for sigmoid

template <>
int64_t _sigmoid(float* x, float* y, int64_t size) {
  using Vec = Vec256<float>;
  int64_t i = 0;
  for (; i < size - (size % (2 * Vec::size)); i += 2 * Vec::size) {
    Vec ret = Vec::loadu(y + i);
    Vec ret2 = Vec::loadu(y + i + Vec::size);
    ret = ret.neg();
    ret2 = ret2.neg();
#if defined(__AVX2__) && !defined(_MSC_VER)
    ret = exp256_ps(ret);
    ret2 = exp256_ps(ret2);
#else
    ret = ret.exp();
    ret2 = ret2.exp();
#endif
    ret = Vec((float)(1)) + ret;
    ret2 = Vec((float)(1)) + ret2;
    ret = ret.reciprocal();
    ret2 = ret2.reciprocal();
    ret.store(x + i);
    ret2.store(x + i + Vec::size);
  }
  return i;
}

template <>
int64_t _sigmoid(double* x, double* y, int64_t size) {
  using Vec = Vec256<double>;
  int64_t i = 0;
  for (; i < size - (size % (2 * Vec::size)); i += 2 * Vec::size) {
    Vec ret = Vec::loadu(y + i);
    Vec ret2 = Vec::loadu(y + i + Vec::size);
    ret = ret.neg();
    ret2 = ret2.neg();
    ret = ret.exp();
    ret2 = ret2.exp();
    ret = Vec((double)(1)) + ret;
    ret2 = Vec((double)(1)) + ret2;
    ret = ret.reciprocal();
    ret2 = ret2.reciprocal();
    ret.store(x + i);
    ret2.store(x + i + Vec::size);
  }
  return i;
}

static void sigmoid_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sigmoid", [&] {
    using Vec = Vec256<scalar_t>;
    CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
        result,
        self,
        [](int64_t size,
           scalar_t* x,
           scalar_t* y,
           int64_t stridex,
           int64_t stridey) {
          int64_t i = 0;
          if (stridex == 1 && stridey == 1) {
            i = _sigmoid(x, y, size);
          }
          for (; i < size; i += Vec::size) {
            scalar_t buffer[Vec::size];
            int64_t width = Vec::size;
            width = std::min(width, size - i);
            for (int64_t j = 0; j < width; j++) {
              buffer[j] = y[stridey * (i + j)];
            }
            Vec ret = Vec::loadu(buffer);
            ret = Vec((scalar_t)(0)) - ret;
            ret = ret.exp();
            ret = Vec((scalar_t)(1)) + ret;
            ret = ret.reciprocal();
            ret.store(buffer);
            for (int64_t j = 0; j < width; j++)
              x[stridex * (i + j)] = buffer[j];
          }
        });
  });
}

#define IMPLEMENT_FLOAT_KERNEL(dispatchtypes, op)                          \
  static void op##_kernel(Tensor& result, const Tensor& self) {            \
    AT_DISPATCH_##dispatchtypes##_TYPES(self.type(), #op, [&] {            \
      if (self.is_contiguous() && result.is_contiguous()) {                \
        vml::v##op(                                                        \
            result.data<scalar_t>(), self.data<scalar_t>(), self.numel()); \
                                                                           \
      } else {                                                             \
        static constexpr int64_t WIDTH = 131072 / sizeof(scalar_t);        \
        CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(             \
            result,                                                        \
            self,                                                          \
            [](int64_t size,                                               \
               scalar_t* x,                                                \
               scalar_t* y,                                                \
               int64_t stridex,                                            \
               int64_t stridey) {                                          \
              if (stridex == 1 && stridey == 1) {                          \
                vml::v##op(x, y, size);                                    \
              } else {                                                     \
                for (int64_t i = 0; i < size; i += WIDTH) {                \
                  scalar_t buffer[WIDTH];                                  \
                  int64_t width = WIDTH;                                   \
                  width = std::min(width, size - i);                       \
                  for (int64_t j = 0; j < width; j++)                      \
                    buffer[j] = y[stridey * (i + j)];                      \
                  vml::v##op(buffer, buffer, width);                       \
                  for (int64_t j = 0; j < width; j++)                      \
                    x[stridex * (i + j)] = buffer[j];                      \
                }                                                          \
              }                                                            \
            });                                                            \
      }                                                                    \
    });                                                                    \
  }                                                                        \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

} // anonymous namespace

REGISTER_DISPATCH(clamp_Impl, &clamp__kernel);
REGISTER_DISPATCH(clampMax_Impl, &clamp_max__kernel);
REGISTER_DISPATCH(clampMin_Impl, &clamp_min__kernel);
REGISTER_DISPATCH(clampImpl, &clamp_kernel);
REGISTER_DISPATCH(clampMaxImpl, &clamp_max_kernel);
REGISTER_DISPATCH(clampMinImpl, &clamp_min_kernel);
REGISTER_DISPATCH(sigmoidImpl, &sigmoid_kernel)

// IMPLEMENT_FLOAT_KERNEL(ALL, abs)
IMPLEMENT_FLOAT_KERNEL(FLOATING, acos)
IMPLEMENT_FLOAT_KERNEL(FLOATING, asin)
IMPLEMENT_FLOAT_KERNEL(FLOATING, atan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, ceil)
IMPLEMENT_FLOAT_KERNEL(FLOATING, cos)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, cosh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, erf)
IMPLEMENT_FLOAT_KERNEL(FLOATING, erfc)
IMPLEMENT_FLOAT_KERNEL(FLOATING, exp)
IMPLEMENT_FLOAT_KERNEL(FLOATING, expm1)
IMPLEMENT_FLOAT_KERNEL(FLOATING, floor)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log10)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log1p)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log2)
IMPLEMENT_FLOAT_KERNEL(FLOATING, round)
IMPLEMENT_FLOAT_KERNEL(FLOATING, rsqrt)
IMPLEMENT_FLOAT_KERNEL(FLOATING, sin)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, sinh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, sqrt)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tanh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, trunc)

}} // namespace at::native
