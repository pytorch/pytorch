// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>

#include <ATen/CPUApplyUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

namespace at {
namespace native {

// NOTE: These are helper functions that reduce redundant code in implementing the most typical kind of unary operators.
// YOU ARE NOT OBLIGED TO USE THESE HELPERS---if you're writing something more specialized, please don't try to make
// them work for your case, but just write something new instead. Here we use helper functions instead of a flat fat
// macro that implements everything, because the former allows some simple preprocessing that are unique to some
// operators (more is forseeable) and is more flexible and elegant than the latter.
template <typename Stub>
static inline Tensor& unary_op_impl_out(Tensor& result, const Tensor& self, Stub& stub) {
  auto iter = TensorIterator::unary_op(result, self,
    /*check_mem_overlap=*/true);
  stub(iter.device_type(), iter);
  return result;
}

// out_impl passed into unary_op_impl and unary_op_impl_  must go through at:: device dispatch
// otherwise it won't dispatch to out-of-source devices like XLA.
// For example it must be at::bitwise_not_out instead of bitwise_not_out(which is at::native!).
template <typename OutImpl>
static inline Tensor unary_op_impl(const Tensor& self, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  return out_impl(self, self);
}

Tensor& bitwise_not_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, bitwise_not_stub); }
Tensor bitwise_not(const Tensor& self) { return unary_op_impl(self, at::bitwise_not_out); }
Tensor& bitwise_not_(Tensor& self) { return unary_op_impl_(self, at::bitwise_not_out); }

Tensor& ceil_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, ceil_stub); }
Tensor ceil(const Tensor& self) { return unary_op_impl(self, at::ceil_out); }
Tensor& ceil_(Tensor& self) { return unary_op_impl_(self, at::ceil_out); }

Tensor& expm1_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, expm1_stub); }
Tensor expm1(const Tensor& self) { return unary_op_impl(self, at::expm1_out); }
Tensor& expm1_(Tensor& self) { return unary_op_impl_(self, at::expm1_out); }

Tensor& floor_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, floor_stub); }
Tensor floor(const Tensor& self) { return unary_op_impl(self, at::floor_out); }
Tensor& floor_(Tensor& self) { return unary_op_impl_(self, at::floor_out); }

Tensor& log_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, log_stub); }
Tensor log(const Tensor& self) { return unary_op_impl(self, at::log_out); }
Tensor& log_(Tensor& self) { return unary_op_impl_(self, at::log_out); }

Tensor& round_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, round_stub); }
Tensor round(const Tensor& self) { return unary_op_impl(self, at::round_out); }
Tensor& round_(Tensor& self) { return unary_op_impl_(self, at::round_out); }

Tensor& digamma_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, digamma_stub); }
Tensor digamma(const Tensor& self) { return unary_op_impl(self, digamma_out); }
Tensor& digamma_(Tensor& self) { return unary_op_impl_(self, digamma_out); }

Tensor& rsqrt_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, rsqrt_stub); }
Tensor rsqrt(const Tensor& self) { return unary_op_impl(self, at::rsqrt_out); }
Tensor& rsqrt_(Tensor& self) { return unary_op_impl_(self, at::rsqrt_out); }

Tensor& sign_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, sign_stub); }
Tensor sign(const Tensor& self) { return unary_op_impl(self, at::sign_out); }
Tensor& sign_(Tensor& self) { return unary_op_impl_(self, at::sign_out); }

Tensor& trunc_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, trunc_stub); }
Tensor trunc(const Tensor& self) { return unary_op_impl(self, at::trunc_out); }
Tensor& trunc_(Tensor& self) { return unary_op_impl_(self, at::trunc_out); }

Tensor& neg_out(Tensor& result, const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  return unary_op_impl_out(result, self, neg_stub);
}
Tensor neg(const Tensor& self) { return unary_op_impl(self, at::neg_out); }
Tensor& neg_(Tensor& self) { return unary_op_impl_(self, at::neg_out); }

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return at::logical_not_out(self, self);
}

Tensor& logical_not_out(Tensor& result, const Tensor& self) {
  TensorIterator iter;
  iter.dont_compute_common_dtype();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(self);
  iter.build();
  logical_not_stub(iter.device_type(), iter);
  return result;
}

Tensor clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  Tensor result = at::empty({0}, self.options());
  return clamp_out(result, self, min, max);
}

Tensor clamp_max(const Tensor& self, Scalar max) {
  Tensor result = at::empty({0}, self.options());
  return clamp_max_out(result, self, max);
}

Tensor clamp_min(const Tensor& self, Scalar min) {
  Tensor result = at::empty({0}, self.options());
  return clamp_min_out(result, self, min);
}

Tensor& _clamp__cpu(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return clamp_out(self, self, min, max);
}

Tensor polygamma(int64_t n, const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  at::polygamma_out(result, n, self);
  return result;
}
Tensor& polygamma_(Tensor& self, int64_t n) {
  return at::polygamma_out(self, n, self);
}
Tensor& polygamma_out(Tensor& result, int64_t n, const Tensor& self) {
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  auto iter = TensorIterator::unary_op(result, self,
    /*check_mem_overlap=*/true);
  polygamma_stub(iter.device_type(), iter, n);
  return result;
}

Tensor& _clamp_out_cpu(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  if (min && max) {
    checkBackend("clamp", result, Backend::CPU);
    auto iter = TensorIterator::unary_op(result, self,
        /*check_mem_overlap=*/true);
    clamp_stub(iter.device_type(), iter, *min, *max);
  } else if (max) {
    clamp_max_out(result, self, *max);
  } else if (min) {
    clamp_min_out(result, self, *min);
  } else {
    AT_ERROR("At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& _clamp_max__cpu(Tensor& self, Scalar max) {
  return clamp_max_out(self, self, max);
}

Tensor& _clamp_max_out_cpu(Tensor& result, const Tensor& self, Scalar max) {
  checkBackend("clamp_max", result, Backend::CPU);
  auto iter = TensorIterator::unary_op(result, self,
      /*check_mem_overlap=*/true);
  clamp_max_stub(iter.device_type(), iter, max);
  return result;
}

Tensor& _clamp_min__cpu(Tensor& self, Scalar min) {
  return clamp_min_out(self, self, min);
}

Tensor& _clamp_min_out_cpu(Tensor& result, const Tensor& self, Scalar min) {
  checkBackend("clamp_min", result, Backend::CPU);
  auto iter = TensorIterator::unary_op(result, self,
      /*check_mem_overlap=*/true);
  clamp_min_stub(iter.device_type(), iter, min);
  return result;
}

Tensor mvlgamma(const Tensor& self, int64_t p) {
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
           "mvlgamma is not implemented for ", self.type());
  TORCH_CHECK((self > 0.5 * (p - 1.)).all().item<uint8_t>(),
           "Condition for computing multivariate log-gamma not met");
  TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
  Tensor args = native::arange(-p / 2. + 0.5, 0.5, 0.5, self.options());
  args = args.add(self.unsqueeze(-1));
  return args.lgamma_().sum(-1).add_(p * (p - 1) * std::log(M_PI) / 4.);
}

Tensor& mvlgamma_(Tensor& self, int64_t p) {
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
           "mvlgamma is not implemented for ", self.type());
  TORCH_CHECK((self > 0.5 * (p - 1.)).all().item<uint8_t>(),
           "Condition for computing multivariate log-gamma not met");
  TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
  Tensor args = native::arange(-p / 2. + 0.5, 0.5, 0.5, self.options());
  args = args.add(self.unsqueeze(-1));
  return self.copy_(args.lgamma_().sum(-1).add_(p * (p - 1) * std::log(M_PI) / 4.));
}

inline void propagate_names_if_namedtensor_enabled(Tensor& result, const Tensor& src) {
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, src);
#endif
}

// NB: If you use this macro, you may also need to add a CUDA forwarding
// stub in CUDAUnaryOps

#define IMPLEMENT_UNARY_OP_CORE(op)                                    \
  Tensor op(const Tensor& self) {                                      \
    Tensor result = at::empty({0}, self.options());                    \
    at::op##_out(result, self);                                        \
    return result;                                                     \
  }

#define IMPLEMENT_UNARY_OP_OUT_INPLACE(op, prefix, device)             \
  Tensor& _##op##__##prefix(Tensor& self) {                            \
    return at::op##_out(self, self);                                   \
  }                                                                    \
  Tensor& _##op##_out_##prefix(Tensor& result, const Tensor& self) {   \
    checkBackend(#op, result, Backend::device);                        \
    auto iter = TensorIterator::unary_op(result, self,                 \
      /*check_mem_overlap=*/true);                                     \
    op##_stub(iter.device_type(), iter);                               \
    return result;                                                     \
  }

#define IMPLEMENT_UNARY_OP_VEC(op)                                     \
  IMPLEMENT_UNARY_OP_CORE(op)                                          \
  IMPLEMENT_UNARY_OP_OUT_INPLACE(op, cpu, CPU)

#define IMPLEMENT_UNARY_OP_VEC_CUDA(op)                                \
  IMPLEMENT_UNARY_OP_CORE(op)                                          \
  IMPLEMENT_UNARY_OP_OUT_INPLACE(op, cpu, CPU)                         \
  IMPLEMENT_UNARY_OP_OUT_INPLACE(op, cuda, CUDA)

IMPLEMENT_UNARY_OP_VEC(abs)
IMPLEMENT_UNARY_OP_VEC(acos)
IMPLEMENT_UNARY_OP_VEC(asin)
IMPLEMENT_UNARY_OP_VEC(atan)
IMPLEMENT_UNARY_OP_VEC(cos)
IMPLEMENT_UNARY_OP_VEC(cosh)
IMPLEMENT_UNARY_OP_VEC(erf)
IMPLEMENT_UNARY_OP_VEC(erfc)
IMPLEMENT_UNARY_OP_VEC_CUDA(erfinv)
IMPLEMENT_UNARY_OP_VEC(exp)
IMPLEMENT_UNARY_OP_VEC(frac)
IMPLEMENT_UNARY_OP_VEC(log10)
IMPLEMENT_UNARY_OP_VEC(log1p)
IMPLEMENT_UNARY_OP_VEC(log2)
IMPLEMENT_UNARY_OP_VEC(reciprocal)
IMPLEMENT_UNARY_OP_VEC(sigmoid)
IMPLEMENT_UNARY_OP_VEC(sin)
IMPLEMENT_UNARY_OP_VEC(sinh)
IMPLEMENT_UNARY_OP_VEC(sqrt)
IMPLEMENT_UNARY_OP_VEC(tan)
IMPLEMENT_UNARY_OP_VEC(tanh)
IMPLEMENT_UNARY_OP_VEC_CUDA(lgamma)

DEFINE_DISPATCH(abs_stub);
DEFINE_DISPATCH(acos_stub);
DEFINE_DISPATCH(asin_stub);
DEFINE_DISPATCH(atan_stub);
DEFINE_DISPATCH(bitwise_not_stub);
DEFINE_DISPATCH(ceil_stub);
DEFINE_DISPATCH(clamp_stub);
DEFINE_DISPATCH(clamp_max_stub);
DEFINE_DISPATCH(clamp_min_stub);
DEFINE_DISPATCH(cos_stub);
DEFINE_DISPATCH(cosh_stub);
DEFINE_DISPATCH(digamma_stub);
DEFINE_DISPATCH(erf_stub);
DEFINE_DISPATCH(erfc_stub);
DEFINE_DISPATCH(erfinv_stub);
DEFINE_DISPATCH(exp_stub);
DEFINE_DISPATCH(expm1_stub);
DEFINE_DISPATCH(floor_stub);
DEFINE_DISPATCH(frac_stub);
DEFINE_DISPATCH(log_stub);
DEFINE_DISPATCH(log10_stub);
DEFINE_DISPATCH(log1p_stub);
DEFINE_DISPATCH(log2_stub);
DEFINE_DISPATCH(logical_not_stub);
DEFINE_DISPATCH(neg_stub);
DEFINE_DISPATCH(polygamma_stub);
DEFINE_DISPATCH(reciprocal_stub);
DEFINE_DISPATCH(round_stub);
DEFINE_DISPATCH(rsqrt_stub);
DEFINE_DISPATCH(sigmoid_stub);
DEFINE_DISPATCH(sign_stub);
DEFINE_DISPATCH(sin_stub);
DEFINE_DISPATCH(sinh_stub);
DEFINE_DISPATCH(sqrt_stub);
DEFINE_DISPATCH(tan_stub);
DEFINE_DISPATCH(tanh_stub);
DEFINE_DISPATCH(trunc_stub);
DEFINE_DISPATCH(lgamma_stub);
}
} // namespace at
