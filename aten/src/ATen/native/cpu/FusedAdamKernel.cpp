#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdam.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000
#include <riscv_vector.h>
#endif
namespace at::native {

namespace{

template <typename scalar_t, typename opmath_t, ADAM_MODE adam_mode>
std::enable_if_t<
    std::is_same_v<scalar_t, Half> || std::is_same_v<scalar_t, BFloat16>,
    void>
    inline adam_math(
  scalar_t* param_ptr,
  scalar_t* exp_avg_ptr,
  scalar_t* exp_avg_sq_ptr,
  scalar_t* grad_ptr,
  scalar_t* max_exp_avg_sq_ptr,
  double lr,
  double bias_correction1,
  double bias_correction2,
  double exp_avg_grad_coefficient,
  double exp_avg_sq_grad_coefficient,
  double bias_correction2_sqrt,
  double eps,
  double weight_decay,
  double beta2,
  bool amsgrad,
  bool maximize,
  const float* grad_scale_ptr,
  int64_t size
){
  double step_size = lr / bias_correction1;
  using lpVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<opmath_t>;
  int64_t d = 0;
  for (; d < size - (size % lpVec::size()); d += lpVec::size()) {
    lpVec param_lpvec = lpVec::loadu(param_ptr + d);
    auto [param_vec1, param_vec2] = vec::convert_to_float<scalar_t>(param_lpvec);
    lpVec grad_lpvec = lpVec::loadu(grad_ptr + d);
    auto [grad_vec1, grad_vec2] = vec::convert_to_float<scalar_t>(grad_lpvec);
    if (grad_scale_ptr) {
      grad_vec1 = grad_vec1 / fVec(float(*grad_scale_ptr));
      grad_vec2 = grad_vec2 / fVec(float(*grad_scale_ptr));
      lpVec grad_vec_to_store = vec::convert_from_float<scalar_t>(grad_vec1, grad_vec2);
      grad_vec_to_store.store(grad_ptr + d);
    }
    if (maximize){
      grad_vec1 = grad_vec1 * fVec(opmath_t(-1.0));
      grad_vec2 = grad_vec2 * fVec(opmath_t(-1.0));
    }
    if (weight_decay != 0.f){
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_vec1 += param_vec1 * fVec(opmath_t(weight_decay));
        grad_vec2 += param_vec2 * fVec(opmath_t(weight_decay));
       } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_vec1 = param_vec1 * fVec(opmath_t(1 - lr * weight_decay));
        param_vec2 = param_vec2 * fVec(opmath_t(1 - lr * weight_decay));
      }
    }

    lpVec exp_avg_lpvec = lpVec::loadu(exp_avg_ptr + d);
    auto [exp_avg_vec1, exp_avg_vec2] = vec::convert_to_float<scalar_t>(exp_avg_lpvec);

    // exp_avg.lerp_(grad, 1 - beta1)
    const fVec lerp_weight = fVec(opmath_t(exp_avg_grad_coefficient));
    auto mask = lerp_weight.abs() < fVec(0.5);
    auto coeff = fVec::blendv(lerp_weight - fVec(1), lerp_weight, mask);

    auto base1 = fVec::blendv(grad_vec1, exp_avg_vec1, mask);
    exp_avg_vec1 = vec::fmadd(coeff, grad_vec1 - exp_avg_vec1, base1);

    auto base2 = fVec::blendv(grad_vec2, exp_avg_vec2, mask);
    exp_avg_vec2 = vec::fmadd(coeff, grad_vec2 - exp_avg_vec2, base2);

    lpVec exp_avg_sq_lpvec = lpVec::loadu(exp_avg_sq_ptr + d);
    auto [exp_avg_sq_vec1, exp_avg_sq_vec2] = vec::convert_to_float<scalar_t>(exp_avg_sq_lpvec);
    exp_avg_sq_vec1 = exp_avg_sq_vec1 * fVec(opmath_t(beta2)) +
        fVec(opmath_t(exp_avg_sq_grad_coefficient)) * grad_vec1 * grad_vec1;
    exp_avg_sq_vec2 = exp_avg_sq_vec2 * fVec(opmath_t(beta2)) +
        fVec(opmath_t(exp_avg_sq_grad_coefficient)) * grad_vec2 * grad_vec2;

    vec::convert_from_float<scalar_t>(exp_avg_vec1, exp_avg_vec2).store(exp_avg_ptr + d);
    vec::convert_from_float<scalar_t>(exp_avg_sq_vec1, exp_avg_sq_vec2).store(exp_avg_sq_ptr + d);

    fVec denom_vec1, denom_vec2;
    if (amsgrad) {
      lpVec max_exp_avg_sq_lpvec = lpVec::loadu(max_exp_avg_sq_ptr + d);
      auto [max_exp_avg_sq_vec1, max_exp_avg_sq_vec2] = vec::convert_to_float<scalar_t>(max_exp_avg_sq_lpvec);
      max_exp_avg_sq_vec1 = maximum(max_exp_avg_sq_vec1, exp_avg_sq_vec1);
      max_exp_avg_sq_vec2 = maximum(max_exp_avg_sq_vec2, exp_avg_sq_vec2);
      vec::convert_from_float<scalar_t>(max_exp_avg_sq_vec1, max_exp_avg_sq_vec2).store(max_exp_avg_sq_ptr + d);
      denom_vec1 =
          (max_exp_avg_sq_vec1.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
      denom_vec2 =
          (max_exp_avg_sq_vec2.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
    } else {
      denom_vec1 =
          (exp_avg_sq_vec1.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
      denom_vec2 =
          (exp_avg_sq_vec2.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
    }
    param_vec1 = param_vec1 + fVec(opmath_t(-step_size)) * exp_avg_vec1 / denom_vec1;
    param_vec2 = param_vec2 + fVec(opmath_t(-step_size)) * exp_avg_vec2 / denom_vec2;
    vec::convert_from_float<scalar_t>(param_vec1, param_vec2).store(param_ptr + d);
  }
  for (; d < size; d++) {
    opmath_t grad_val = grad_ptr[d];
    opmath_t param_val = param_ptr[d];
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / float(*grad_scale_ptr);
      grad_ptr[d] = grad_val;
    }
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.f){
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_val += param_val * opmath_t(weight_decay);
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_val = param_val * opmath_t(1 - lr * weight_decay);
      }
    }
    // exp_avg.lerp_(grad, 1 - beta1)
    opmath_t exp_avg_var = exp_avg_ptr[d];
    auto is_lerp_weight_small = std::abs(opmath_t(exp_avg_grad_coefficient)) < opmath_t(0.5);
    if (is_lerp_weight_small) {
      exp_avg_var = exp_avg_var + opmath_t(exp_avg_grad_coefficient) * (grad_val - exp_avg_var);
    } else {
      exp_avg_var = grad_val - (grad_val - exp_avg_var) * (opmath_t(1) - opmath_t(exp_avg_grad_coefficient));
    }
    exp_avg_ptr[d] = scalar_t(exp_avg_var);
    opmath_t exp_avg_sq_var = exp_avg_sq_ptr[d];
    exp_avg_sq_var = exp_avg_sq_var * opmath_t(beta2);
    exp_avg_sq_var = exp_avg_sq_var +
        opmath_t(exp_avg_sq_grad_coefficient) * grad_val * grad_val;
    exp_avg_sq_ptr[d] = scalar_t(exp_avg_sq_var);
    opmath_t demon_val;
    if (amsgrad) {
      opmath_t max_exp_avg_sq_var = max_exp_avg_sq_ptr[d];
      max_exp_avg_sq_var = std::max(max_exp_avg_sq_var, exp_avg_sq_var);
      max_exp_avg_sq_ptr[d] =
          scalar_t(max_exp_avg_sq_var);
      demon_val =
          std::sqrt(max_exp_avg_sq_var) / opmath_t(bias_correction2_sqrt) + opmath_t(eps);
    } else {
      demon_val = std::sqrt(exp_avg_sq_var) / opmath_t(bias_correction2_sqrt) + opmath_t(eps);
    }
    param_ptr[d] = param_val - opmath_t(step_size) * exp_avg_var / demon_val;
  }
}


#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000

// Maps scalar_t to an LMUL=2 RVV vector type plus the small intrinsic set the
// kernel below needs, so a single kernel body serves float and double.
template <typename scalar_t>
struct rvv_traits;

template <>
struct rvv_traits<float> {
  using v_t = vfloat32m2_t;
  static size_t setvl(size_t n) { return __riscv_vsetvl_e32m2(n); }
  static size_t setvlmax() { return __riscv_vsetvlmax_e32m2(); }
  static v_t undef() { return __riscv_vundefined_f32m2(); }
  static v_t load(const float* p, size_t vl) { return __riscv_vle32_v_f32m2(p, vl); }
  static void store(float* p, v_t v, size_t vl) { __riscv_vse32_v_f32m2(p, v, vl); }
  static v_t fmul(v_t a, float b, size_t vl) { return __riscv_vfmul_vf_f32m2(a, b, vl); }
  static v_t fmul(v_t a, v_t b, size_t vl) { return __riscv_vfmul_vv_f32m2(a, b, vl); }
  static v_t fdiv(v_t a, float b, size_t vl) { return __riscv_vfdiv_vf_f32m2(a, b, vl); }
  static v_t fdiv(v_t a, v_t b, size_t vl) { return __riscv_vfdiv_vv_f32m2(a, b, vl); }
  static v_t fadd(v_t a, float b, size_t vl) { return __riscv_vfadd_vf_f32m2(a, b, vl); }
  static v_t fadd(v_t a, v_t b, size_t vl) { return __riscv_vfadd_vv_f32m2(a, b, vl); }
  static v_t fsub(v_t a, v_t b, size_t vl) { return __riscv_vfsub_vv_f32m2(a, b, vl); }
  static v_t fsqrt(v_t a, size_t vl) { return __riscv_vfsqrt_v_f32m2(a, vl); }
  static v_t fmax(v_t a, v_t b, size_t vl) { return __riscv_vfmax_vv_f32m2(a, b, vl); }
};

template <>
struct rvv_traits<double> {
  using v_t = vfloat64m2_t;
  static size_t setvl(size_t n) { return __riscv_vsetvl_e64m2(n); }
  static size_t setvlmax() { return __riscv_vsetvlmax_e64m2(); }
  static v_t undef() { return __riscv_vundefined_f64m2(); }
  static v_t load(const double* p, size_t vl) { return __riscv_vle64_v_f64m2(p, vl); }
  static void store(double* p, v_t v, size_t vl) { __riscv_vse64_v_f64m2(p, v, vl); }
  static v_t fmul(v_t a, double b, size_t vl) { return __riscv_vfmul_vf_f64m2(a, b, vl); }
  static v_t fmul(v_t a, v_t b, size_t vl) { return __riscv_vfmul_vv_f64m2(a, b, vl); }
  static v_t fdiv(v_t a, double b, size_t vl) { return __riscv_vfdiv_vf_f64m2(a, b, vl); }
  static v_t fdiv(v_t a, v_t b, size_t vl) { return __riscv_vfdiv_vv_f64m2(a, b, vl); }
  static v_t fadd(v_t a, double b, size_t vl) { return __riscv_vfadd_vf_f64m2(a, b, vl); }
  static v_t fadd(v_t a, v_t b, size_t vl) { return __riscv_vfadd_vv_f64m2(a, b, vl); }
  static v_t fsub(v_t a, v_t b, size_t vl) { return __riscv_vfsub_vv_f64m2(a, b, vl); }
  static v_t fsqrt(v_t a, size_t vl) { return __riscv_vfsqrt_v_f64m2(a, vl); }
  static v_t fmax(v_t a, v_t b, size_t vl) { return __riscv_vfmax_vv_f64m2(a, b, vl); }
};

// RVV kernel for the float/double adam_math overload. VLA style, runs on any
// VLEN >= 128. Computation order matches the generic Vectorized path below
// operation for operation (IEEE-exact vfsqrt/vfdiv, no reciprocal
// approximation), so results are bit-identical to it.
template <typename scalar_t, ADAM_MODE adam_mode>
inline void adam_math_rvv(
    scalar_t* param_ptr,
    scalar_t* exp_avg_ptr,
    scalar_t* exp_avg_sq_ptr,
    scalar_t* grad_ptr,
    scalar_t* max_exp_avg_sq_ptr,
    double lr,
    double step_size,
    double exp_avg_grad_coefficient,
    double exp_avg_sq_grad_coefficient,
    double bias_correction2_sqrt,
    double eps,
    double weight_decay,
    double beta2,
    bool amsgrad,
    bool maximize,
    const float* grad_scale_ptr,
    int64_t size) {
  using traits = rvv_traits<scalar_t>;
  using v_t = typename traits::v_t;

  // The lerp weight is lane-uniform, so the generic path's per-lane blendv
  // collapses to one scalar branch hoisted out of the loop. Comparison and
  // arithmetic stay on the narrowed value, exactly like the generic path.
  const scalar_t lerp_weight_s = static_cast<scalar_t>(exp_avg_grad_coefficient);
  const bool lerp_small = std::abs(lerp_weight_s) < scalar_t(0.5);
  const scalar_t lerp_coeff = lerp_small ? lerp_weight_s : lerp_weight_s - scalar_t(1);
  const scalar_t beta2_s = static_cast<scalar_t>(beta2);
  const scalar_t sq_coeff = static_cast<scalar_t>(exp_avg_sq_grad_coefficient);
  const scalar_t bc2_sqrt = static_cast<scalar_t>(bias_correction2_sqrt);
  const scalar_t eps_s = static_cast<scalar_t>(eps);
  const scalar_t neg_step_size = static_cast<scalar_t>(-step_size);

  // max_v arrives preloaded: the amsgrad maximum's load sits at the head of
  // the load -> fmax -> sqrt dependency chain, so it is issued one tile ahead
  // by the caller's prefetch group instead of inside the chain here.
  auto compute_tile = [&](v_t param_v, v_t grad_v, v_t exp_avg_v,
                          v_t exp_avg_sq_v, v_t max_v, int64_t d, size_t vl) {
    if (grad_scale_ptr) {
      grad_v = traits::fdiv(grad_v, *grad_scale_ptr, vl);
      traits::store(grad_ptr + d, grad_v, vl);
    }
    if (maximize) {
      grad_v = traits::fmul(grad_v, scalar_t(-1.0), vl);
    }
    if (weight_decay != 0.f) {
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        // grad += param * wd, split into mul then add like the generic path
        grad_v = traits::fadd(
            grad_v, traits::fmul(param_v, static_cast<scalar_t>(weight_decay), vl), vl);
      } else {
        param_v = traits::fmul(param_v, static_cast<scalar_t>(1 - lr * weight_decay), vl);
      }
    }

    // exp_avg.lerp_(grad, 1 - beta1); every op separate (no FMA fusion) and
    // ordered exactly like the generic Vectorized path, so results stay
    // bit-identical to it
    const v_t diff_v = traits::fsub(grad_v, exp_avg_v, vl);
    const v_t base_v = lerp_small ? exp_avg_v : grad_v;
    exp_avg_v = traits::fadd(traits::fmul(diff_v, lerp_coeff, vl), base_v, vl);

    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad, again
    // op-for-op the generic path's expression tree (left-associative)
    exp_avg_sq_v = traits::fadd(
        traits::fmul(exp_avg_sq_v, beta2_s, vl),
        traits::fmul(traits::fmul(grad_v, sq_coeff, vl), grad_v, vl),
        vl);
    traits::store(exp_avg_ptr + d, exp_avg_v, vl);
    traits::store(exp_avg_sq_ptr + d, exp_avg_sq_v, vl);

    v_t root_v;
    if (amsgrad) {
      const v_t max_out_v = traits::fmax(max_v, exp_avg_sq_v, vl);
      traits::store(max_exp_avg_sq_ptr + d, max_out_v, vl);
      root_v = traits::fsqrt(max_out_v, vl);
    } else {
      root_v = traits::fsqrt(exp_avg_sq_v, vl);
    }
    const v_t denom_v = traits::fadd(
        traits::fdiv(root_v, bc2_sqrt, vl), eps_s, vl);
    // param += (-step_size) * exp_avg / denom, multiply before divide like
    // the generic path; the final div/add chain overlaps the next tile's loads
    param_v = traits::fadd(
        param_v,
        traits::fdiv(traits::fmul(exp_avg_v, neg_step_size, vl), denom_v, vl),
        vl);
    traits::store(param_ptr + d, param_v, vl);
  };

  const size_t vlmax = traits::setvlmax();
  const int64_t nfull = size / static_cast<int64_t>(vlmax);
  int64_t d = 0;

  // Software-pipelined main loop, two tiles deep on the four
  // read-modify-write streams and one tile deep on the amsgrad stream:
  // 4*3 + 2 live m2 vectors = 28 of the 32 physical vector registers at
  // LMUL=2, leaving only transients, so a spill or two in the steady state is
  // expected and accepted (bandwidth-bound machines hide it). Each tile's
  // loads issue two tiles ahead of its math, doubling the load bytes in
  // flight; the amsgrad maximum rides one tile ahead of its fmax consumer
  // instead of stalling the chain head on a same-tile load. Consecutive
  // tiles carry independent sqrt/div chains, and each LMUL=2 vector holds
  // 8 fp32 / 4 fp64 elements, so at VLEN=128 a single instruction already
  // spans both 128-bit pipes.
  if (nfull > 0) {
    const int64_t step = static_cast<int64_t>(vlmax);
    v_t param_a = traits::load(param_ptr, vlmax);
    v_t grad_a = traits::load(grad_ptr, vlmax);
    v_t exp_avg_a = traits::load(exp_avg_ptr, vlmax);
    v_t exp_avg_sq_a = traits::load(exp_avg_sq_ptr, vlmax);
    v_t max_a = traits::undef();
    if (amsgrad) {
      max_a = traits::load(max_exp_avg_sq_ptr, vlmax);
    }
    if (nfull > 1) {
      v_t param_b = traits::load(param_ptr + step, vlmax);
      v_t grad_b = traits::load(grad_ptr + step, vlmax);
      v_t exp_avg_b = traits::load(exp_avg_ptr + step, vlmax);
      v_t exp_avg_sq_b = traits::load(exp_avg_sq_ptr + step, vlmax);
      for (int64_t k = 2; k < nfull; ++k) {
        const int64_t dc = (k - 2) * step;  // tile being computed
        const int64_t dk = k * step;  // tile being prefetched, two ahead
        v_t param_n = traits::load(param_ptr + dk, vlmax);
        v_t grad_n = traits::load(grad_ptr + dk, vlmax);
        v_t exp_avg_n = traits::load(exp_avg_ptr + dk, vlmax);
        v_t exp_avg_sq_n = traits::load(exp_avg_sq_ptr + dk, vlmax);
        v_t max_n = traits::undef();
        if (amsgrad) {
          max_n = traits::load(max_exp_avg_sq_ptr + dk - step, vlmax);
        }
        compute_tile(
            param_a, grad_a, exp_avg_a, exp_avg_sq_a, max_a, dc, vlmax);
        param_a = param_b;
        grad_a = grad_b;
        exp_avg_a = exp_avg_b;
        exp_avg_sq_a = exp_avg_sq_b;
        max_a = max_n;
        param_b = param_n;
        grad_b = grad_n;
        exp_avg_b = exp_avg_n;
        exp_avg_sq_b = exp_avg_sq_n;
      }
      const int64_t dl0 = (nfull - 2) * step;
      const int64_t dl1 = (nfull - 1) * step;
      v_t max_b = traits::undef();
      if (amsgrad) {
        max_b = traits::load(max_exp_avg_sq_ptr + dl1, vlmax);
      }
      compute_tile(
          param_a, grad_a, exp_avg_a, exp_avg_sq_a, max_a, dl0, vlmax);
      compute_tile(
          param_b, grad_b, exp_avg_b, exp_avg_sq_b, max_b, dl1, vlmax);
    } else {
      compute_tile(param_a, grad_a, exp_avg_a, exp_avg_sq_a, max_a, 0, vlmax);
    }
    d = nfull * step;
  }

  // Masked tail: vsetvl clamps vl to the remaining elements, so no scalar
  // remainder loop is needed.
  for (; d < size;) {
    const size_t vl = traits::setvl(size - d);
    v_t max_v = traits::undef();
    if (amsgrad) {
      max_v = traits::load(max_exp_avg_sq_ptr + d, vl);
    }
    compute_tile(
        traits::load(param_ptr + d, vl),
        traits::load(grad_ptr + d, vl),
        traits::load(exp_avg_ptr + d, vl),
        traits::load(exp_avg_sq_ptr + d, vl),
        max_v,
        d,
        vl);
    d += vl;
  }
}

#endif  // __riscv_v_intrinsic


template <typename scalar_t, typename opmath_t, ADAM_MODE adam_mode>
std::enable_if_t<
    std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
    void>
    inline adam_math(
  scalar_t* param_ptr,
  scalar_t* exp_avg_ptr,
  scalar_t* exp_avg_sq_ptr,
  scalar_t* grad_ptr,
  scalar_t* max_exp_avg_sq_ptr,
  double lr,
  double bias_correction1,
  double bias_correction2,
  double exp_avg_grad_coefficient,
  double exp_avg_sq_grad_coefficient,
  double bias_correction2_sqrt,
  double eps,
  double weight_decay,
  double beta2,
  bool amsgrad,
  bool maximize,
  const float* grad_scale_ptr,
  int64_t size
){
  double step_size = lr / bias_correction1;
#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000
  adam_math_rvv<scalar_t, adam_mode>(
      param_ptr,
      exp_avg_ptr,
      exp_avg_sq_ptr,
      grad_ptr,
      max_exp_avg_sq_ptr,
      lr,
      step_size,
      exp_avg_grad_coefficient,
      exp_avg_sq_grad_coefficient,
      bias_correction2_sqrt,
      eps,
      weight_decay,
      beta2,
      amsgrad,
      maximize,
      grad_scale_ptr,
      size
  );
  return;
#endif
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec param_vec = Vec::loadu(param_ptr + d);
    Vec grad_vec = Vec::loadu(grad_ptr + d);
    if (grad_scale_ptr) {
      grad_vec = grad_vec / Vec(scalar_t(*grad_scale_ptr));
      Vec grad_vec_to_store = grad_vec;
      grad_vec_to_store.store(grad_ptr + d);
    }
    if (maximize) grad_vec = grad_vec * Vec(scalar_t(-1.0));
    if (weight_decay != 0.f){
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_vec += param_vec * Vec(scalar_t(weight_decay));
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_vec = param_vec * Vec(scalar_t(1 - lr * weight_decay));
      }
    }
    Vec exp_avg_vec = Vec::loadu(exp_avg_ptr + d);
    // exp_avg.lerp_(grad, 1 - beta1)
    const Vec lerp_weight = Vec(scalar_t(exp_avg_grad_coefficient));
    auto mask = lerp_weight.abs() < Vec(0.5);
    auto coeff = Vec::blendv(lerp_weight - Vec(1), lerp_weight, mask);
    auto base = Vec::blendv(grad_vec, exp_avg_vec, mask);
    exp_avg_vec = vec::fmadd(coeff, grad_vec - exp_avg_vec, base);

    Vec exp_avg_sq_vec = Vec::loadu(exp_avg_sq_ptr + d) * Vec(scalar_t(beta2)) +
        Vec(scalar_t(exp_avg_sq_grad_coefficient)) * grad_vec * grad_vec;
    exp_avg_vec.store(exp_avg_ptr + d);
    exp_avg_sq_vec.store(exp_avg_sq_ptr + d);

    Vec denom_vec;
    if (amsgrad) {
      Vec max_exp_avg_sq_vec =
          maximum(Vec::loadu(max_exp_avg_sq_ptr + d), exp_avg_sq_vec);
      max_exp_avg_sq_vec.store(max_exp_avg_sq_ptr + d);
      denom_vec =
          (max_exp_avg_sq_vec.sqrt() / Vec(scalar_t(bias_correction2_sqrt))) + Vec(scalar_t(eps));
    } else {
      denom_vec =
          (exp_avg_sq_vec.sqrt() / Vec(scalar_t(bias_correction2_sqrt))) + Vec(scalar_t(eps));
    }
    param_vec = param_vec + Vec(scalar_t(-step_size)) * exp_avg_vec / denom_vec;
    param_vec.store(param_ptr + d);
  }
  for (; d < size; d++) {
    scalar_t grad_val = grad_ptr[d];
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / scalar_t(*grad_scale_ptr);
      grad_ptr[d] = grad_val;
    }
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.f){
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_val += param_ptr[d] * scalar_t(weight_decay);
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_ptr[d] = param_ptr[d] * scalar_t(1 - lr * weight_decay);
      }
    }
    // exp_avg.lerp_(grad, 1 - beta1)
    auto is_lerp_weight_small = std::abs(scalar_t(exp_avg_grad_coefficient)) < scalar_t(0.5);
    if (is_lerp_weight_small) {
      exp_avg_ptr[d] = exp_avg_ptr[d] + scalar_t(exp_avg_grad_coefficient) * (grad_val - exp_avg_ptr[d]);
    } else {
      exp_avg_ptr[d] = grad_val - (grad_val - exp_avg_ptr[d]) * (scalar_t(1) - scalar_t(exp_avg_grad_coefficient));
    }
    exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * scalar_t(beta2);
    exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] +
        scalar_t(exp_avg_sq_grad_coefficient) * grad_val * grad_val;
    scalar_t demon_val;
    if (amsgrad) {
      max_exp_avg_sq_ptr[d] =
          std::max(max_exp_avg_sq_ptr[d], exp_avg_sq_ptr[d]);
      demon_val =
          std::sqrt(max_exp_avg_sq_ptr[d]) / scalar_t(bias_correction2_sqrt) + scalar_t(eps);
    } else {
      demon_val = std::sqrt(exp_avg_sq_ptr[d]) / scalar_t(bias_correction2_sqrt) + scalar_t(eps);
    }
    param_ptr[d] = param_ptr[d] - scalar_t(step_size) * exp_avg_ptr[d] / demon_val;
  }
}


template <typename scalar_t, ADAM_MODE adam_mode>
void adam_fused_step_impl(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& max_exp_avg_sq,
    const at::Tensor& state_step,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const float* grad_scale_ptr) {
  using opmath_t = at::opmath_type<scalar_t>;
  double step = state_step.item<float>();
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* exp_avg_data = exp_avg.data_ptr<scalar_t>();
  scalar_t* exp_avg_sq_data = exp_avg_sq.data_ptr<scalar_t>();
  scalar_t* max_exp_avg_sq_data = amsgrad ? max_exp_avg_sq.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

  // need to use double here to align with non-fused adam
  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);
  double exp_avg_grad_coefficient = 1 - beta1;
  double exp_avg_sq_grad_coefficient = 1 - beta2;
  double bias_correction2_sqrt = std::sqrt(bias_correction2);


  constexpr size_t cache_line_size = 64;
  constexpr int64_t cache_line_aligned_task_unit = cache_line_size / sizeof(scalar_t);
  size_t num_units = divup(param.numel(), cache_line_aligned_task_unit);

  auto adam_fn = [&](int64_t begin, int64_t end) {
        // local pointers
        begin *= cache_line_aligned_task_unit;
        end = std::min(end * cache_line_aligned_task_unit, param.numel());
        scalar_t* param_ptr = param_data + begin;
        scalar_t* exp_avg_ptr = exp_avg_data + begin;
        scalar_t* exp_avg_sq_ptr = exp_avg_sq_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* max_exp_avg_sq_ptr = amsgrad ? max_exp_avg_sq_data + begin : nullptr;

        const int64_t size = end - begin;
        adam_math<scalar_t, opmath_t, adam_mode>(
          param_ptr,
          exp_avg_ptr,
          exp_avg_sq_ptr,
          grad_ptr,
          max_exp_avg_sq_ptr,
          lr,
          bias_correction1,
          bias_correction2,
          exp_avg_grad_coefficient,
          exp_avg_sq_grad_coefficient,
          bias_correction2_sqrt,
          eps,
          weight_decay,
          beta2,
          amsgrad,
          maximize,
          grad_scale_ptr,
          size
        );
      };
  at::parallel_for(
      0, num_units, 0, adam_fn);
}

void fused_adam_kernel(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& max_exp_avg_sq,
    const at::Tensor& state_step,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const float* grad_scale_ptr,
    const ADAM_MODE adam_mode
  ) {
  Tensor grad_contiguous = grad.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_adam_kernel", [&] {
    if(adam_mode == ADAM_MODE::ORIGINAL){
      adam_fused_step_impl<scalar_t, ADAM_MODE::ORIGINAL>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale_ptr);
    } else {
      adam_fused_step_impl<scalar_t, ADAM_MODE::ADAMW>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale_ptr);
    }

  });
}

}

REGISTER_DISPATCH(fused_adam_stub, &fused_adam_kernel)
} // namespace at::native
