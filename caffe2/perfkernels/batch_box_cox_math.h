#pragma once

#include <cstdint>

#include <ATen/cpu/vec/vec.h>

using Vf = at::vec::Vectorized<float>;
using Vi = at::vec::Vectorized<int32_t>;

//Max ULP: 2.85
inline at::vec::Vectorized<float> logf(at::vec::Vectorized<float> d) {

  const Vf u = d * Vf(1.0f / 0.75f);
  Vi e = cast<int32_t>(u);
#ifdef __aarch64__
  e = svget_neonq(svand_n_s32_x(svptrue_b32(), svset_neonq(svundef_s32(), e), 0xff << 23));
#else
  e = e & Vi(0xff << 23);
#endif
  e = e - Vi(0x7f << 23);
  const Vf m = cast<float>(cast<int32_t>(d) - e);

  Vf ef = convert_to_fp_of_same_size<float>(e >> Vi(23));
#ifdef __aarch64__
  const Vf mDecd = svget_neonq(svsub_n_f32_x(svptrue_b32(), svset_neonq(svundef_f32(), m), 1.0));
  const Vf mIncd = svget_neonq(svadd_n_f32_x(svptrue_b32(), svset_neonq(svundef_f32(), m), 1.0));
#else
  const Vf one = Vf(1.0f);
  const Vf mDecd = m - one;
  const Vf mIncd = m + one;
#endif
  const Vf x = mDecd / mIncd;
  const Vf x2 = x * x;

  Vf t(0.2392828464508056640625f);
  t = fmadd(t, x2, Vf(0.28518211841583251953125f));
  t = fmadd(t, x2, Vf(0.400005877017974853515625f));
  t = fmadd(t, x2, Vf(0.666666686534881591796875f));
  t = fmadd(t, x2, Vf(2.0f));

  // log(d) = x * t + ln(2) * e
  Vf r = fmadd(x, t, Vf(0.693147180559945286226764f) * ef);
  
  return r;
}

//Max ULP: 1.95
inline at::vec::Vectorized<float> expf(const at::vec::Vectorized<float> x_in) {

  // Clamp interval set to prevent denormals!
  const Vf max_input(88.722839f);
  const Vf min_input(-87.33654f);
  // 2^23 + 127: rounds x*log2(e) into the low mantissa bits, pre-biased so
  // that bits(z) << 23 is exactly 2^n.
  const Vf shift(0x1.0000FEp+23f);

  const Vf x = clamp(x_in, min_input, max_input);

  const Vf z = fmadd(x, Vf(0x1.715476p+0f), shift);
  const Vf n = z - shift;
#ifdef __aarch64__
  const Vi w = svget_neonq(svlsl_n_s32_x(svptrue_b32(), svset_neonq(svundef_s32(), cast<int32_t>(z)), 23));
#else
  const Vi w = cast<int32_t>(z) << Vi(23);
#endif
  const Vf scale = cast<float>(w);

  const Vf r_hi = fnmadd(n, Vf(0x1.62E400p-1f), x);
  const Vf r = fnmadd(n, Vf(0x1.7F7D1Cp-20f), r_hi);
  const Vf r2 = r * r;

  const Vf c = fmadd(r, Vf(0x1.0E4020p-7f), Vf(0x1.573E2Ep-5f));
  const Vf b = fmadd(r, Vf(0x1.555E66p-3f), Vf(0x1.FFFDB6p-2f));
  const Vf a = r * Vf(0x1.FFFFECp-1f);

  return fmadd(fmadd(fmadd(c, r2, b), r2, a), scale, scale);
}

inline at::vec::Vectorized<float> powf(at::vec::Vectorized<float> a, at::vec::Vectorized<float> b) {
  return expf(logf(a) * b);
}

// Fallback to veclib for doubles

inline at::vec::Vectorized<double> logf(at::vec::Vectorized<double> d) {
  return d.log();
}

inline at::vec::Vectorized<double> powf(at::vec::Vectorized<double> a, at::vec::Vectorized<double> b) {
  return a.pow(b);
}
