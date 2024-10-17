#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/rvv/rvv_helper.h>
#include <c10/util/qint8.h>
#include <array>

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<c10::qint8> {
 private:
    fixed_vint8m2_t vals;

 public:
    static constexpr int size() {
        return 32;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }
    static constexpr int int_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vectorized<float>, 4>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;
    using value_type = typename c10::qint8::underlying;

    Vectorized() {}
    Vectorized(vint8m2_t v) : vals(v) {}
    // Broadcast constructor
    Vectorized(const c10::qint8& val) {
        vals = __riscv_vmv_v_x_i8m2(val.val_, VQINT8_VL);
    }

    Vectorized(const Vectorized<c10::qint8>& other) : vals(other.vals) {}

    operator vint8m2_t() const {
        return vals;
    }

    void store(void* ptr, int count = size()) const {
        __riscv_vse8_v_i8m2(reinterpret_cast<value_type*>(ptr), vals, count);
    }

    static Vectorized<c10::qint8> loadu(const void* ptr, int count = size()) {
        vint8m2_t zero_vec = __riscv_vmv_v_x_i8m2(0, VQINT8_VL);
        return __riscv_vle8_v_i8m2_tu(zero_vec, reinterpret_cast<const value_type*>(ptr), count);
    }

 public:
    float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
        vint64m2_t i64_vec= __riscv_vreinterpret_v_i8m2_i64m2(vals);

        vint64m2_t int_val0 = __riscv_vslidedown_vx_i64m2(i64_vec, 0, 4);
        vint64m2_t int_val1 = __riscv_vslidedown_vx_i64m2(i64_vec, 1, 4);
        vint64m2_t int_val2 = __riscv_vslidedown_vx_i64m2(i64_vec, 2, 4);
        vint64m2_t int_val3 = __riscv_vslidedown_vx_i64m2(i64_vec, 3, 4);

        vint8m2_t i8_val0 = __riscv_vreinterpret_v_i64m2_i8m2(int_val0);
        vint8m2_t i8_val1 = __riscv_vreinterpret_v_i64m2_i8m2(int_val1);
        vint8m2_t i8_val2 = __riscv_vreinterpret_v_i64m2_i8m2(int_val2);
        vint8m2_t i8_val3 = __riscv_vreinterpret_v_i64m2_i8m2(int_val3);

        vfloat32m2_t float_val0 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val0), 8), 8);
        vfloat32m2_t float_val1 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val1), 8), 8);
        vfloat32m2_t float_val2 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val2), 8), 8);
        vfloat32m2_t float_val3 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val3), 8), 8);

        auto val0 =
            vec::fmadd(scale, Vectorized<float>(float_val0), scale_zp_premul);
        auto val1 =
            vec::fmadd(scale, Vectorized<float>(float_val1), scale_zp_premul);
        auto val2 =
            vec::fmadd(scale, Vectorized<float>(float_val2), scale_zp_premul);
        auto val3 =
            vec::fmadd(scale, Vectorized<float>(float_val3), scale_zp_premul);
        return {val0, val1, val2, val3};
    }

    float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
        vint64m2_t i64_vec= __riscv_vreinterpret_v_i8m2_i64m2(vals);

        vint64m2_t int_val0 = __riscv_vslidedown_vx_i64m2(i64_vec, 0, 4);
        vint64m2_t int_val1 = __riscv_vslidedown_vx_i64m2(i64_vec, 1, 4);
        vint64m2_t int_val2 = __riscv_vslidedown_vx_i64m2(i64_vec, 2, 4);
        vint64m2_t int_val3 = __riscv_vslidedown_vx_i64m2(i64_vec, 3, 4);

        vint8m2_t i8_val0 = __riscv_vreinterpret_v_i64m2_i8m2(int_val0);
        vint8m2_t i8_val1 = __riscv_vreinterpret_v_i64m2_i8m2(int_val1);
        vint8m2_t i8_val2 = __riscv_vreinterpret_v_i64m2_i8m2(int_val2);
        vint8m2_t i8_val3 = __riscv_vreinterpret_v_i64m2_i8m2(int_val3);

        vfloat32m2_t float_val0 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val0), 8), 8);
        vfloat32m2_t float_val1 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val1), 8), 8);
        vfloat32m2_t float_val2 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val2), 8), 8);
        vfloat32m2_t float_val3 =
            __riscv_vfcvt_f_x_v_f32m2(__riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val3), 8), 8);

        auto val0 = (Vectorized<float>(float_val0) - zero_point) * scale;
        auto val1 = (Vectorized<float>(float_val1) - zero_point) * scale;
        auto val2 = (Vectorized<float>(float_val2) - zero_point) * scale;
        auto val3 = (Vectorized<float>(float_val3) - zero_point) * scale;
        return {val0, val1, val2, val3};
    }

    static Vectorized<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
        Vectorized<float> vf0 = rhs[0];
        Vectorized<float> vf1 = rhs[1];
        Vectorized<float> vf2 = rhs[2];
        Vectorized<float> vf3 = rhs[3];

        vfloat32m2_t vecf0 = __riscv_vfmul_vf_f32m2(vf0, inverse_scale, 8);
        vfloat32m2_t vecf1 = __riscv_vfmul_vf_f32m2(vf1, inverse_scale, 8);
        vfloat32m2_t vecf2 = __riscv_vfmul_vf_f32m2(vf2, inverse_scale, 8);
        vfloat32m2_t vecf3 = __riscv_vfmul_vf_f32m2(vf3, inverse_scale, 8);

        vecf0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vfcvt_x_f_v_i32m2_rm(vecf0, __RISCV_FRM_RNE, 8), 8);
        vecf1 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vfcvt_x_f_v_i32m2_rm(vecf1, __RISCV_FRM_RNE, 8), 8);
        vecf2 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vfcvt_x_f_v_i32m2_rm(vecf2, __RISCV_FRM_RNE, 8), 8);
        vecf3 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vfcvt_x_f_v_i32m2_rm(vecf3, __RISCV_FRM_RNE, 8), 8);

        vecf0 = __riscv_vfadd_vf_f32m2(vecf0, (float)zero_point, 8);
        vecf1 = __riscv_vfadd_vf_f32m2(vecf1, (float)zero_point, 8);
        vecf2 = __riscv_vfadd_vf_f32m2(vecf2, (float)zero_point, 8);
        vecf3 = __riscv_vfadd_vf_f32m2(vecf3, (float)zero_point, 8);

        vint32m2_t veci0 = __riscv_vfcvt_rtz_x_f_v_i32m2(vecf0, 8);
        vint32m2_t veci1 = __riscv_vfcvt_rtz_x_f_v_i32m2(vecf1, 8);
        vint32m2_t veci2 = __riscv_vfcvt_rtz_x_f_v_i32m2(vecf2, 8);
        vint32m2_t veci3 = __riscv_vfcvt_rtz_x_f_v_i32m2(vecf3, 8);

        vint16m1_t vecshi0 = __riscv_vnclip_wx_i16m1(veci0, 0, __RISCV_VXRM_RDN, 8);
        vint16m1_t vecshi1 = __riscv_vnclip_wx_i16m1(veci1, 0, __RISCV_VXRM_RDN, 8);
        vint16m1_t vecshi2 = __riscv_vnclip_wx_i16m1(veci2, 0, __RISCV_VXRM_RDN, 8);
        vint16m1_t vecshi3 = __riscv_vnclip_wx_i16m1(veci3, 0, __RISCV_VXRM_RDN, 8);

        vint8m2_t vec0 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi0, 0, __RISCV_VXRM_RDN, 8));
        vint8m2_t vec1 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi1, 0, __RISCV_VXRM_RDN, 8));
        vint8m2_t vec2 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi2, 0, __RISCV_VXRM_RDN, 8));
        vint8m2_t vec3 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi3, 0, __RISCV_VXRM_RDN, 8));

        vint8m2_t res;
        res = __riscv_vslideup_vx_i8m2(vec0, vec1, 8, VQINT8_VL);
        res = __riscv_vslideup_vx_i8m2(res, vec2, 16, VQINT8_VL);
        res = __riscv_vslideup_vx_i8m2(res, vec3, 24, VQINT8_VL);
        return Vectorized<c10::qint8>(res);
    }

    Vectorized<c10::qint8> maximum(Vectorized<c10::qint8> b) const {
        return __riscv_vmax_vv_i8m2(vals, b.vals, VQINT8_VL);
    }

    Vectorized<c10::qint8> minimum(Vectorized<c10::qint8> b) const {
        return __riscv_vmin_vv_i8m2(vals, b.vals, VQINT8_VL);
    }

    Vectorized<c10::qint8> relu(Vectorized<c10::qint8> zero_point) const {
        return maximum(zero_point);
    }

    Vectorized<c10::qint8> relu6(
      Vectorized<c10::qint8> zero_point,
      Vectorized<c10::qint8> q_six) {
        return __riscv_vmin_vv_i8m2(
          __riscv_vmax_vv_i8m2(vals, zero_point.vals, VQINT8_VL),
          q_six.vals,
          VQINT8_VL);
    }

    int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
        vint64m2_t vals_i64_vec= __riscv_vreinterpret_v_i8m2_i64m2(vals);

        vint64m2_t int_val0 = __riscv_vslidedown_vx_i64m2(vals_i64_vec, 0, 4);
        vint64m2_t int_val1 = __riscv_vslidedown_vx_i64m2(vals_i64_vec, 1, 4);
        vint64m2_t int_val2 = __riscv_vslidedown_vx_i64m2(vals_i64_vec, 2, 4);
        vint64m2_t int_val3 = __riscv_vslidedown_vx_i64m2(vals_i64_vec, 3, 4);

        vint8m2_t i8_val0 = __riscv_vreinterpret_v_i64m2_i8m2(int_val0);
        vint8m2_t i8_val1 = __riscv_vreinterpret_v_i64m2_i8m2(int_val1);
        vint8m2_t i8_val2 = __riscv_vreinterpret_v_i64m2_i8m2(int_val2);
        vint8m2_t i8_val3 = __riscv_vreinterpret_v_i64m2_i8m2(int_val3);

        vint32m2_t int32_val0 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val0), 8);
        vint32m2_t int32_val1 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val1), 8);
        vint32m2_t int32_val2 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val2), 8);
        vint32m2_t int32_val3 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_val3), 8);

        vint64m2_t b_i64_vec= __riscv_vreinterpret_v_i8m2_i64m2(b.vals);

        vint64m2_t int_b0 = __riscv_vslidedown_vx_i64m2(b_i64_vec, 0, 4);
        vint64m2_t int_b1 = __riscv_vslidedown_vx_i64m2(b_i64_vec, 1, 4);
        vint64m2_t int_b2 = __riscv_vslidedown_vx_i64m2(b_i64_vec, 2, 4);
        vint64m2_t int_b3 = __riscv_vslidedown_vx_i64m2(b_i64_vec, 3, 4);

        vint8m2_t i8_b0 = __riscv_vreinterpret_v_i64m2_i8m2(int_b0);
        vint8m2_t i8_b1 = __riscv_vreinterpret_v_i64m2_i8m2(int_b1);
        vint8m2_t i8_b2 = __riscv_vreinterpret_v_i64m2_i8m2(int_b2);
        vint8m2_t i8_b3 = __riscv_vreinterpret_v_i64m2_i8m2(int_b3);

        vint32m2_t int32_b0 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_b0), 8);
        vint32m2_t int32_b1 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_b1), 8);
        vint32m2_t int32_b2 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_b2), 8);
        vint32m2_t int32_b3 = __riscv_vsext_vf4_i32m2(__riscv_vlmul_trunc_v_i8m2_i8mf2(i8_b3), 8);

        vint32m2_t res_0 = __riscv_vsub_vv_i32m2(int32_val0, int32_b0, 8);
        vint32m2_t res_1 = __riscv_vsub_vv_i32m2(int32_val1, int32_b1, 8);
        vint32m2_t res_2 = __riscv_vsub_vv_i32m2(int32_val2, int32_b2, 8);
        vint32m2_t res_3 = __riscv_vsub_vv_i32m2(int32_val3, int32_b3, 8);

        return {Vectorized<c10::qint32>(res_0),
                Vectorized<c10::qint32>(res_1),
                Vectorized<c10::qint32>(res_2),
                Vectorized<c10::qint32>(res_3)};
    }

    static Vectorized<c10::qint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
        Vectorized<c10::qint32> vi0 = inp[0];
        Vectorized<c10::qint32> vi1 = inp[1];
        Vectorized<c10::qint32> vi2 = inp[2];
        Vectorized<c10::qint32> vi3 = inp[3];

        vfloat32m2_t vecf0 = __riscv_vfcvt_f_x_v_f32m2(vi0, 8);
        vfloat32m2_t vecf1 = __riscv_vfcvt_f_x_v_f32m2(vi1, 8);
        vfloat32m2_t vecf2 = __riscv_vfcvt_f_x_v_f32m2(vi2, 8);
        vfloat32m2_t vecf3 = __riscv_vfcvt_f_x_v_f32m2(vi3, 8);

        vecf0 = __riscv_vfmul_vf_f32m2(vecf0, multiplier, 8);
        vecf1 = __riscv_vfmul_vf_f32m2(vecf1, multiplier, 8);
        vecf2 = __riscv_vfmul_vf_f32m2(vecf2, multiplier, 8);
        vecf3 = __riscv_vfmul_vf_f32m2(vecf3, multiplier, 8);

        vint32m2_t veci0 = __riscv_vfcvt_x_f_v_i32m2(vecf0, 8);
        vint32m2_t veci1 = __riscv_vfcvt_x_f_v_i32m2(vecf1, 8);
        vint32m2_t veci2 = __riscv_vfcvt_x_f_v_i32m2(vecf2, 8);
        vint32m2_t veci3 = __riscv_vfcvt_x_f_v_i32m2(vecf3, 8);

        veci0 = __riscv_vadd_vx_i32m2(veci0, zero_point, 8);
        veci1 = __riscv_vadd_vx_i32m2(veci1, zero_point, 8);
        veci2 = __riscv_vadd_vx_i32m2(veci2, zero_point, 8);
        veci3 = __riscv_vadd_vx_i32m2(veci3, zero_point, 8);

        vint16m1_t vecshi0 = __riscv_vnclip_wx_i16m1(veci0, 0, __RISCV_VXRM_RDN, 8);
        vint16m1_t vecshi1 = __riscv_vnclip_wx_i16m1(veci1, 0, __RISCV_VXRM_RDN, 8);
        vint16m1_t vecshi2 = __riscv_vnclip_wx_i16m1(veci2, 0, __RISCV_VXRM_RDN, 8);
        vint16m1_t vecshi3 = __riscv_vnclip_wx_i16m1(veci3, 0, __RISCV_VXRM_RDN, 8);

        vint8m2_t vec0 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi0, 0, __RISCV_VXRM_RDN, 8));
        vint8m2_t vec1 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi1, 0, __RISCV_VXRM_RDN, 8));
        vint8m2_t vec2 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi2, 0, __RISCV_VXRM_RDN, 8));
        vint8m2_t vec3 = __riscv_vlmul_ext_v_i8mf2_i8m2(__riscv_vnclip_wx_i8mf2(vecshi3, 0, __RISCV_VXRM_RDN, 8));

        vint8m2_t res;
        res = __riscv_vslideup_vx_i8m2(vec0, vec1, 8, VQINT8_VL);
        res = __riscv_vslideup_vx_i8m2(res, vec2, 16, VQINT8_VL);
        res = __riscv_vslideup_vx_i8m2(res, vec3, 24, VQINT8_VL);
        return Vectorized<c10::qint8>(res);
    }

};

template <>
Vectorized<c10::qint8> inline maximum(const Vectorized<c10::qint8>& a, const Vectorized<c10::qint8>& b) {
    return a.maximum(b);
}

} // namespace
} // namespace vec
} // namespace at
