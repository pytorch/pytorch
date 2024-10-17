#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/rvv/rvv_helper.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>
#include <c10/util/qint32.h>
#include <array>

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<c10::qint32> {
 private:
    fixed_vint32m2_t vals;

 public:
    static constexpr int size() {
        return 8;
    }

    static constexpr int float_num_vecs() {
        return 1;
    }
    static constexpr int int_num_vecs() {
        return 1;
    }

    using float_vec_return_type = std::array<Vectorized<float>, 1>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 1>;
    using value_type = typename c10::qint32::underlying;

    Vectorized() {}
    Vectorized(vint32m2_t v) : vals(v) {}
    // Broadcast constructor
    Vectorized(const c10::qint32& val) {
        vals =  __riscv_vmv_v_x_i32m2(val.val_, VQINT32_VL);
    }

    operator vint32m2_t() const {
        return vals;
    }

    void store(void* ptr, int count = size()) const {
        value_type tmp_values[size()];
        __riscv_vse32_v_i32m2(reinterpret_cast<value_type*>(tmp_values), vals, VQINT32_VL);
        std::memcpy(ptr, tmp_values, count * sizeof(value_type));
    }

    static Vectorized<c10::qint32> loadu(const void* ptr, int count = size()) {
        __at_align__ value_type tmp_values[size()];
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        std::memcpy(
            tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        return __riscv_vle32_v_i32m2(tmp_values, VQINT32_VL);
    }

    float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> /*zero_point*/,
      Vectorized<float> scale_zp_premul) const {
        vfloat32m2_t float_vals = __riscv_vfcvt_f_x_v_f32m2(vals, VQINT32_VL);
        return {vec::fmadd(scale, Vectorized<float>(float_vals), scale_zp_premul)};
    }

    float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
        vfloat32m2_t float_vals = __riscv_vfcvt_f_x_v_f32m2(vals, VQINT32_VL);
        return {(Vectorized<float>(float_vals) - zero_point) * scale};
    }

    static Vectorized<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
        Vectorized<c10::qint32> retval;
        auto rhs_data = (vfloat32m2_t)rhs[0];
        at::native::quantize_vec<c10::qint32, /*precision=*/32>(
            scale, zero_point, (float*)&rhs_data, (c10::qint32*)&retval.vals, 8);
        return retval;
    }

    Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
        return __riscv_vmax_vv_i32m2(vals, b.vals, VQINT32_VL);
    }

    Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
        return __riscv_vmin_vv_i32m2(vals, b.vals, VQINT32_VL);
    }

    Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const {
        return maximum(zero_point);
    }

    Vectorized<c10::qint32> relu6(
      Vectorized<c10::qint32> zero_point,
      Vectorized<c10::qint32> q_six) {
        return __riscv_vmin_vv_i32m2(
            __riscv_vmax_vv_i32m2(vals, zero_point.vals, VQINT32_VL),
            q_six.vals,
            VQINT32_VL);
    }

    int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
        return {__riscv_vsub_vv_i32m2(vals, b, VQINT32_VL)};
    }

    static Vectorized<c10::qint32> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
        vfloat32m2_t multiplier_v = __riscv_vfmv_v_f_f32m2(multiplier, VQINT32_VL);
        vint32m2_t zero_point_v = __riscv_vmv_v_x_i32m2(zero_point, VQINT32_VL);
        vfloat32m2_t scaled = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(inp[0], VQINT32_VL), multiplier_v, VQINT32_VL);
        vint32m2_t rounded = __riscv_vfcvt_x_f_v_i32m2(scaled, VQINT32_VL);
        return __riscv_vadd_vv_i32m2(rounded, zero_point_v, VQINT32_VL);
    }
};

template <>
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
    return a.maximum(b);
}

template <>
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
    return __riscv_vmul_vv_i32m2(a, b, VQINT32_VL);
}

template <>
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
    return __riscv_vadd_vv_i32m2(a, b, VQINT32_VL);
}

}
}
}
