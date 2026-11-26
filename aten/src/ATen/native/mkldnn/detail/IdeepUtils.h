#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <dnnl.hpp>
// Ported from ideep/utils.hpp.
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace at::native::mkldnn {

template <typename T, typename... Items>
constexpr bool one_of(T val, Items... items) {
  return ((val == items) || ...);
}

inline bool is_avx2_vnni_2() {
  static const bool result =
      dnnl::get_effective_cpu_isa() == dnnl::cpu_isa::avx2_vnni_2;
  return result;
}

// Ported from ideep::IDEEP_DEF_SCALE / IDEEP_DEF_ZP / IDEEP_EMPTY_SCALE /
// IDEEP_EMPTY_ZP.
const std::vector<float> DEF_SCALE{1.0f};
const std::vector<int32_t> DEF_ZP{0};
const std::vector<float> EMPTY_SCALE;
const std::vector<int32_t> EMPTY_ZP;

// Ported from ideep::lowp_kind.
enum lowp_kind {
  u8s8 = 0,
  s8s8 = 1,
  LOWP_U8S8 = u8s8,
  LOWP_S8S8 = s8s8,
};

// Ported from ideep::rnn_kind.
enum rnn_kind { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

// Ported from ideep::has_bf16_type_support.
inline bool has_bf16_type_support() {
  // for v1.8
  // static bool support_bf16 = isa >= dnnl::cpu_isa::avx512_core
  //                           && isa != dnnl::cpu_isa::avx2_vnni;
  static bool support_bf16 =
      dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx2_vnni_2;
  return support_bf16;
}

// Ported from ideep::has_fp16_type_support.
inline bool has_fp16_type_support() {
  static bool support_fp16 =
      dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_fp16 ||
      dnnl::get_effective_cpu_isa() == dnnl::cpu_isa::avx2_vnni_2;
  return support_fp16;
}

namespace utils {

// Ported from ideep::utils::fmap.
template <
    typename F,
    typename T,
    typename U = decltype(std::declval<F>()(std::declval<T>()))>
std::vector<U> fmap(const std::vector<T>& vec, const F& f) {
  std::vector<U> result;
  std::transform(vec.begin(), vec.end(), std::back_inserter(result), f);
  return result;
}

// Ported from ideep::utils::any_le.
template <typename T>
inline bool any_le(const std::vector<T>& v, T i) {
  return std::any_of(v.begin(), v.end(), [i](T k) { return k <= i; });
}

// Ported from ideep::utils::get_compatible_dilates.
inline dnnl::memory::dims get_compatible_dilates(
    const dnnl::memory::dims& dilates,
    int input_size = 4) {
  if (!dilates.empty() && !any_le(dilates, static_cast<dnnl::memory::dim>(0)))
    return fmap(dilates, [](dnnl::memory::dim x) { return x - 1; });
  if (4 == input_size) {
    return {0, 0};
  } else {
    return {0, 0, 0};
  }
}

// Ported from ideep::utils::group_dims.
inline dnnl::memory::dims group_dims(
    const dnnl::memory::dims& adims,
    dnnl::memory::dim groups) {
  auto new_dims = adims;
  new_dims.insert(new_dims.begin(), groups);
  new_dims[1] /= groups;
  return new_dims;
}

// Ported from ideep::utils::op_scale_mask.
inline int op_scale_mask(dnnl::memory::dim scale_size) {
  return scale_size > 1 ? 2 : 0;
}

// Ported from ideep::utils::tensor_scale_mask.
inline int tensor_scale_mask(dnnl::memory::dim scale_size, bool grouped) {
  return scale_size > 1 ? grouped ? 3 : 1 : 0;
}

// Ported from ideep::utils::conv_weight_scale_mask.
inline int conv_weight_scale_mask(
    int scale_size,
    int channels_per_group,
    int groups,
    bool is_deconv) {
  /* Mask defines the correspondence between the tensor dimensions and the scales vector.
   * If scale is a scalar and applies to the entire tensor, mask is 0.
   * If scales apply to i-th dimension, i-th bit of mask should be set.
   * Refer to oneDNN's doc for more details.
   * Example 1: weight format is oihw, scales apply on the o dimension (the 1st dimension).
   *            The first bit should be set. Mask = 0001b = 1
   * Example 2: weight format is goihw, scales apply on the g and o dimensions (the 1st and
   *            2nd dimensions). The first and second bits should be set. Mask = 00011b = 3
  */
  if (scale_size <= 1) return 0;
  if (groups > 1) {
    if (is_deconv) {
      // format = giow/giohw/giodhw for 1/2/3d
      if (scale_size == channels_per_group) {
        // scales apply on o (3rd dimension)
        return 4;
      } else {
        // scale_size = groups * channels_per_group
        // scales apply on g + o (1st and 3rd dimensions)
        return 5;
      }
    } else {
      // format = goiw/goihw/goidhw for 1/2/3d
      if (scale_size == channels_per_group) {
        // scales apply on o (2nd dimension)
        return 2;
      } else {
        // scale_size = groups * channels_per_group
        // scales apply on g + o (1st and 2nd dimensions)
        return 3;
      }
    }
  }
  // scales apply on o dimension
  // i.e., 1st dimension for conv and 2nd for deconv
  return is_deconv ? 2 : 1;
}

// Ported from ideep::utils::tensor_zp_mask.
inline int tensor_zp_mask(dnnl::memory::dim zp_size) {
  return zp_size > 1 ? 1 : 0;
}

// Ported from ideep::utils::array_cmp.
template <typename T>
inline bool array_cmp(const T* a1, const T* a2, size_t size) {
  for (size_t i = 0; i < size; ++i)
    if (a1[i] != a2[i])
      return false;
  return true;
}

// Ported from ideep::utils::set_verbose.
inline int set_verbose(int level) {
  dnnl::status ret = dnnl::set_verbose(level);
  return ret == dnnl::status::success;
}

} // namespace utils

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
