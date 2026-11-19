#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <dnnl.hpp>

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

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
