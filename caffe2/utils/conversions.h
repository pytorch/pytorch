#pragma once

// See Note [hip-clang differences to hcc]

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || \
    defined(__HIP__) || (defined(__clang__) && defined(__CUDA__))
#define CONVERSIONS_DECL __host__ __device__ inline
#else
#define CONVERSIONS_DECL inline
#endif

#ifdef _MSC_VER
#undef IN
#undef OUT
#endif

namespace caffe2 {

namespace convert {

template <typename IN, typename OUT>
CONVERSIONS_DECL OUT To(const IN in) {
  return static_cast<OUT>(in);
}

template <typename OUT, typename IN>
CONVERSIONS_DECL OUT Get(IN x) {
  return static_cast<OUT>(x);
}

}; // namespace convert

}; // namespace caffe2

#undef CONVERSIONS_DECL
