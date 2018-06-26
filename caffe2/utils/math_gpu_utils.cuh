#ifndef CAFFE2_UTILS_MATH_GPU_UTILS_CUH_
#define CAFFE2_UTILS_MATH_GPU_UTILS_CUH_

namespace caffe2 {
namespace math {
namespace gpu_utils {

inline __host__ __device__ bool Not(const bool x) {
  return !x;
}

template <typename T>
inline __host__ __device__ T Sign(const T x) {
  return x > 0 ? T(1) : (x < 0 ? T(-1) : T(0));
}

template <typename T>
inline __host__ __device__ T Negate(const T x) {
  return -x;
}

template <typename T>
inline __host__ __device__ T Inv(const T x) {
  return T(1) / x;
}

template <typename T>
inline __host__ __device__ T Square(const T x) {
  return x * x;
}

template <typename T>
inline __host__ __device__ T Cube(const T x) {
  return x * x * x;
}

} // namespace gpu_utils
} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_GPU_UTILS_CUH_
