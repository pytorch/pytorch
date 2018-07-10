#include "hip/hip_runtime.h"

namespace {
template <typename T>
inline __device__ T hip_pow(T x, T y);

template <typename T>
inline __device__ T hip_abs(T x);

template <>
inline __device__ float hip_pow<float>(float x, float y)
{
    return powf(x, y);
}
template <>
inline __device__ double hip_pow<double>(double x, double y)
{
    return pow(x, y);
}

template <>
inline __device__ float hip_abs(float x)
{
    return fabsf(x);
}
template <>
inline __device__ double hip_abs(double x)
{
    return fabs(x);
}
}