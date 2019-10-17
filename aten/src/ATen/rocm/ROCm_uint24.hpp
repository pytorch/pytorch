#pragma once

#ifdef __HIP_PLATFORM_HCC__
template<typename T>
struct __attribute__((packed)) uint24_t {
    T x : 24;
};

template<typename T>
inline __host__ __device__
T mul24(T x, T y) noexcept
{
    return uint24_t<T>{x}.x * uint24_t<T>{y}.x;
}

template long long mul24<long long>(long long, long long);

template<typename T>
inline __host__ __device__
T mad24(T x, T y, T z) noexcept
{
    return uint24_t<T>{x}.x * uint24_t<T>{y}.x + uint24_t<T>{z}.x;
}

template long long mad24<long long>(long long, long long, long long);

template<typename T>
inline __host__ __device__
T div24(T x, T y) noexcept
{
    return uint24_t<T>{x}.x / uint24_t<T>{y}.x;
}

template long long div24<long long>(long long, long long);

template<typename T>
inline __host__ __device__
T mod24(T x, T y) noexcept
{
    return uint24_t<T>{x}.x % uint24_t<T>{y}.x;
}

template long long mod24<long long>(long long, long long);
#endif
