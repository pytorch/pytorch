#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Half.h>
#include <torch/csrc/Export.h>
#include <cstddef>
#include <cstdint>

#ifdef __FreeBSD__
#include <sys/endian.h>
#include <sys/types.h>
#define thp_bswap16(x) bswap16(x)
#define thp_bswap32(x) bswap32(x)
#define thp_bswap64(x) bswap64(x)
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define thp_bswap16(x) OSSwapInt16(x)
#define thp_bswap32(x) OSSwapInt32(x)
#define thp_bswap64(x) OSSwapInt64(x)
#elif defined(__GNUC__) && !defined(__MINGW32__)
#include <byteswap.h>
#define thp_bswap16(x) bswap_16(x)
#define thp_bswap32(x) bswap_32(x)
#define thp_bswap64(x) bswap_64(x)
#elif defined _WIN32 || defined _WIN64
#define thp_bswap16(x) _byteswap_ushort(x)
#define thp_bswap32(x) _byteswap_ulong(x)
#define thp_bswap64(x) _byteswap_uint64(x)
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define to_be16(x) thp_bswap16(x)
#define from_be16(x) thp_bswap16(x)
#define to_be32(x) thp_bswap32(x)
#define from_be32(x) thp_bswap32(x)
#define to_be64(x) thp_bswap64(x)
#define from_be64(x) thp_bswap64(x)
#define to_le16(x) (x)
#define from_le16(x) (x)
#define to_le32(x) (x)
#define from_le32(x) (x)
#define to_le64(x) (x)
#define from_le64(x) (x)
#else
#define to_be16(x) (x)
#define from_be16(x) (x)
#define to_be32(x) (x)
#define from_be32(x) (x)
#define to_be64(x) (x)
#define from_be64(x) (x)
#define to_le16(x) thp_bswap16(x)
#define from_le16(x) thp_bswap16(x)
#define to_le32(x) thp_bswap32(x)
#define from_le32(x) thp_bswap32(x)
#define to_le64(x) thp_bswap64(x)
#define from_le64(x) thp_bswap64(x)
#endif

namespace torch {
namespace utils {

enum THPByteOrder { THP_LITTLE_ENDIAN = 0, THP_BIG_ENDIAN = 1 };

TORCH_API THPByteOrder THP_nativeByteOrder();

TORCH_API void THP_decodeInt16Buffer(
    int16_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeInt32Buffer(
    int32_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeInt64Buffer(
    int64_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeHalfBuffer(
    c10::Half* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeFloatBuffer(
    float* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeDoubleBuffer(
    double* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeBoolBuffer(
    bool* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeBFloat16Buffer(
    at::BFloat16* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeComplexFloatBuffer(
    c10::complex<float>* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);
TORCH_API void THP_decodeComplexDoubleBuffer(
    c10::complex<double>* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len);

TORCH_API void THP_decodeInt16Buffer(
    int16_t* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeInt32Buffer(
    int32_t* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeInt64Buffer(
    int64_t* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeHalfBuffer(
    c10::Half* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeFloatBuffer(
    float* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeDoubleBuffer(
    double* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeBoolBuffer(
    bool* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeBFloat16Buffer(
    at::BFloat16* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeFloat8_e5m2Buffer(
    at::Float8_e5m2* dst,
    const uint8_t* src,
    size_t len);
TORCH_API void THP_decodeFloat8_e4m3fnBuffer(
    at::Float8_e4m3fn* dst,
    const uint8_t* src,
    size_t len);
TORCH_API void THP_decodeComplexFloatBuffer(
    c10::complex<float>* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_decodeComplexDoubleBuffer(
    c10::complex<double>* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len);

TORCH_API void THP_encodeInt16Buffer(
    uint8_t* dst,
    const int16_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_encodeInt32Buffer(
    uint8_t* dst,
    const int32_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_encodeInt64Buffer(
    uint8_t* dst,
    const int64_t* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_encodeFloatBuffer(
    uint8_t* dst,
    const float* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_encodeDoubleBuffer(
    uint8_t* dst,
    const double* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_encodeComplexFloatBuffer(
    uint8_t* dst,
    const c10::complex<float>* src,
    THPByteOrder order,
    size_t len);
TORCH_API void THP_encodeComplexDoubleBuffer(
    uint8_t* dst,
    const c10::complex<double>* src,
    THPByteOrder order,
    size_t len);

} // namespace utils
} // namespace torch
