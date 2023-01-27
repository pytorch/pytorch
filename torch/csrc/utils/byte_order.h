#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/csrc/Export.h>
#include <cstddef>
#include <cstdint>

namespace {

static inline void swapBytes16(void* ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint16_t output;
  memcpy(&output, ptr, sizeof(uint16_t));
#if defined(_MSC_VER) && !defined(_DEBUG)
  output = _byteswap_ushort(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap16(output);
#else
  uint16_t Hi = output >> 8;
  uint16_t Lo = output << 8;
  output = Hi | Lo;
#endif
  memcpy(ptr, &output, sizeof(uint16_t));
}

static inline void swapBytes32(void* ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t output;
  memcpy(&output, ptr, sizeof(uint32_t));
#if defined(_MSC_VER) && !defined(_DEBUG)
  output = _byteswap_ulong(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap32(output);
#else
  uint32_t Byte0 = output & 0x000000FF;
  uint32_t Byte1 = output & 0x0000FF00;
  uint32_t Byte2 = output & 0x00FF0000;
  uint32_t Byte3 = output & 0xFF000000;
  output = (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
#endif
  memcpy(ptr, &output, sizeof(uint32_t));
}

static inline void swapBytes64(void* ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint64_t output;
  memcpy(&output, ptr, sizeof(uint64_t));
#if defined(_MSC_VER)
  output = _byteswap_uint64(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap64(output);
#else
  uint64_t Byte0 = output & 0x00000000000000FF;
  uint64_t Byte1 = output & 0x000000000000FF00;
  uint64_t Byte2 = output & 0x0000000000FF0000;
  uint64_t Byte3 = output & 0x00000000FF000000;
  uint64_t Byte4 = output & 0x000000FF00000000;
  uint64_t Byte5 = output & 0x0000FF0000000000;
  uint64_t Byte6 = output & 0x00FF000000000000;
  uint64_t Byte7 = output & 0xFF00000000000000;
  output = (Byte0 << (7 * 8)) | (Byte1 << (5 * 8)) | (Byte2 << (3 * 8)) |
      (Byte3 << (1 * 8)) | (Byte7 >> (7 * 8)) | (Byte6 >> (5 * 8)) |
      (Byte5 >> (3 * 8)) | (Byte4 >> (1 * 8));
#endif
  memcpy(ptr, &output, sizeof(uint64_t));
}

} // anonymous namespace

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
TORCH_API void THP_encodeComplexloatBuffer(
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
