#pragma once

#include <THHalf.h>
#include <c10/util/BFloat16.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <cstddef>
#include <cstdint>

namespace torch {
namespace utils {

enum THPByteOrder {
  THP_LITTLE_ENDIAN = 0,
  THP_BIG_ENDIAN = 1
};

TORCH_API THPByteOrder THP_nativeByteOrder();

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
    THHalf* dst,
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

} // namespace utils
} // namespace torch
