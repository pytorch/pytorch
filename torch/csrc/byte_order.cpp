#include "byte_order.h"

#include <string.h>

static inline uint16_t decodeUInt16LE(const uint8_t *data) {
  return (data[0]<<0) | (data[1]<<8);
}

static inline uint16_t decodeUInt16BE(const uint8_t *data) {
  return (data[1]<<0) | (data[0]<<8);
}

static inline uint32_t decodeUInt32LE(const uint8_t *data) {
  return (data[0]<<0) | (data[1]<<8) | (data[2]<<16) | (data[3]<<24);
}

static inline uint32_t decodeUInt32BE(const uint8_t *data) {
  return (data[3]<<0) | (data[2]<<8) | (data[1]<<16) | (data[0]<<24);
}

static inline uint64_t decodeUInt64LE(const uint8_t *data) {
  return (((uint64_t)data[0])<< 0) | (((uint64_t)data[1])<< 8) |
         (((uint64_t)data[2])<<16) | (((uint64_t)data[3])<<24) |
         (((uint64_t)data[4])<<32) | (((uint64_t)data[5])<<40) |
         (((uint64_t)data[6])<<48) | (((uint64_t)data[7])<<56);
}

static inline uint64_t decodeUInt64BE(const uint8_t *data) {
  return (((uint64_t)data[7])<< 0) | (((uint64_t)data[6])<< 8) |
         (((uint64_t)data[5])<<16) | (((uint64_t)data[4])<<24) |
         (((uint64_t)data[3])<<32) | (((uint64_t)data[2])<<40) |
         (((uint64_t)data[1])<<48) | (((uint64_t)data[0])<<56);
}

THPByteOrder THP_nativeByteOrder()
{
  uint32_t x = 1;
  return *(uint8_t*)&x ? THP_LITTLE_ENDIAN : THP_BIG_ENDIAN;
}

void THP_decodeInt16Buffer(int16_t* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for (size_t i = 0; i < len; i++) {
    dst[i] = (int16_t) (order == THP_BIG_ENDIAN ? decodeUInt16BE(src) : decodeUInt16LE(src));
    src += sizeof(int16_t);
  }
}

void THP_decodeInt32Buffer(int32_t* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for (size_t i = 0; i < len; i++) {
    dst[i] = (int32_t) (order == THP_BIG_ENDIAN ? decodeUInt32BE(src) : decodeUInt32LE(src));
    src += sizeof(int32_t);
  }
}

void THP_decodeInt64Buffer(int64_t* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for (size_t i = 0; i < len; i++) {
    dst[i] = (int64_t) (order == THP_BIG_ENDIAN ? decodeUInt64BE(src) : decodeUInt64LE(src));
    src += sizeof(int64_t);
  }
}

void THP_decodeHalfBuffer(THHalf* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for (size_t i = 0; i < len; i++) {
    union { uint16_t x; THHalf f; };
    x = (order == THP_BIG_ENDIAN ? decodeUInt16BE(src) : decodeUInt16LE(src));
    dst[i] = f;
    src += sizeof(uint16_t);
  }
}

void THP_decodeFloatBuffer(float* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for (size_t i = 0; i < len; i++) {
    union { uint32_t x; float f; };
    x = (order == THP_BIG_ENDIAN ? decodeUInt32BE(src) : decodeUInt32LE(src));
    dst[i] = f;
    src += sizeof(float);
  }
}

void THP_decodeDoubleBuffer(double* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for (size_t i = 0; i < len; i++) {
    union { uint64_t x; double d; };
    x = (order == THP_BIG_ENDIAN ? decodeUInt64BE(src) : decodeUInt64LE(src));
    dst[i] = d;
    src += sizeof(double);
  }
}

template<size_t size>
static void swapBytes(uint8_t *ptr)
{
  uint8_t tmp;
  for (size_t i = 0; i < size / 2; i++) {
    tmp = ptr[i];
    ptr[i] = ptr[size-i];
    ptr[size-i] = tmp;
  }
}


void THP_encodeInt16Buffer(uint8_t* dst, const int16_t* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(int16_t) * len);
  if (order != THP_nativeByteOrder()) {
    for (size_t i = 0; i < len; i++) {
      swapBytes<sizeof(int16_t)>(dst);
      dst += sizeof(int16_t);
    }
  }
}

void THP_encodeInt32Buffer(uint8_t* dst, const int32_t* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(int32_t) * len);
  if (order != THP_nativeByteOrder()) {
    for (size_t i = 0; i < len; i++) {
      swapBytes<sizeof(int32_t)>(dst);
      dst += sizeof(int32_t);
    }
  }
}

void THP_encodeInt64Buffer(uint8_t* dst, const int64_t* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(int64_t) * len);
  if (order != THP_nativeByteOrder()) {
    for (size_t i = 0; i < len; i++) {
      swapBytes<sizeof(int64_t)>(dst);
      dst += sizeof(int64_t);
    }
  }
}

void THP_encodeFloatBuffer(uint8_t* dst, const float* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(float) * len);
  if (order != THP_nativeByteOrder()) {
    for (size_t i = 0; i < len; i++) {
      swapBytes<sizeof(float)>(dst);
      dst += sizeof(float);
    }
  }
}

void THP_encodeDoubleBuffer(uint8_t* dst, const double* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(double) * len);
  if (order != THP_nativeByteOrder()) {
    for (size_t i = 0; i < len; i++) {
      swapBytes<sizeof(double)>(dst);
      dst += sizeof(double);
    }
  }
}
