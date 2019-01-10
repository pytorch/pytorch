#ifndef THP_BYTE_ORDER_H
#define THP_BYTE_ORDER_H

#include <stdint.h>
#include <stddef.h>
#include <THHalf.h>

enum THPByteOrder {
  THP_LITTLE_ENDIAN = 0,
  THP_BIG_ENDIAN = 1
};

THPByteOrder THP_nativeByteOrder();

void THP_decodeInt16Buffer(int16_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeInt32Buffer(int32_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeInt64Buffer(int64_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeHalfBuffer(THHalf* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeFloatBuffer(float* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeDoubleBuffer(double* dst, const uint8_t* src, THPByteOrder order, size_t len);

void THP_encodeInt16Buffer(uint8_t* dst, const int16_t* src, THPByteOrder order, size_t len);
void THP_encodeInt32Buffer(uint8_t* dst, const int32_t* src, THPByteOrder order, size_t len);
void THP_encodeInt64Buffer(uint8_t* dst, const int64_t* src, THPByteOrder order, size_t len);
void THP_encodeFloatBuffer(uint8_t* dst, const float* src, THPByteOrder order, size_t len);
void THP_encodeDoubleBuffer(uint8_t* dst, const double* src, THPByteOrder order, size_t len);

#endif
