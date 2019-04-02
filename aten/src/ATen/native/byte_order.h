#ifndef THP_BYTE_ORDER_H
#define THP_BYTE_ORDER_H

#include <cstdint>
#include <cstddef>
#include <THHalf.h>
#include <iostream>

namespace at {
namespace native {
enum THPByteOrder {
  THP_LITTLE_ENDIAN = 0,
  THP_BIG_ENDIAN = 1
};

THPByteOrder THP_nativeByteOrder();

void THP_decodeBuffer(uint8_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBuffer(int16_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBuffer(int32_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBuffer(int64_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBuffer(THHalf* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBuffer(float* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBuffer(double* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBuffer(bool* dst, const uint8_t* src, THPByteOrder order, size_t len);

template <typename scalar_t>
 void THP_decodeBuffer(scalar_t* dst, const uint8_t* src, THPByteOrder order, size_t len) {
   std::cout << __PRETTY_FUNCTION__ << "\n";
   std::cout << typeid(dst).name() << '\n';
   std::cout << typeid(src).name() << '\n';
   std::cout << typeid(order).name() << '\n';
   // THP_decodeBuffer(dst, src, order, len);
 }

void THP_decodeInt16Buffer(int16_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeInt32Buffer(int32_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeInt64Buffer(int64_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeHalfBuffer(THHalf* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeFloatBuffer(float* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeDoubleBuffer(double* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeBoolBuffer(bool* dst, const uint8_t* src, THPByteOrder order, size_t len);

void THP_encodeInt16Buffer(uint8_t* dst, const int16_t* src, THPByteOrder order, size_t len);
void THP_encodeInt32Buffer(uint8_t* dst, const int32_t* src, THPByteOrder order, size_t len);
void THP_encodeInt64Buffer(uint8_t* dst, const int64_t* src, THPByteOrder order, size_t len);
void THP_encodeFloatBuffer(uint8_t* dst, const float* src, THPByteOrder order, size_t len);
void THP_encodeDoubleBuffer(uint8_t* dst, const double* src, THPByteOrder order, size_t len);
}
}

#endif
