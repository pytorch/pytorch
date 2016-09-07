#include <stdint.h>
#include <stddef.h>

enum THPByteOrder {
  THP_LITTLE_ENDIAN = 0,
  THP_BIG_ENDIAN = 1
};

THPByteOrder THP_nativeByteOrder();
void THP_decodeInt16Buffer(int16_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeInt32Buffer(int32_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeInt64Buffer(int64_t* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeFloatBuffer(float* dst, const uint8_t* src, THPByteOrder order, size_t len);
void THP_decodeDoubleBuffer(double* dst, const uint8_t* src, THPByteOrder order, size_t len);
