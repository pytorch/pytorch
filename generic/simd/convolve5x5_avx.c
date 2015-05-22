#include <immintrin.h>
#include "common_simd.h"

#define CLEAR_AVX() _mm256_zeroupper()

void convolve_5x5_1_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_1()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS(1, i)
  }
}

void convolve_5x5_2_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_2()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS(2, i)
  }
}

void convolve_5x5_4_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_4()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS(4, i)
  }
}

void convolve_5x5_5_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_5()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS(5, i)
  }
}

void convolve_5x5_6_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_6()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS(6, i)
  }
}

void convolve_5x5_7_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_7()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS(7, i)
  }
}

void convolve_5x5_8_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_8()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS(8, i)
  }
}

void convolve_5x5_64x64_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 60; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_8COLS_XROWS(6, 0)
    CONVOLVE_8COLS_XROWS(6, 8)
    CONVOLVE_8COLS_XROWS(6, 16)
    CONVOLVE_8COLS_XROWS(6, 24)
    CONVOLVE_8COLS_XROWS(6, 32)
    CONVOLVE_8COLS_XROWS(6, 40)
    CONVOLVE_8COLS_XROWS(6, 48)
    CONVOLVE_8COLS_XROWS(6, 56)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_4()
  CONVOLVE_8COLS_XROWS(4, 0)
  CONVOLVE_8COLS_XROWS(4, 8)
  CONVOLVE_8COLS_XROWS(4, 16)
  CONVOLVE_8COLS_XROWS(4, 24)
  CONVOLVE_8COLS_XROWS(4, 32)
  CONVOLVE_8COLS_XROWS(4, 40)
  CONVOLVE_8COLS_XROWS(4, 48)
  CONVOLVE_8COLS_XROWS(4, 56)
}

void convolve_5x5_32x32_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 30; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_8COLS_XROWS(6, 0)
    CONVOLVE_8COLS_XROWS(6, 8)
    CONVOLVE_8COLS_XROWS(6, 16)
    CONVOLVE_8COLS_XROWS(6, 24)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_2()
  CONVOLVE_8COLS_XROWS(2, 0)
  CONVOLVE_8COLS_XROWS(2, 8)
  CONVOLVE_8COLS_XROWS(2, 16)
  CONVOLVE_8COLS_XROWS(2, 24)
}

void convolve_5x5_16x16_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 12; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_8COLS_XROWS(6, 0)
    CONVOLVE_8COLS_XROWS(6, 8)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_4()
  CONVOLVE_8COLS_XROWS(4, 0)
  CONVOLVE_8COLS_XROWS(4, 8)
}

void convolve_5x5_8x8_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  DECLARE_OUTPUT_8()
  CONVOLVE_8COLS_XROWS(8, 0)
}

void convolve_5x5_sse(float* output, float* input, float* kernel, long outRows, long outCols, long outStride, long inCols);

void convolve_5x5_avx(float* output, float* input, float* kernel, long outRows, long outCols, long outStride, long inCols) {
  long ic = inCols;
  long yy = 0;
  float* t_ = input;
  float* r_ = output;
  float* k_ = kernel;

  if((outRows == 64) && (outCols == 64)) {
    convolve_5x5_64x64_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 32) && (outCols == 32)) {
    convolve_5x5_32x32_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 16) && (outCols == 16)) {
    convolve_5x5_16x16_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 8) && (outCols == 8)) {
    convolve_5x5_8x8_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  for(; yy < (outRows / 6 ) * 6; yy += 6) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_6_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 6);
  }

  // more than 2 rows left to process and we ended up on a non-multiple of 4
  if((yy < (outRows & 0xFFFFFFFE)) && ((yy % 4) != 0)) {
    // process 2 rows to align on the next multiple of 4 rows (because we were a multiple of 6 after the previous loop)
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_2_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 2);
    yy += 2;
  }

  for(; yy < (outRows & 0xFFFFFFFC); yy += 4) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_4_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 4);
  }

  for(; yy < (outRows & 0xFFFFFFFE); yy += 2) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_2_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 2);
  }

  for(; yy < outRows; yy += 1) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_1_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 1);
  }

  long procCols = outCols & 0xFFFFFFF8; // avx version processes 8 cols at a time
  long remCols = outCols - procCols;

  //process the rest using sse
  if( remCols > 0) {
    CLEAR_AVX();
    convolve_5x5_sse(&output[procCols], &input[procCols], kernel, outRows, remCols, outStride, inCols);
  }
}