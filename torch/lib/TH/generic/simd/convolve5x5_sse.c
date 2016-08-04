#include <smmintrin.h>
#include "common_simd.h"


/* SSE variants */
void convolve_5x5_1_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_1()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS(1, i)
  }
  for (; i < (count); i++) {
    float output0 = output[i + outputStride * 0];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
  }
}

void convolve_5x5_2_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_2()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS(2, i)
  }
  for (; i < (count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
  }
}

void convolve_5x5_4_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_4()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS(4, i)
  }
  for (; i < (count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    float output2 = output[i + outputStride * 2];
    float output3 = output[i + outputStride * 3];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
        output2 += weight[5 * row + col] * image[i + (row + 2) * inputStride + col];
        output3 += weight[5 * row + col] * image[i + (row + 3) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
    output[i + outputStride * 2] = output2;
    output[i + outputStride * 3] = output3;
  }
}

void convolve_5x5_6_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_6()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS(6, i)
  }
  for (; i<(count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    float output2 = output[i + outputStride * 2];
    float output3 = output[i + outputStride * 3];
    float output4 = output[i + outputStride * 4];
    float output5 = output[i + outputStride * 5];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
        output2 += weight[5 * row + col] * image[i + (row + 2) * inputStride + col];
        output3 += weight[5 * row + col] * image[i + (row + 3) * inputStride + col];
        output4 += weight[5 * row + col] * image[i + (row + 4) * inputStride + col];
        output5 += weight[5 * row + col] * image[i + (row + 5) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
    output[i + outputStride * 2] = output2;
    output[i + outputStride * 3] = output3;
    output[i + outputStride * 4] = output4;
    output[i + outputStride * 5] = output5;
  }
}

void convolve_5x5_8_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_8()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS(8, i)
  }
  for (; i<(count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    float output2 = output[i + outputStride * 2];
    float output3 = output[i + outputStride * 3];
    float output4 = output[i + outputStride * 4];
    float output5 = output[i + outputStride * 5];
    float output6 = output[i + outputStride * 6];
    float output7 = output[i + outputStride * 7];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
        output2 += weight[5 * row + col] * image[i + (row + 2) * inputStride + col];
        output3 += weight[5 * row + col] * image[i + (row + 3) * inputStride + col];
        output4 += weight[5 * row + col] * image[i + (row + 4) * inputStride + col];
        output5 += weight[5 * row + col] * image[i + (row + 5) * inputStride + col];
        output6 += weight[5 * row + col] * image[i + (row + 6) * inputStride + col];
        output7 += weight[5 * row + col] * image[i + (row + 7) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
    output[i + outputStride * 2] = output2;
    output[i + outputStride * 3] = output3;
    output[i + outputStride * 4] = output4;
    output[i + outputStride * 5] = output5;
    output[i + outputStride * 6] = output6;
    output[i + outputStride * 7] = output7;
  }
}

#define UNROLL_SSE_CONVOLUTION 0
#if (UNROLL_SSE_CONVOLUTION)

void convolve_5x5_64x64_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 60; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_4COLS_XROWS(6, 0)
    CONVOLVE_4COLS_XROWS(6, 4)
    CONVOLVE_4COLS_XROWS(6, 8)
    CONVOLVE_4COLS_XROWS(6, 12)
    CONVOLVE_4COLS_XROWS(6, 16)
    CONVOLVE_4COLS_XROWS(6, 20)
    CONVOLVE_4COLS_XROWS(6, 24)
    CONVOLVE_4COLS_XROWS(6, 28)
    CONVOLVE_4COLS_XROWS(6, 32)
    CONVOLVE_4COLS_XROWS(6, 36)
    CONVOLVE_4COLS_XROWS(6, 40)
    CONVOLVE_4COLS_XROWS(6, 44)
    CONVOLVE_4COLS_XROWS(6, 48)
    CONVOLVE_4COLS_XROWS(6, 52)
    CONVOLVE_4COLS_XROWS(6, 56)
    CONVOLVE_4COLS_XROWS(6, 60)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_4()
  CONVOLVE_4COLS_XROWS(4, 0)
  CONVOLVE_4COLS_XROWS(4, 4)
  CONVOLVE_4COLS_XROWS(4, 8)
  CONVOLVE_4COLS_XROWS(4, 12)
  CONVOLVE_4COLS_XROWS(4, 16)
  CONVOLVE_4COLS_XROWS(4, 20)
  CONVOLVE_4COLS_XROWS(4, 24)
  CONVOLVE_4COLS_XROWS(4, 28)
  CONVOLVE_4COLS_XROWS(4, 32)
  CONVOLVE_4COLS_XROWS(4, 36)
  CONVOLVE_4COLS_XROWS(4, 40)
  CONVOLVE_4COLS_XROWS(4, 44)
  CONVOLVE_4COLS_XROWS(4, 48)
  CONVOLVE_4COLS_XROWS(4, 52)
  CONVOLVE_4COLS_XROWS(4, 56)
  CONVOLVE_4COLS_XROWS(4, 60)
}

void convolve_5x5_32x32_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 30; i+=6)
  {
    DECLARE_OUTPUT_6()

      CONVOLVE_4COLS_XROWS(6, 0)
      CONVOLVE_4COLS_XROWS(6, 4)
      CONVOLVE_4COLS_XROWS(6, 8)
      CONVOLVE_4COLS_XROWS(6, 12)
      CONVOLVE_4COLS_XROWS(6, 16)
      CONVOLVE_4COLS_XROWS(6, 20)
      CONVOLVE_4COLS_XROWS(6, 24)
      CONVOLVE_4COLS_XROWS(6, 28)

    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_2()
  CONVOLVE_4COLS_XROWS(2, 0)
  CONVOLVE_4COLS_XROWS(2, 4)
  CONVOLVE_4COLS_XROWS(2, 8)
  CONVOLVE_4COLS_XROWS(2, 12)
  CONVOLVE_4COLS_XROWS(2, 16)
  CONVOLVE_4COLS_XROWS(2, 20)
  CONVOLVE_4COLS_XROWS(2, 24)
  CONVOLVE_4COLS_XROWS(2, 28)
}

void convolve_5x5_16x16_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 12; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_4COLS_XROWS(6, 0)
    CONVOLVE_4COLS_XROWS(6, 4)
    CONVOLVE_4COLS_XROWS(6, 8)
    CONVOLVE_4COLS_XROWS(6, 12)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_4()
  CONVOLVE_4COLS_XROWS(4, 0)
  CONVOLVE_4COLS_XROWS(4, 4)
  CONVOLVE_4COLS_XROWS(4, 8)
  CONVOLVE_4COLS_XROWS(4, 12)
}

void convolve_5x5_8x8_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  DECLARE_OUTPUT_8()
  CONVOLVE_4COLS_XROWS(8, 0)
  CONVOLVE_4COLS_XROWS(8, 4)
}

#endif

void convolve_5x5_sse(float* output, float* input, float* kernel, long outRows, long outCols, long outStride, long inCols) {
  long yy = 0;
  float* t_ = input;
  float* r_ = output;
  float* k_ = kernel;
#if (UNROLL_SSE_CONVOLUTION)
  if((outRows == 64) && (outCols == 64)) {
    convolve_5x5_64x64_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 32) && (outCols == 32)) {
    convolve_5x5_32x32_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 16) && (outCols == 16)) {
    convolve_5x5_16x16_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 8) && (outCols == 8)) {
    convolve_5x5_8x8_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }
#endif
  for(; yy < (outRows / 6 ) * 6; yy += 6) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_6_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 6);
  }
  // more than 2 rows left to process and we ended up on a non-multiple of 4
  if((yy < (outRows & 0xFFFFFFFE)) && ((yy % 4) != 0)) {
    // process 2 rows to align on the next multiple of 4 rows (because we were a multiple of 6 after the previous loop)
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_2_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 2);
    yy += 2;
  }

  for(; yy < (outRows & 0xFFFFFFFC); yy += 4) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_4_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 4);
  }

  for(; yy < (outRows & 0xFFFFFFFE); yy += 2) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_2_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 2);
  }

  for(; yy < outRows; yy += 1) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_5x5_1_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 1);
  }
}
