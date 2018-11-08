// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "common_fpga.h"
#include "context.h"

namespace caffe2 {
class FPGAContextSingleton : public BaseSingletonContext {
 public:
  FPGAContextSingleton(const string& devname, int n_queues);
  void LoadProgram();
  static FPGAContextSingleton& getInstance();

 public:
  std::map<std::string, cl::Program> program_catalog;
  std::map<std::string, cl::Kernel> kernel_catalog;

  static const int MAT_A_BLOCK_HEIGHT = 512;
  static const int MAT_A_BLOCK_WIDTH = 512;
  static const int MAT_B_BLOCK_HEIGHT = 512;
  static const int MAT_B_BLOCK_WIDTH = 512;
  static const int MAT_C_BLOCK_HEIGHT = 512;
  static const int MAT_C_BLOCK_WIDTH = 512;

  static const int VEC = 16;
  static const int DOT_PROD_VECTOR_SIZE = VEC;
  static const int VECTORS_AT_A_TIME = 32;

  static const int COLUMNS_INTERLEAVED = 32;
  static const int ROWS_INTERLEAVED = 32;

  static const int PE_COLS = 16;
  static const int PE_ROWS = 16;

  static const int ACCUM_SHIFT_REG_SIZE =
      COLUMNS_INTERLEAVED * ROWS_INTERLEAVED;

  // C = A * B
  // Optionally transpose A or B
  // if reluA is true, relu(A) * B -> C
  // if revreluA is true, revrelu(A, dA) * B -> C
  // revRelu(A, dA) = A if dA > 0
  // A is of size (ha, wa)
  // B is of size (wa, wb)
  bool MatMul(
      const bool TransA,
      const bool TransB,
      const float* A,
      const float* dA,
      const float* B,
      float* C,
      const int ha,
      const int wa,
      const int wb,
      const bool reluA,
      const bool revreluA);
  // C += A * B
  // Optionally transpose A or B
  // A is of size (ha, wa)
  // B is of size (wa, wb)
  bool MatMulAccum(
      const bool TransA,
      const bool TransB,
      const float* A,
      const float* B,
      float* C,
      const int ha,
      const int wa,
      const int wb);
  // y = A * x
  // Optionally transpose A
  // A is of size (ha, wa)
  bool MatVecMul(
      const bool TransA,
      const float* A,
      const float* x,
      float* y,
      const int ha,
      const int wa);

  // A_r = relu(A)
  // relu(A) = A if A > 0 else 0
  bool ReLU(const float* A, float* A_r, int ha, int wa);

  // C = revrelu(A, dA)
  // revRelu(A, dA) = A if dA > 0
  bool ReLUGrad(const float* A, const float* dA, float* C, int ha, int wa);
  bfloat16* readBuffer(cl::Buffer* buff, int h, int w);
  void
  writeBuffer(const bfloat16* data, const int h, const int w, cl::Buffer* buff);
  cl::Buffer* createIdentity(int n);
  cl::Buffer* transposeBuffer(cl::Buffer* buff, int h, int w);
  void printBuffer(cl::Buffer* buff, int h, int w, int ph, int pw);

  static std::mutex& mutex();

 private:
  cl::Buffer* copyToTileSizeBuffer(
      const float* A,
      const bool trans,
      int ha,
      int wa,
      int* newha,
      int* newwa);
  void copyBuffer(cl::Buffer* src, cl::Buffer* dst, size_t nbytes);

  void copyFromTileSizeBuffer(
      cl::Buffer* src,
      int srch,
      int srcw,
      const float* A,
      int ha,
      int wa);
};

} // namespace caffe2
