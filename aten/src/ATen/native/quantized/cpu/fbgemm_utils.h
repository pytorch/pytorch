#pragma once

#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
#include "fbgemm/QuantUtils.h"

// The struct for the packed weight matrix (PackBMatrix) and the corresponding
// column offsets used for the fully connect layer, which are both prepared in
// the prepacking step to save the computations in the inference. Note the
// column offsets include the sum of the B columns as well as the scalar term
// B_zero_point * K, whereas the row offsets created by
// PackAWithQuantRowOffset/PackAWithIm2Col/PackAWithRowOffset are only the sum
// of the A rows. The column offsets are needed for the asymmetric quantization
// (affine quantization) of input matrix.
// Note that in JIT mode we can think of a way to fuse col_offsets with bias.
struct FBGEMM_API PackedFCWeight {
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> w;
  std::vector<int32_t> col_offsets;
  float w_scale;
  int w_zp;
};

// Convert the weight from uint8 to int8.
static void convert_uint8_int8(int K, int N, const uint8_t* src_uint8, int8_t* dst_int8) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < K; ++j) {
      dst_int8[i * K + j] =
          static_cast<int8_t>(static_cast<int32_t>(src_uint8[i * K + j]) - 128);
    }
  }
}

#endif // USE_FBGEMM
