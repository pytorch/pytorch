#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "fp16_fma.h"

using namespace std;
using namespace fake_fp16;

TEST(FP16_FMA, Simple) {
  int x = 1;
  x += 2;
  int N = 6;

  vector<float> A(N, 1.23);
  vector<float> B(N, 2.34);
  vector<float> C(N, 3.45);
  fma_fp16(N, A.data(), B.data(), C.data());

  for (int i = 0; i < N; i++) {
    LOG(INFO) << C[i] << " ";
    ASSERT_TRUE(abs(C[i] - 6.32812) < 1e-3);
  }
}

TEST(FP16_FMA, Comprehensive) {
#if 0
#pragma omp parallel num_threads(30)
  for (uint16_t a = 0; a < 1 << 15; a++) {
    for (uint16_t b = 0; b < 1 << 15; b++) {
      for (uint16_t c = 0; c < 1 << 15; c++) {
        uint16_t z = a + b * c;

        //       fake_fma_fp16_slow(A[0], B[0], C[0]);
      }
    }
  }

  fake_fma_fp16_slow(A[0], B[0], C[0]);
#endif
}
