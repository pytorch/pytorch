#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "./fake_nnpi_ops_utils.h"

int main() {
  using namespace std;

  int m = 40;
  int n = 2048;
  int k = 2048;

  vector<uint8_t> a(m * k);
  vector<int8_t> b(k * n);
  vector<uint8_t> c(m * n);
  vector<int32_t> bias(n);

  int NUM_WARMUPS = 4;
  int NUM_ITERS = 16;
  chrono::time_point<chrono::high_resolution_clock> t_begin;
  for (int i = 0; i < NUM_WARMUPS + NUM_ITERS; ++i) {
    if (i == NUM_WARMUPS) {
      t_begin = chrono::high_resolution_clock::now();
    }
    caffe2::fake_nnpi::matmul_u8i8u8acc32_ref(
        m,
        n,
        k,
        k,
        k,
        n,
        a.data(),
        0,
        b.data(),
        0,
        bias.data(),
        c.data(),
        1.0f,
        0,
        false);
  }

  double dur = chrono::duration_cast<chrono::nanoseconds>(
                   chrono::high_resolution_clock::now() - t_begin)
                   .count() /
      1e9;
  cout << static_cast<double>(NUM_ITERS) * m * n * k * 2 / dur / 1e9 << " GF/s"
       << endl;

  return 0;
}
