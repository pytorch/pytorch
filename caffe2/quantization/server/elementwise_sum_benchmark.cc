#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "utility_dnnlowp_ops.h"

using namespace std;

int main(int argc, const char* argv[]) {
  int LEN = argc > 1 ? atoi(argv[1]) : 65536;

  vector<uint8_t> a(LEN), b(LEN), c_avx2(LEN), c_avx512(LEN);
  for (int i = 0; i < LEN; ++i) {
    a[i] = i % 256;
    b[i] = (i * 2) % 256;
  }

  chrono::time_point<chrono::system_clock> t = chrono::system_clock::now();
  caffe2::internal::ElementWiseSumAVX2<uint8_t, false>(
      a.data(),
      b.data(),
      c_avx2.data(),
      a.size(),
      1.0f,
      11,
      2.0f,
      22,
      3.0f,
      33);
  double dt = chrono::duration<double>(chrono::system_clock::now() - t).count();
  double bytes = 3. * LEN * sizeof(a[0]);
  cout << bytes / dt / 1e9 << " GB/s" << endl;

  return 0;
}
