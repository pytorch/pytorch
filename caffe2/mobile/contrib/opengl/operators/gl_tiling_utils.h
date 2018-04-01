#pragma once
#include <cmath>

struct point {
  int x;
  int y;
};

struct tile_descriptor {
  point tile_dims;
  point tile_size;
  int tiles;
};

namespace caffe2 {
inline static void squareFactors(int N, int& r1, int& r2) {
  int f = sqrt(N);

  if (f * f == N) {
    r1 = r2 = f;
  } else {
    while (N % f != 0) {
      f--;
    }
    r1 = N / f;
    r2 = f;
  }
}

inline static void computeOutputTiles(int output_channels, int& output_tile_x, int& output_tile_y) {
  squareFactors((output_channels + 3) / 4, output_tile_x, output_tile_y);
}
} // namespace caffe2
