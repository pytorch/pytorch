/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// pragma once
#include <cmath>

namespace {
struct point {
  int x;
  int y;
};

struct tile_descriptor {
  point tile_dims;
  point tile_size;
  int tiles;
};
} // namespace

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
