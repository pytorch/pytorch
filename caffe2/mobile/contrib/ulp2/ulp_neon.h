#pragma once

#include "ulp.h"

namespace caffe2 {

constexpr size_t kGEMMTileSize = 64;
constexpr size_t kGEMMTileDepthBytes = 16;

bool run2b1bConvNeon(QConvState* state, const ConvArgs& args, const TensorCPU& X, TensorCPU* Y);
}
