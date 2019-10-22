#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {

DropoutOptions::DropoutOptions(double p, bool inplace) : p_(p), inplace_(inplace) {}

} // namespace nn
} // namespace torch
