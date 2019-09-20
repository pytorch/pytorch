#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {

DropoutOptions::DropoutOptions(double rate) : rate_(rate) {}

} // namespace nn
} // namespace torch
