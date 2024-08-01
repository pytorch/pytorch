#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {

DropoutOptions::DropoutOptions(double p) : p_(p) {}

} // namespace nn
} // namespace torch
