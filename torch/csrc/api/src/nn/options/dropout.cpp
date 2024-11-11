#include <torch/nn/options/dropout.h>

namespace torch::nn {

DropoutOptions::DropoutOptions(double p) : p_(p) {}

} // namespace torch::nn
