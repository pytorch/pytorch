#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {

DropoutOptions::DropoutOptions(double p) : p_(p) {}

template struct DropoutOptions<1>;
template struct DropoutOptions<2>;
template struct DropoutOptions<3>;

} // namespace nn
} // namespace torch
