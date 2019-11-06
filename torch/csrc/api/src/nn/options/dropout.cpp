#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {

DropoutOptionsBase::DropoutOptionsBase(double p) : p_(p) {}

template struct DropoutOptionsBase<1>;
template struct DropoutOptionsBase<2>;
template struct DropoutOptionsBase<3>;

} // namespace nn
} // namespace torch
