#include <torch/nn/options/conv.h>

namespace torch {
namespace nn {

template struct ConvOptions<1>;
template struct ConvOptions<2>;
template struct ConvOptions<3>;

namespace functional {

template struct ConvFuncOptions<1>;
template struct ConvFuncOptions<2>;
template struct ConvFuncOptions<3>;

template struct ConvTransposeFuncOptions<1>;
template struct ConvTransposeFuncOptions<2>;
template struct ConvTransposeFuncOptions<3>;

} // namespace functional

} // namespace nn
} // namespace torch
