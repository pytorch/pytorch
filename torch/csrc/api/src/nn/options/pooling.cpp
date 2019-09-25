#include <torch/nn/options/pooling.h>

namespace torch {
namespace nn {

template struct AvgPoolOptions<1>;
template struct AvgPoolOptions<2>;
template struct AvgPoolOptions<3>;

template struct MaxPoolOptions<1>;
template struct MaxPoolOptions<2>;
template struct MaxPoolOptions<3>;

template struct AdaptiveMaxPoolOptions<1>;
template struct AdaptiveMaxPoolOptions<2>;
template struct AdaptiveMaxPoolOptions<3>;

template struct AdaptiveAvgPoolOptions<1>;

} // namespace nn
} // namespace torch
