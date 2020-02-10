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
template struct AdaptiveAvgPoolOptions<2>;
template struct AdaptiveAvgPoolOptions<3>;

template struct MaxUnpoolOptions<1>;
template struct MaxUnpoolOptions<2>;
template struct MaxUnpoolOptions<3>;

template struct LPPoolOptions<1>;
template struct LPPoolOptions<2>;

} // namespace nn
} // namespace torch
