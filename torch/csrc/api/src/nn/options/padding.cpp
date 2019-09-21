#include <torch/nn/options/padding.h>

namespace torch {
namespace nn {

template struct ReflectionPadOptions<1>;
template struct ReflectionPadOptions<2>;

template struct ReplicationPadOptions<1>;
template struct ReplicationPadOptions<2>;
template struct ReplicationPadOptions<3>;

template struct ConstantPadOptions<1>;
template struct ConstantPadOptions<2>;
template struct ConstantPadOptions<3>;

} // namespace nn
} // namespace torch
