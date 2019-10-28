#include <torch/nn/options/padding.h>

namespace torch {
namespace nn {

PadOptions::PadOptions(std::vector<int64_t> pad) : pad_(std::move(pad)) {}

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
