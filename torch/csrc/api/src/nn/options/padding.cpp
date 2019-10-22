#include <torch/nn/options/padding.h>

namespace torch {
namespace nn {

PadOptions::PadOptions(std::vector<int64_t> pad) : pad_(std::move(pad)) {}

template struct ReflectionPadOptions<1>;
template struct ReflectionPadOptions<2>;

} // namespace nn
} // namespace torch
