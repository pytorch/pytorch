#include <torch/nn/options/padding.h>

namespace torch {
namespace nn {

PadOptions::PadOptions(std::vector<int64_t> pad) : pad_(std::move(pad)) {}

} // namespace nn
} // namespace torch
