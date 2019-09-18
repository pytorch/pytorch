#include <torch/nn/options/linear.h>

namespace torch {
namespace nn {

LinearOptions::LinearOptions(int64_t in, int64_t out) : in_(in), out_(out) {}

} // namespace nn
} // namespace torch
