#include <torch/nn/options/activation.h>

namespace torch {
namespace nn {

SELUOptions::SELUOptions(bool inplace) : inplace_(inplace) {}

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

SoftmaxOptions::SoftmaxOptions(int64_t dim) : dim_(dim) {}

SoftminOptions::SoftminOptions(int64_t dim) : dim_(dim) {}

LogSoftmaxOptions::LogSoftmaxOptions(int64_t dim) : dim_(dim) {}

ReLUOptions::ReLUOptions(bool inplace) : inplace_(inplace) {}

ReLU6Options::ReLU6Options(bool inplace) : inplace_(inplace) {}

SoftshrinkOptions::SoftshrinkOptions(double lambda) : lambda_(lambda) {}

namespace functional {

SoftmaxFuncOptions::SoftmaxFuncOptions(int64_t dim) : dim_(dim) {}

SoftminFuncOptions::SoftminFuncOptions(int64_t dim) : dim_(dim) {}

LogSoftmaxFuncOptions::LogSoftmaxFuncOptions(int64_t dim) : dim_(dim) {}

} // namespace functional

} // namespace nn
} // namespace torch
