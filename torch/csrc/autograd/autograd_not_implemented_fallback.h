#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torch {
namespace autograd {

TORCH_API torch::CppFunction autogradNotImplementedFallback();

}} // namespace torch::autograd
