#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torch {
namespace autograd {

TORCH_API torch::CppFunction autogradNotImplementedFallback();

TORCH_API torch::CppFunction autogradNotImplementedInplaceOrViewFallback();

}} // namespace torch::autograd
