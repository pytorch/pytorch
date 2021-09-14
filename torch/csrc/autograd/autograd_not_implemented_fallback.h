#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torch {
namespace autograd {

TORCH_API torch::CppFunction autogradNotImplementedFallback();

TORCH_API torch::CppFunction autogradNotImplementedInplaceOrViewFallback();


#define REGISTER_AUTOGRAD_NOT_IMPLEMENTED_FALLBACK(ns, op)       \
  TORCH_LIBRARY_IMPL(ns, Autograd, m) {                          \
    m.impl(op, autogradNotImplementedFallback());                \
  }                                                              \
  TORCH_LIBRARY_IMPL(ns, ADInplaceOrView, m) {                   \
    m.impl(op, autogradNotImplementedInplaceOrViewFallback());   \
  }

}} // namespace torch::autograd
