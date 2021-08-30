#pragma once

#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {
namespace mobile {
void call_setup_methods();
void consume_tensor(at::Tensor& t);
std::set<std::string> get_operators();

} // namespace mobile
} // namespace jit
} // namespace torch
