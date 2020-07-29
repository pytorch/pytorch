#pragma once

#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {
namespace mobile {

TORCH_API void _save_parameters(const Module& module, std::ostream& out);

TORCH_API void _save_parameters(
    const Module& module,
    const std::string& filename);

} // namespace mobile
} // namespace jit
} // namespace torch
