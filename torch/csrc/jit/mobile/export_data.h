#pragma once

#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {
namespace mobile {

TORCH_API void _save_data(const Module& module, std::ostream& out);

TORCH_API void _save_data(const Module& module, const std::string& filename);

} // namespace mobile

TORCH_API void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    std::ostream& out);

TORCH_API void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    const std::string& filename);

} // namespace jit
} // namespace torch
