#pragma once

#include <torch/csrc/jit/script/module.h>

#include <ostream>

namespace torch {
namespace jit {
namespace mobile {

TORCH_API void SaveBytecode(const script::Module& module, const std::string& filename);

}
} // namespace jit
} // namespace torch
