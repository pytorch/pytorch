#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch {
namespace jit {

TORCH_API void import_methods(const std::shared_ptr<script::Module>& mod, const std::string& src, const std::vector<at::Tensor>& constant_table);

} // namespace jit
} // namespace torch
