#pragma once

#include <map>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

enum class ModuleLintCode
{
    ML_BUNDLED_INPUT,
};

TORCH_API std::map<std::string, std::string> GenerateModuleLints(const script::Module& module);

} // namespace jit
} // namespace torch
