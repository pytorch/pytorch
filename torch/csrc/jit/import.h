#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/module.h"

namespace torch { namespace jit {

TORCH_API void ImportIRModule(
    const std::shared_ptr<script::Module> module,
    const std::string& filename);

TORCH_API std::shared_ptr<script::Module> load(const std::string& filename);

}}
