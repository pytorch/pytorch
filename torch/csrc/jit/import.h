#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/module.h"

namespace torch {
namespace jit {

using ModuleLookup = std::function<std::shared_ptr<script::Module>(
    const std::vector<std::string>&)>;

TORCH_API void import_ir_module(
    ModuleLookup module_lookup,
    const std::string& filename);

/// Loads a serialized `script::Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `script::Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API std::shared_ptr<script::Module> load(const std::string& filename);

} // namespace jit
} // namespace torch
