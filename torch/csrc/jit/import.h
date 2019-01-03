#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

#include <istream>

namespace torch {
namespace jit {

using ModuleLookup = std::function<std::shared_ptr<script::Module>(
    const std::vector<std::string>&)>;

TORCH_API void import_ir_module(
    ModuleLookup module_lookup,
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt);

TORCH_API void import_ir_module(
    ModuleLookup module_lookup,
    std::istream& in,
    c10::optional<c10::Device> device = c10::nullopt);

/// Loads a serialized `script::Module` from the given `istream`.
///
/// The istream must contain a serialized `script::Module`, exported via
/// `torch::jit::ExportModule` in C++.
TORCH_API std::shared_ptr<script::Module> load(
    std::istream& in,
    c10::optional<c10::Device> device = c10::nullopt);

/// Loads a serialized `script::Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `script::Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API std::shared_ptr<script::Module> load(
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt);

} // namespace jit
} // namespace torch
