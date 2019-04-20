#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {
namespace script {
// Helpers to define modules and classes from source strings. Used in model
// saving/loading, so it expects the format produced by the model exporter.

// Add the methods defined in `src` to the module `mod`.
TORCH_API void import_methods(
    const std::shared_ptr<script::Module>& mod,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table,
    const std::function<void(const std::string&)>& import_callback);

// Defined the list of classes in `src`.
TORCH_API void import_libs(
    // Qualifier for any classes that `src` defines. Looks like a module path,
    // like "foo.bar.baz"
    const std::string& class_qualifier,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table,
    const std::function<void(const std::string&)>& import_callback);

} // namespace script
} // namespace jit
} // namespace torch
