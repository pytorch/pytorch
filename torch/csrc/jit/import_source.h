#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace script {
// Helpers to define modules and classes from source strings. Used in model
// saving/loading, so it expects the format produced by the model exporter.

// Define the list of classes in `src`.
TORCH_API void import_libs(
    // The compilation unit that will own the imported libs
    std::shared_ptr<CompilationUnit> cu,
    // Qualifier for any classes that `src` defines. Looks like a module path,
    // like "foo.bar.baz"
    const std::string& class_qualifier,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& tensor_table,
    // Callback to import any dependencies of this source before compiling
    const std::function<void(const std::string&)>& import_callback);

// Add the methods defined in `src` to the module `mod`.
TORCH_API void LEGACY_import_methods(
    const script::Module& mod,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    // Callback to import any dependencies of this source before compiling
    const std::function<void(const std::string&)>& import_callback);
} // namespace script
} // namespace jit
} // namespace torch
