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

// Add the methods defined in `src` to the module `mod`.
TORCH_API void import_methods(
    const script::Module& mod,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    // Callback to import any dependencies of this source before compiling
    const std::function<void(const std::string&)>& import_callback);

// Define the list of classes in `src`.
TORCH_API void import_libs(
    // The compilation unit that will own the imported libs
    std::shared_ptr<CompilationUnit> cu,
    // Qualifier for any classes that `src` defines. Looks like a module path,
    // like "foo.bar.baz"
    const std::string& class_qualifier,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    // Callback to import any dependencies of this source before compiling
    const std::function<void(const std::string&)>& import_callback);

// Add the functions defined in `src` to the compilation unit `cu`.
// self is passed through the CompilationUnit's define function.
// If present, it determines the SugaredValue for the first argument
// and that argument is no longer expected to have type annotations.
TORCH_API void import_functions(
    // Prefix to use when importing these functions in to the CU
    const c10::optional<c10::QualifiedName>& prefix,
    // CompilationoUnit to define the functions in.
    std::shared_ptr<CompilationUnit> cu,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    const Self* self = nullptr,
    const std::function<void(const std::string&)>& import_callback = nullptr);

} // namespace script
} // namespace jit
} // namespace torch
