#pragma once

#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/stack.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <string>
#include <memory>

namespace torch {
namespace jit {

/// Compiles script code into an executable graph.
///
/// Takes a string containing functions in script syntax and compiles them into
/// a module (graph). The returned module provides a `run_method` function
/// that may be used to invoke the compiled functions.
///
/// For example:
/// \rst
/// .. code-block:: cpp
///
///   auto module = torch::jit::compile(R"JIT(
///     def relu_script(a, b):
///       return torch.relu(a + b)
///     def test_while(a, i):
///       while i < 10:
///         a += a
///         i += 1
///       return a
///   )JIT");
///   IValue output = module->run_method("relu_script", a, b);
/// \endrst
TORCH_API std::shared_ptr<script::Module> compile(const std::string& source);

} // namespace jit
} // namespace torch
