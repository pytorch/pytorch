#pragma once
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/stack.h>

#include <string>

namespace torch {
namespace jit {

/// Compiles Python JIT code into a graph that can be executed.
///
/// For example:
/// @code
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
/// @endcode
///
/// @param source A string containing functions containing script code to
/// compile
/// @return A module containing the compiled functions
std::shared_ptr<script::Module> compile(const std::string& source);

} // namespace jit
} // namespace torch
