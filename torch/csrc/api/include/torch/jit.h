#pragma once

#include <string>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/stack.h>

namespace torch {
namespace jit {

/// Compiles Python JIT code into a graph that can be executed.
///
/// For example:
/// ```
/// auto module = torch::jit::compile(R"JIT(
///   def relu_script(a, b):
///     return torch.relu(a + b)
///   def test_while(a, i):
///     while i < 10:
///       a += a
///       i += 1
///     return a
/// )JIT");
/// auto output = torch::jit::run(module, "relu_script", a, b);
/// auto output = torch::jit::run(module, "test_while", a, b);
/// ```
///
/// @param source A JIT string containing functions that are valid under the
///               scope of the script compiler
/// @return A module containing the compiled functions
std::shared_ptr<script::Module> compile(const std::string& source);

/// Run a method from a module and get a list of the returns.
///
/// For example:
/// ```
/// auto output = torch::jit::run(module, "relu_script", a, b);
/// std::cout << output[0].toTensor().toCLong() << std::endl;
/// ```
///
///
/// @param module A module containing the method `method_name` (see torch::jit::compile)
/// @param method_name The name of the method to run
/// @param args Arguments to be passed to the method
/// @return A vector of `IValue`s that contain the results of the method
template<typename... Types>
Stack run(std::shared_ptr<script::Module> module, const std::string& method_name, Types&... args) {
  // Marhsal arguments into stack of IValues
  Stack stack;
  std::initializer_list<int>{ (stack.push_back(IValue(args)), 0)... };

  // Interpret graph and run computation
  module->get_method(method_name).run(stack);
  return stack;
}

} // namespace jit
} // namespace torch
