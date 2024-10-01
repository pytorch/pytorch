#include <torch/jit.h>

#include <ATen/core/stack.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>

#include <memory>
#include <string>

namespace torch::jit {

std::shared_ptr<CompilationUnit> compile(const std::string& source) {
  auto module = std::make_shared<CompilationUnit>();
  module->define(std::nullopt, source, nativeResolver(), nullptr);
  return module;
}

} // namespace torch::jit
