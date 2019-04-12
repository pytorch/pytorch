#include <torch/jit.h>

#include <torch/csrc/jit/script/compiler.h>
#include <ATen/core/stack.h>

#include <memory>
#include <string>

namespace torch {
namespace jit {

std::shared_ptr<script::CompilationUnit> compile(const std::string& source) {
  auto module = std::make_shared<script::CompilationUnit>();
  module->define(source, script::nativeResolver, nullptr);
  return module;
}

} // namespace jit
} // namespace torch
