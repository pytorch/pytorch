#include <torch/jit.h>

#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/stack.h>

#include <memory>
#include <string>

namespace torch {
namespace jit {

std::shared_ptr<script::Module> compile(const std::string& source) {
  auto module = std::make_shared<script::Module>();
  defineMethodsInModule(*module, source, script::nativeResolver, /*self=*/nullptr);
  return module;
}

} // namespace jit
} // namespace torch
