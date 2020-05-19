#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/runtime/jit_exception.h>

#include <ATen/record_function.h>

namespace torch {
namespace jit {
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {

const c10::QualifiedName& Function::qualname() const {
  return name_;
}

const std::string& Function::name() const {
  return name_.name();
}

void CompilationUnit::register_function(std::unique_ptr<Function> fn) {
  methods_.emplace_back(std::move(fn));
}

Function* CompilationUnit::find_function(const c10::QualifiedName& qn) {
  for (auto& fn : methods_) {
    if (fn->qualname() == qn) {
      return fn.get();
    }
  }
  return nullptr;
}

c10::IValue Module::run_method(const std::string& method_name, Stack stack) {
  auto observer = torch::observerConfig().getModuleObserver();
  if (observer) {
    observer->onEnter(name(), method_name);
  }

  if (at::hasGlobalCallbacks()) {
    auto debug_info = std::make_shared<MobileDebugInfo>();
    debug_info->setModelName(name());
    debug_info->setMethodName(method_name);
    at::DebugInfoGuard guard(
        at::DebugInfoKind::MOBILE_RUNTIME_INFO, debug_info);
  }

  auto m = find_method(method_name);
  if (m == nullptr) {
    AT_ERROR("Method '", method_name, "' is not defined.");
  }
  stack.insert(stack.begin(), object_);
  m->run(stack);
  c10::IValue result = stack.front();

  if (observer) {
    observer->onExit();
  }
  return result;
}

Function* Module::find_method(const std::string& basename) const {
  for (auto& fn : cu_->methods()) {
    if (fn->name() == basename) {
      return fn.get();
    }
  }
  return nullptr;
}

namespace {
void slot_params_recurse(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj,
    std::vector<at::Tensor>* params) {
  for (const auto& slot : obj->slots()) {
    if (slot.isTensor()) {
      params->emplace_back(slot.toTensor());
    } else if (slot.isObject()) {
      slot_params_recurse(slot.toObject(), params);
    }
  }
}
} // namespace

const std::vector<at::Tensor> Module::parameters() const {
  std::vector<at::Tensor> params;
  slot_params_recurse(object_, &params);
  return params;
}
} // namespace mobile
} // namespace jit
} // namespace torch
