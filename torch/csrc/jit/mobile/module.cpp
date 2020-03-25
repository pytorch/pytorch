#include "module.h"
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#if defined(PYTORCH_MOBILE_OBSERVER)
#include <torch/csrc/jit/mobile/observer.h>
#endif

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
#if defined(PYTORCH_MOBILE_OBSERVER)
  auto observer = torch::observerConfig().getModuleObserver();
  if (observer) {
    observer->onEnter(name(), method_name);
  }
#endif

#if defined(PYTORCH_MOBILE_OPERATOR_OBSERVER)
  auto debug_info = std::make_shared<MobileDebugInfo>();
  debug_info->setModelName(name());
  debug_info->setMethodName(method_name);
  at::setThreadLocalDebugInfo(debug_info);
#endif

  auto m = find_method(method_name);
  stack.insert(stack.begin(), object_);
  m->run(stack);
  c10::IValue result = stack.front();

#if defined(PYTORCH_MOBILE_OBSERVER)
  if (observer) {
    observer->onExit();
  }
#endif
  return result;
}

Function* Module::find_method(const std::string& basename) const {
  for (auto& fn : cu_->methods()) {
    if (fn->name() == basename) {
      return fn.get();
    }
  }
  AT_ERROR("Method '", basename, "' is not defined.");
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
