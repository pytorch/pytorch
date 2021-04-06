#include <torch/csrc/jit/mobile/module.h>

#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <exception>

#include <ATen/record_function.h>

namespace torch {
namespace jit {
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {

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

Method Module::get_method(const std::string& name) const {
  if (auto method = find_method(name)) {
    return *method;
  }
  AT_ERROR("Method '", name, "' is not defined.");
}

c10::optional<Method> Module::find_method(const std::string& basename) const {
  for (auto& fn : cu_->methods()) {
    if (fn->name() == basename) {
      return c10::make_optional<Method>(Method(this, fn.get()));
    }
  }
  return c10::nullopt;
}

namespace {
void set_train_recurse(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj,
    bool on) {
  if (auto slot = obj->type()->findAttributeSlot("training")) {
    obj->setSlot(*slot, on);
  } else {
    TORCH_INTERNAL_ASSERT(false, "'training' attribute not found");
  }
  for (const auto& slot : obj->slots()) {
    if (slot.isObject()) {
      set_train_recurse(slot.toObject(), on);
    }
  }
}

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

void slot_named_params_recurse(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj,
    std::map<std::string, at::Tensor>* params,
    const std::string& parent_name) {
  auto slots = obj->slots();
  size_t nslots = slots.size();
  for (size_t i = 0; i < nslots; ++i) {
    auto slot = slots[i];
    std::string name =
        parent_name.size() == 0 ? parent_name : parent_name + ".";
    name += obj->type()->getAttributeName(i);
    if (slot.isTensor()) {
      (*params)[name] = slot.toTensor();
    } else if (slot.isObject()) {
      slot_named_params_recurse(slot.toObject(), params, name);
    }
  }
}
} // namespace

const std::vector<at::Tensor> Module::parameters() const {
  std::vector<at::Tensor> params;
  slot_params_recurse(object_, &params);
  return params;
}

const std::map<std::string, at::Tensor> Module::named_parameters() const {
  std::map<std::string, at::Tensor> params;
  const std::string name = "";
  slot_named_params_recurse(object_, &params, name);
  return params;
}

std::string Module::get_forward_method_debug_info(size_t pc) const {
  return find_method("forward")->get_module_debug_info(pc);
}

void Module::train(bool on) {
  set_train_recurse(object_, on);
}

bool Module::is_training() const {
  if (auto slot = object_->type()->findAttributeSlot("training")) {
    return object_->getSlot(*slot).toBool();
  }
  return true;
}

const std::vector<Method> Module::get_methods() const {
  std::vector<Method> methods;
  for (std::unique_ptr<Function>& fn : cu_->methods()) {
    methods.emplace_back(this, fn.get());
  }
  return methods;
}

Method::Method(const Module* owner, Function* function)
    : owner_(owner), function_(function) {}

void Method::run(Stack& stack) const {
  auto observer = torch::observerConfig().getModuleObserver();
  auto instance_key = std::rand();
  /* if the metadata dict doesn't contain "model_name", copy the metadata and
  set the value of "model_name" as name() */
  std::unordered_map<std::string, std::string> copied_metadata =
      owner_->metadata();
  if (owner_->metadata().find("model_name") == owner_->metadata().end()) {
    copied_metadata["model_name"] = owner_->name();
  }
  if (observer) {
    observer->onEnterRunMethod(
        copied_metadata, instance_key, function_->name());
  }

  auto debug_info = std::make_shared<MobileDebugInfo>();
  std::string name = copied_metadata["model_name"];
  debug_info->setModelName(name);
  debug_info->setMethodName(function_->name());
  at::DebugInfoGuard guard(at::DebugInfoKind::MOBILE_RUNTIME_INFO, debug_info);

  try {
    stack.insert(stack.begin(), owner_->_ivalue()); // self
    function_->run(stack);
    if (observer) {
      observer->onExitRunMethod(instance_key);
    }
  } catch (c10::Error& error) {
    if (observer) {
      observer->onFailRunMethod(instance_key, error.what());
    }
    TORCH_RETHROW(error);
  } catch (...) {
    auto currentException = std::current_exception();
    try {
      if (!currentException) {
        TORCH_CHECK(false, "Unknown exception");
      } else {
        try {
          std::rethrow_exception(currentException);
        } catch (const std::exception& e) {
          TORCH_CHECK(false, e.what());
        }
      }
    } catch (c10::Error& error) {
      if (observer) {
        observer->onFailRunMethod(instance_key, error.what());
      }
      TORCH_RETHROW(error);
    }
  }
}

c10::IValue Method::operator()(std::vector<c10::IValue> stack) const {
  run(stack);
  TORCH_INTERNAL_ASSERT(!stack.empty());
  return stack.front();
}

} // namespace mobile
} // namespace jit
} // namespace torch
