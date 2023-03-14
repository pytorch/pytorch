#include <torch/csrc/jit/mobile/module.h>

#include <torch/csrc/jit/backends/backend_exception.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <exception>

#include <ATen/record_function.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>

namespace torch {
namespace jit {
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {

void CompilationUnit::register_function(std::unique_ptr<Function> fn) {
  methods_.emplace_back(std::move(fn));
}

const Function* CompilationUnit::find_function(
    const c10::QualifiedName& qn) const {
  for (auto& fn : methods_) {
    if (fn->qualname() == qn) {
      return fn.get();
    }
  }
  return nullptr;
}

Function* CompilationUnit::find_function(const c10::QualifiedName& qn) {
  // NOLINTNEXTLINE
  return const_cast<Function*>(
      static_cast<const CompilationUnit*>(this)->find_function(qn));
}

Method Module::get_method(const std::string& name) const {
  if (auto method = find_method(name)) {
    return *method;
  }
  AT_ERROR("Method '", name, "' is not defined.");
}

bool Module::compareMethodSchemas(
    const std::string& name_1,
    const std::string& name_2) {
  c10::optional<c10::FunctionSchema> schema_1, schema_2;
  for (const auto& fn : cu_->methods()) {
    if (fn->name() == name_1) {
      schema_1 = fn->getSchema();
    }
    if (fn->name() == name_2) {
      schema_2 = fn->getSchema();
    }
  }
  if (schema_1.has_value() && schema_2.has_value()) {
    return (schema_1 == schema_2);
  }
  return false;
}

void Module::unsafeRemoveMethod(const std::string& basename) {
  int64_t i = 0;
  for (; i < static_cast<int64_t>(cu_->methods().size()); ++i) {
    if ((cu_->methods()[i])->name() == basename) {
      break;
    }
  }
  object_->type()->unsafeRemoveMethod(basename);
  cu_->unsafeRemoveFunction(i);
}

void Module::unsafeCopyMethod(
    const std::string& new_method_name,
    const Function& to_be_copied) {
  TORCH_CHECK(
      !find_method(new_method_name).has_value(),
      "Trying to replace existing method.");
  const c10::QualifiedName& tobe_copied_name = to_be_copied.qualname();
  c10::QualifiedName qualified_method_name(
      tobe_copied_name.prefix(), new_method_name);
  std::unique_ptr<Function> new_fn = std::make_unique<Function>(
      qualified_method_name, to_be_copied.get_code(), to_be_copied.getSchema());
  object_->type()->addMethod(new_fn.get());
  cu_->register_function(std::move(new_fn));
}

c10::optional<Method> Module::find_method(const std::string& basename) const {
  for (const auto& fn : cu_->methods()) {
    if (fn->name() == basename) {
      return c10::make_optional<Method>(Method(this, fn.get()));
    }
  }
  return c10::nullopt;
}

namespace {
// For JIT, there is a private function to get all modules by iteration in
// struct slot_iterator_impl (jit/api/module.h). The following function use
// recursion to mimic the logic without allocating extra memory to get module
// list and set training attribute directly.
void set_train_recurse(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj,
    bool on) {
  if (auto slot = obj->type()->findAttributeSlot("training")) {
    obj->setSlot(*slot, on);
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "'training' attribute not found. Did you accidentally "
        "call .eval() before saving your model?");
  }
  for (const auto& slot : obj->slots()) {
    // slots is a list of IValue. Continue setting training attribute only
    // if the slot is an object and a module.
    if (slot.isObject() && slot.toObjectRef().type()->is_module()) {
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
  for (const auto i : c10::irange(nslots)) {
    auto slot = slots[i];
    std::string name = parent_name.empty() ? parent_name : parent_name + ".";
    name += obj->type()->getAttributeName(i);
    // TODO: Fix this filter. Requires_grad is not the appropriate
    // filter of a parameter, but is a temporary hack to help probable
    // users of this api. The correct behavior is to filter by the
    // obj->type->is_parameter() but this currently always returns
    // false on mobile.
    if (slot.isTensor() && slot.toTensor().requires_grad()) {
      (*params)[name] = slot.toTensor();
    } else if (slot.isObject()) {
      slot_named_params_recurse(slot.toObject(), params, name);
    }
  }
}

#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
std::string getTopModuleTypeName(const Module& m) {
  std::string name;
  if (m._ivalue()->type() && m._ivalue()->type()->name()) {
    name = m._ivalue()->type()->name().value().name();
  }
  return name;
}
#endif

} // namespace

const std::vector<at::Tensor> Module::parameters() const {
  std::vector<at::Tensor> params;
  slot_params_recurse(object_, &params);
  return params;
}

// Returns a mapping for all attributes that requires_grad=True in a module.
// This behavior differs from full torch script modules. This is a bug,
// but currently there is no way to correctly label parameters in the
// loading of a mobile module. TODO
const std::map<std::string, at::Tensor> Module::named_parameters() const {
  std::map<std::string, at::Tensor> params;
  const std::string name = "";
  slot_named_params_recurse(object_, &params, name);
  return params;
}

std::string Module::getModuleHierarchy(const int64_t debug_handle) const {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  return getDebugTable().getModuleHierarchyInfo(
      debug_handle, getTopModuleTypeName(*this));
#else
  return "";
#endif
}

std::string Module::getCallStack(const int64_t debug_handle) const {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  return getDebugTable().getSourceDebugString(
      debug_handle, getTopModuleTypeName(*this));
#else
  return "";
#endif
}

// We will continue to support this API for now as this is being relied upon
// for profiling.
// We really need to change this part, so in the next step for profiling support
// for delegates, the first thing will be to rewrite how profiling is done
// for lite interpreter.
std::string Module::get_forward_method_debug_info(int64_t debug_handle) const {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  return getDebugTable().getModuleHierarchyInfo(
      debug_handle, getTopModuleTypeName(*this));
#else
  return "";
#endif
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
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  auto instance_key = std::rand();
  /* if the metadata dict doesn't contain "model_name", copy the metadata and
  set the value of "model_name" as name() */
  std::unordered_map<std::string, std::string> copied_metadata =
      owner_->getMetadata();

  if (observer) {
    observer->onEnterRunMethod(instance_key);
  }

  auto debug_info = std::make_shared<MobileDebugInfo>();
  std::string name = copied_metadata["model_name"];
  debug_info->setModelName(name);
  debug_info->setMethodName(function_->name());
  at::DebugInfoGuard guard(at::DebugInfoKind::MOBILE_RUNTIME_INFO, debug_info);

  std::string error_message;
  auto failure_guard = c10::make_scope_exit([&]() {
    if (!observer) {
      return;
    }

#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
    if (error_message.empty()) {
      error_message = owner_->getDebugTable().getSourceDebugString(
          function_->getExceptionDebugHandles(), getTopModuleTypeName(*owner_));
    }
#endif

    observer->onFailRunMethod(
        copied_metadata,
        function_->name(),
        instance_key,
        error_message.empty() ? "Unknown exception" : error_message.c_str());
  });

  try {
    stack.insert(stack.begin(), owner_->_ivalue()); // self
    function_->run(stack);
    if (observer) {
      observer->onExitRunMethod(
          copied_metadata, function_->name(), instance_key);
    }
    failure_guard.release();
    // This exception must be caught first as it derived from c10::Error
  } catch (c10::BackendRuntimeException& e) {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
    for (auto handle : function_->getExceptionDebugHandles()) {
      e.pushDebugHandle(handle);
    }
    // symbolicate all handles
    auto debug_string = owner_->getDebugTable().getSourceDebugString(
        e.getDebugHandles(), getTopModuleTypeName(*owner_));
    e.add_context(debug_string);
#endif
    error_message = e.what();
    TORCH_RETHROW(e);
  } catch (c10::Error& error) {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
    auto debug_string = owner_->getDebugTable().getSourceDebugString(
        function_->getExceptionDebugHandles(), getTopModuleTypeName(*owner_));
    error.add_context(debug_string);
#endif
    error_message = error.what();
    TORCH_RETHROW(error);
  }
}

c10::IValue Method::operator()(std::vector<c10::IValue> stack) const {
  run(stack);
  TORCH_INTERNAL_ASSERT(!stack.empty());
  return stack.front();
}

c10::optional<std::string> print_type(const c10::Type& t) {
  auto namedType = t.cast<c10::NamedType>();
  if (namedType && namedType->name()) {
    return namedType->name().value().qualifiedName();
  }
  if (auto dyn = t.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str();
  }
  return c10::nullopt;
}

TORCH_API ModuleInfo get_module_info(const mobile::Module& module) {
  ModuleInfo minfo;
  minfo.operator_version = module.min_operator_version();
  minfo.bytecode_version = module.bytecode_version();
  std::vector<std::string> type_name_list;
  for (const auto& func_ptr : module.compilation_unit().methods()) {
    const auto& function = *func_ptr;
    for (const auto i : c10::irange(function.get_code().op_names_.size())) {
      const auto& op = function.get_code().op_names_[i];
      minfo.opname_to_num_args[mobile::operator_str(op)] =
          function.get_code().operator_input_sizes_[i];
    }
    for (const c10::TypePtr& tp : function.get_code().types_) {
      type_name_list.push_back(tp->annotation_str(print_type));
    }
    minfo.function_names.insert(function.qualname().qualifiedName());
  }
  c10::TypeParser parser(type_name_list);
  parser.parseList();
  minfo.type_names = parser.getContainedTypes();
  return minfo;
}

} // namespace mobile
} // namespace jit
} // namespace torch
