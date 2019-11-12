#include <torch/csrc/jit/script/concrete_module_type.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace jit {
namespace script {

ClassTypePtr ConcreteModuleType::getJitType() const {
  TORCH_INTERNAL_ASSERT(jitType_);
  return jitType_;
}

ClassTypePtr ConcreteModuleType::createNewTypeFromThis() {
  TORCH_INTERNAL_ASSERT(!jitType_);
  TORCH_INTERNAL_ASSERT(pyClass_);

  auto cu = get_python_cu();
  py::object pyQualName = py::module::import("torch._jit_internal")
                              .attr("_qualified_name")(pyClass_);

  auto className = c10::QualifiedName(py::cast<std::string>(pyQualName));
  if (className.prefix().empty()) {
    className = c10::QualifiedName("__torch__", className.name());
  }
  if (cu->get_class(className) != nullptr) {
    className = cu->mangle(className);
  }
  auto cls = ClassType::create(std::move(className), cu, /*is_module=*/true);
  cu->register_type(cls);

  // populate type with info from the concrete type information
  for (const auto& pr : attributes_) {
    const auto& name = pr.first;
    const auto& type = pr.second.type_;
    const auto& isParameter = pr.second.isParam_;

    cls->addAttribute(name, type, isParameter);
  }

  for (const auto& pr : constants_) {
    const auto& name = pr.first;
    const auto& val = pr.second.v_;
    cls->addConstant(name, toTypeInferredIValue(val));
  }

  for (const auto& moduleInfo : modules_) {
    cls->addAttribute(
        moduleInfo.name_, moduleInfo.getJitType(), /*is_parameter=*/false);
  }

  jitType_ = std::move(cls);
  return jitType_;
}

py::object ConcreteModuleType::getPyClass() const {
  TORCH_INTERNAL_ASSERT(jitType_);
  TORCH_INTERNAL_ASSERT(pyClass_);
  return pyClass_;
}

c10::optional<std::vector<std::string>> ConcreteModuleType::findOverloads(
    const std::string& name) const {
  TORCH_INTERNAL_ASSERT(jitType_);
  const auto it = overloads_.find(name);
  if (it != overloads_.end()) {
    return it->second;
  }
  return c10::nullopt;
}

c10::optional<Function*> ConcreteModuleType::findFunctionAttribute(
    const std::string& name) const {
  TORCH_INTERNAL_ASSERT(jitType_);
  const auto it = functionAttributes_.find(name);
  if (it != functionAttributes_.end()) {
    return it->second.function_->function();
  }
  return c10::nullopt;
}

c10::optional<std::string> ConcreteModuleType::findFailedAttribute(
    const std::string& name) const {
  TORCH_INTERNAL_ASSERT(jitType_);
  const auto it = failedAttributes_.find(name);
  if (it != failedAttributes_.end()) {
    return it->second;
  }
  return c10::nullopt;
}

std::shared_ptr<ConcreteModuleType> ConcreteModuleType::
    findSubmoduleConcreteType(const std::string& name) const {
  TORCH_INTERNAL_ASSERT(jitType_);
  const auto it = std::find_if(
      modules_.cbegin(), modules_.cend(), [&](const ModuleInfo& info) {
        return info.name_ == name;
      });
  if (it == modules_.end()) {
    return nullptr;
  }
  return it->meta_;
}

void ConcreteModuleType::setIterableModuleKind(IterableModuleKind kind) {
  TORCH_INTERNAL_ASSERT(!jitType_);
  iterableModuleKind_ = kind;
}

IterableModuleKind ConcreteModuleType::getIterableModuleKind() const {
  TORCH_INTERNAL_ASSERT(jitType_);
  return iterableModuleKind_;
}

void ConcreteModuleType::setPoisoned() {
  TORCH_INTERNAL_ASSERT(!jitType_)
  isPoisoned_ = true;
}

void ConcreteModuleType::addJitType(ClassTypePtr type) {
  TORCH_INTERNAL_ASSERT(!jitType_)
  jitType_ = std::move(type);
}

void ConcreteModuleType::addPyClass(py::object pyClass) {
  TORCH_INTERNAL_ASSERT(!jitType_);
  pyClass_ = std::move(pyClass);
}

void ConcreteModuleType::addConstant(std::string name, py::object value) {
  TORCH_INTERNAL_ASSERT(!jitType_);
  constants_.emplace(std::move(name), std::move(value));
}

void ConcreteModuleType::addAttribute(
    std::string name,
    TypePtr type,
    bool isParameter) {
  TORCH_INTERNAL_ASSERT(type);
  TORCH_INTERNAL_ASSERT(!jitType_);
  // Function attributes should be handled separately
  TORCH_INTERNAL_ASSERT(type->cast<FunctionType>() == nullptr);
  attributes_.emplace(
      std::move(name), Attribute(unshapedType(type), isParameter));
}

void ConcreteModuleType::addFunctionAttribute(
    std::string name,
    const TypePtr& type,
    py::object pyFunction) {
  TORCH_INTERNAL_ASSERT(type);
  TORCH_INTERNAL_ASSERT(!jitType_);
  functionAttributes_.emplace(
      std::move(name),
      FunctionAttribute{type->expect<FunctionType>(), std::move(pyFunction)});
}

void ConcreteModuleType::addModule(
    std::string name,
    std::shared_ptr<ConcreteModuleType> meta) {
  TORCH_INTERNAL_ASSERT(!jitType_);
  modules_.emplace_back(ModuleInfo{std::move(name), std::move(meta)});
}

void ConcreteModuleType::addModuleInterface(
    std::string name,
    const TypePtr& type) {
  TORCH_INTERNAL_ASSERT(!jitType_);
  TORCH_INTERNAL_ASSERT(type->cast<InterfaceType>() && type->is_module());
  modules_.emplace_back(ModuleInfo{std::move(name), type});
}


void ConcreteModuleType::addOverload(
    std::string methodName,
    std::vector<std::string> overloadedMethodNames) {
  TORCH_INTERNAL_ASSERT(!jitType_);
  overloads_.emplace(std::move(methodName), std::move(overloadedMethodNames));
}

void ConcreteModuleType::addFailedAttribute(
    std::string name,
    std::string failureReason) {
  TORCH_INTERNAL_ASSERT(!jitType_);
  failedAttributes_.emplace(std::move(name), std::move(failureReason));
}

c10::optional<py::object> ConcreteModuleType::findConstant(
    const std::string& name) const {
  auto it = constants_.find(name);
  if (it != constants_.end()) {
    return it->second.v_;
  }
  return c10::nullopt;
}

void ConcreteModuleType::dump() const {
  std::cout << "ConcreteModuleType for: " << py::getattr(pyClass_, "__name__") << "\n";
  std::cout << "Constants: \n";
  for (const auto& pr : constants_) {
    std::cout << "\t" << pr.first << ": " << pr.second.v_ << "\n";
  }
  std::cout << "\nAttributes: \n";
  for (const auto& pr : attributes_) {
    std::cout << "\t" << pr.first << ": " << pr.second.type_->python_str()
              << "\n";
  }
  std::cout << "\nSubmodules: \n";
  for (const auto& info : modules_) {
    std::cout << "\t" << info.name_ << ": "
          << info.getJitType()->python_str() << "\n";
  }
  std::cout << "\nOverloads: \n";
  for (const auto& pr : overloads_) {
    std::cout << "\t" << pr.first << ": " << pr.second << "\n";
  }
  std::string isPoisoned = isPoisoned_ ? "true" : "false";
  std::cout << "isPoisoned: " << isPoisoned << "\n";
  if (jitType_) {
    std::cout << "jit type: " << jitType_->python_str() << "\n";
  }
}

std::unordered_map<std::string, py::object> ConcreteModuleType::getConstantsPy()
    const {
  TORCH_INTERNAL_ASSERT(jitType_);
  // Convert to a more pybind-friendly representation, so we don't
  // need to bind ConcreteModuleType::Constant as well.
  std::unordered_map<std::string, py::object> ret;
  for (const auto& pr : constants_) {
    ret.emplace(pr.first, pr.second.v_);
  }
  return ret;
}

std::unordered_map<std::string, std::pair<TypePtr, bool>> ConcreteModuleType::
    getAttributesPy() const {
  TORCH_INTERNAL_ASSERT(jitType_);
  // Convert to a more pybind-friendly representation, so we don't
  // need to bind ConcreteModuleType::Attribute as well.
  std::unordered_map<std::string, std::pair<TypePtr, bool>> ret;
  for (auto& pr : attributes_) {
    ret.emplace(
        pr.first,
        std::pair<TypePtr, bool>(pr.second.type_, pr.second.isParam_));
  }
  return ret;
}

std::vector<std::pair<std::string, TypePtr>> ConcreteModuleType::getModulesPy()
    const {
  TORCH_INTERNAL_ASSERT(jitType_);
  std::vector<std::pair<std::string, TypePtr>> ret;

  for (const ModuleInfo& info: modules_) {
    ret.emplace_back(std::make_pair(info.name_, info.getJitType()));
  }
  return ret;
}

} // namespace script
} // namespace jit
} // namespace torch
