#include <torch/csrc/jit/script/concrete_module_type.h>

namespace torch {
namespace jit {
namespace script {
ClassTypePtr ConcreteModuleTypeBuilder::createTypeFromThis() const {
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

  for (const auto& moduleInfo : modules_) {
    cls->addAttribute(
        moduleInfo.name_, moduleInfo.meta_->getJitType(), /*is_parameter=*/false);
  }

  return cls;
}

std::shared_ptr<ConcreteModuleType> ConcreteModuleType::fromJitType(
    TypePtr type) {
  // `type` should either be a module interface or a class type
  if (auto interface = type->cast<InterfaceType>()){
    TORCH_INTERNAL_ASSERT(interface->is_module());
  } else {
    TORCH_INTERNAL_ASSERT(type->cast<ClassType>());
  }
  auto ret = std::shared_ptr<ConcreteModuleType>(new ConcreteModuleType());
  ret->jitType_ = std::move(type);
  return ret;
}

ConcreteModuleType::ConcreteModuleType(ConcreteModuleTypeBuilder data)
    : data_(std::move(data)) {
  jitType_ = data_.createTypeFromThis();
}


bool operator==(
    const ConcreteModuleTypeBuilder::ModuleInfo& lhs,
    const ConcreteModuleTypeBuilder::ModuleInfo& rhs) {
  return lhs.meta_->equals(*rhs.meta_);
}

bool ConcreteModuleTypeBuilder::equals(
    const ConcreteModuleTypeBuilder& other) const {
  if (isPoisoned_ || other.isPoisoned_) {
    return false;
  }

  // clang-format off
    // These are vaguely ordered so that cheap, discriminating checks happen first.
    bool equal =
      pyClass_.is(other.pyClass_) &&
      iterableModuleKind_ == other.iterableModuleKind_ &&
      constants_ == other.constants_ &&
      attributes_ == other.attributes_ &&
      overloads_ == other.overloads_ &&
      functionAttributes_ == other.functionAttributes_;
  // clang-format on
  if (!equal) {
    return false;
  }

  // We store modules in order of insertion (to make compilation
  // deterministic). However, for the purposes of equality, insertion order
  // should not matter, so sort them by name.
  // We put this check last because it involves the most work.
  auto thisSorted = modules_;
  std::sort(
      thisSorted.begin(),
      thisSorted.end(),
      [](const ModuleInfo& a, const ModuleInfo& b) {
        return a.name_ < b.name_;
      });

  auto otherSorted = other.modules_;
  std::sort(
      otherSorted.begin(),
      otherSorted.end(),
      [](const ModuleInfo& a, const ModuleInfo& b) {
        return a.name_ < b.name_;
      });

  return thisSorted == otherSorted;
}

TypePtr ConcreteModuleType::getJitType() const {
  return jitType_;
}

py::object ConcreteModuleType::getPyClass() const {
  return data_.pyClass_;
}

c10::optional<std::vector<std::string>> ConcreteModuleType::findOverloads(
    const std::string& name) const {
  const auto it = data_.overloads_.find(name);
  if (it != data_.overloads_.end()) {
    return it->second;
  }
  return c10::nullopt;
}

c10::optional<Function*> ConcreteModuleType::findFunctionAttribute(
    const std::string& name) const {
  const auto it = data_.functionAttributes_.find(name);
  if (it != data_.functionAttributes_.end()) {
    return it->second.function_->function();
  }
  return c10::nullopt;
}

c10::optional<std::string> ConcreteModuleType::findFailedAttribute(
    const std::string& name) const {
  const auto it = data_.failedAttributes_.find(name);
  if (it != data_.failedAttributes_.end()) {
    return it->second;
  }
  return c10::nullopt;
}

std::shared_ptr<ConcreteModuleType> ConcreteModuleType::
    findSubmoduleConcreteType(const std::string& name) const {
  const auto it = std::find_if(
      data_.modules_.cbegin(),
      data_.modules_.cend(),
      [&](const ConcreteModuleTypeBuilder::ModuleInfo& info) {
        return info.name_ == name;
      });
  if (it == data_.modules_.end()) {
    return nullptr;
  }
  return it->meta_;
}

void ConcreteModuleTypeBuilder::setIterableModuleKind(IterableModuleKind kind) {
  iterableModuleKind_ = kind;
}

IterableModuleKind ConcreteModuleType::getIterableModuleKind() const {
  return data_.iterableModuleKind_;
}

void ConcreteModuleTypeBuilder::setPoisoned() {
  isPoisoned_ = true;
}

void ConcreteModuleTypeBuilder::addConstant(std::string name, py::object value) {
  constants_.emplace(std::move(name), std::move(value));
}

void ConcreteModuleTypeBuilder::addAttribute(
    std::string name,
    TypePtr type,
    bool isParameter) {
  TORCH_INTERNAL_ASSERT(type);
  // Function attributes should be handled separately
  TORCH_INTERNAL_ASSERT(type->cast<FunctionType>() == nullptr);
  attributes_.emplace(
      std::move(name),
      ConcreteModuleTypeBuilder::Attribute(unshapedType(type), isParameter));
}

void ConcreteModuleTypeBuilder::addFunctionAttribute(
    std::string name,
    const TypePtr& type,
    py::object pyFunction) {
  TORCH_INTERNAL_ASSERT(type);
  functionAttributes_.emplace(
      std::move(name),
      ConcreteModuleTypeBuilder::FunctionAttribute{type->expect<FunctionType>(),
                                                std::move(pyFunction)});
}

void ConcreteModuleTypeBuilder::addModule(
    std::string name,
    std::shared_ptr<ConcreteModuleType> meta) {
  modules_.emplace_back(
      ConcreteModuleTypeBuilder::ModuleInfo{std::move(name), std::move(meta)});
}

void ConcreteModuleTypeBuilder::addOverload(
    std::string methodName,
    std::vector<std::string> overloadedMethodNames) {
  overloads_.emplace(std::move(methodName), std::move(overloadedMethodNames));
}

void ConcreteModuleTypeBuilder::addFailedAttribute(
    std::string name,
    std::string failureReason) {
  failedAttributes_.emplace(std::move(name), std::move(failureReason));
}

c10::optional<py::object> ConcreteModuleType::findConstant(
    const std::string& name) const {
  auto it = data_.constants_.find(name);
  if (it != data_.constants_.end()) {
    return it->second.v_;
  }
  return c10::nullopt;
}

void ConcreteModuleType::dump() const {
  std::cout << "ConcreteModuleType for: " << py::getattr(data_.pyClass_, "__name__") << "\n";
  std::cout << "Constants: \n";
  for (const auto& pr : data_.constants_) {
    std::cout << "\t" << pr.first << ": " << pr.second.v_ << "\n";
  }
  std::cout << "\nAttributes: \n";
  for (const auto& pr : data_.attributes_) {
    std::cout << "\t" << pr.first << ": " << pr.second.type_->python_str()
              << "\n";
  }
  std::cout << "\nSubmodules: \n";
  for (const auto& info : data_.modules_) {
    std::cout << "\t" << info.name_ << ": "
              << info.meta_->getJitType()->python_str() << "\n";
  }
  std::cout << "\nOverloads: \n";
  for (const auto& pr : data_.overloads_) {
    std::cout << "\t" << pr.first << ": " << pr.second << "\n";
  }
  std::string isPoisoned = data_.isPoisoned_ ? "true" : "false";
  std::cout << "isPoisoned: " << isPoisoned << "\n";
  if (jitType_) {
    std::cout << "jit type: " << jitType_->python_str() << "\n";
  }
}

std::unordered_map<std::string, py::object> ConcreteModuleType::getConstantsPy()
    const {
  // Convert to a more pybind-friendly representation, so we don't
  // need to bind ConcreteModuleType::Constant as well.
  std::unordered_map<std::string, py::object> ret;
  for (const auto& pr : data_.constants_) {
    ret.emplace(pr.first, pr.second.v_);
  }
  return ret;
}

std::unordered_map<std::string, std::pair<TypePtr, bool>> ConcreteModuleType::
    getAttributesPy() const {
  // Convert to a more pybind-friendly representation, so we don't
  // need to bind ConcreteModuleType::Attribute as well.
  std::unordered_map<std::string, std::pair<TypePtr, bool>> ret;
  for (auto& pr : data_.attributes_) {
    ret.emplace(
        pr.first,
        std::pair<TypePtr, bool>(pr.second.type_, pr.second.isParam_));
  }
  return ret;
}

std::vector<std::pair<std::string, TypePtr>> ConcreteModuleType::getModulesPy()
    const {
  std::vector<std::pair<std::string, TypePtr>> ret;

  for (const auto& info : data_.modules_) {
    ret.emplace_back(std::make_pair(info.name_, info.meta_->getJitType()));
  }
  return ret;
}

} // namespace script
} // namespace jit
} // namespace torch
