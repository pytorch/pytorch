#include <torch/csrc/jit/frontend/concrete_module_type.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch::jit {

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
    const auto& name = pr.key();
    const auto& type = pr.value().type_;
    const auto& isParameter = pr.value().isParam_;
    const auto& isBuffer = pr.value().isBuffer_;
    cls->addAttribute(name, type, isParameter, isBuffer);
  }

  for (const auto& pr : constants_) {
    cls->addConstant(pr.first, pr.second);
  }

  for (const auto& moduleInfo : modules_) {
    cls->addAttribute(
        moduleInfo.name_,
        moduleInfo.meta_->getJitType(),
        /*is_parameter=*/false);
  }

  return cls;
}

std::shared_ptr<ConcreteModuleType> ConcreteModuleType::fromJitType(
    TypePtr type) {
  ConcreteModuleTypeBuilder builder;
  builder.setPoisoned();

  // `type` should either be a module interface or a class type
  if (auto interface = type->cast<InterfaceType>()) {
    TORCH_INTERNAL_ASSERT(interface->is_module());
  } else {
    const auto classType = type->expect<ClassType>();

    // Populate the builder metadata from the JIT type. This is to ensure
    // ConcreteModuleTypes produced from Python and ones produced from a JIT
    // type directly behave the same to the rest of the system.
    for (const auto i : c10::irange(classType->numAttributes())) {
      const auto& attrName = classType->getAttributeName(i);
      const auto& attrType = classType->getAttribute(i);
      if (attrType->is_module()) {
        builder.addModule(attrName, ConcreteModuleType::fromJitType(attrType));
      } else {
        builder.addAttribute(
            attrName,
            attrType,
            classType->is_parameter(i),
            classType->is_buffer(i));
      }
    }

    for (const auto i : c10::irange(classType->numConstants())) {
      builder.addConstant(
          classType->getConstantName(i), classType->getConstant(i));
    }
  }

  // Not make_shared because the constructor is private.
  auto ret = std::shared_ptr<ConcreteModuleType>(new ConcreteModuleType());
  ret->jitType_ = std::move(type);
  ret->data_ = builder;

  return ret;
}

ConcreteModuleType::ConcreteModuleType(ConcreteModuleTypeBuilder data)
    : data_(std::move(data)) {
  jitType_ = data_.createTypeFromThis();
}

bool operator==(
    const ConcreteModuleTypeBuilder::ModuleInfo& lhs,
    const ConcreteModuleTypeBuilder::ModuleInfo& rhs) {
  return lhs.name_ == rhs.name_ && lhs.meta_->equals(*rhs.meta_);
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
      ignoredAttributes_ == other.ignoredAttributes_ &&
      constants_ == other.constants_ &&
      attributes_ == other.attributes_ &&
      overloads_ == other.overloads_ &&
      functionAttributes_ == other.functionAttributes_ &&
      builtinFunctions_ == other.builtinFunctions_ &&
      forwardHooks_ == other.forwardHooks_ &&
      forwardPreHooks_ == other.forwardPreHooks_;
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

c10::optional<py::object> ConcreteModuleType::getPyClass() const {
  if (!data_.pyClass_) {
    return c10::nullopt;
  }
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

c10::optional<c10::Symbol> ConcreteModuleType::findBuiltinFunction(
    const std::string& name) const {
  const auto it = data_.builtinFunctions_.find(name);
  if (it != data_.builtinFunctions_.end()) {
    return it->second;
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

bool ConcreteModuleType::isIgnoredAttribute(const std::string& name) const {
  return data_.ignoredAttributes_.count(name) > 0;
}

std::shared_ptr<ConcreteModuleType> ConcreteModuleType::
    findSubmoduleConcreteType(const std::string& name) const {
  const auto it = std::find_if(
      data_.modules_.cbegin(),
      data_.modules_.cend(),
      [&](const ConcreteModuleTypeBuilder::ModuleInfo& info) {
        return info.name_ == name;
      });
  TORCH_INTERNAL_ASSERT(it != data_.modules_.end());
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

void ConcreteModuleTypeBuilder::addConstant(
    std::string name,
    py::object value) {
  auto match = tryToInferType(value);
  if (!match.success()) {
    TORCH_INTERNAL_ASSERT(
        false,
        "We need to infer the type of constant to convert the python value to IValue,"
        " but failed to infer type of ",
        py::str(value),
        "\n:",
        match.reason());
  }
  constants_.emplace(std::move(name), toIValue(std::move(value), match.type()));
}

void ConcreteModuleTypeBuilder::addConstant(std::string name, IValue value) {
  constants_.emplace(std::move(name), std::move(value));
}

void ConcreteModuleTypeBuilder::addAttribute(
    std::string name,
    const TypePtr& type,
    bool isParameter,
    bool isBuffer) {
  TORCH_INTERNAL_ASSERT(type);
  // Function attributes should be handled separately
  TORCH_INTERNAL_ASSERT(type->cast<FunctionType>() == nullptr);
  attributes_.insert(
      std::move(name),
      ConcreteModuleTypeBuilder::Attribute(
          unshapedType(type), isParameter, isBuffer));
}

void ConcreteModuleTypeBuilder::addFunctionAttribute(
    std::string name,
    const TypePtr& type,
    py::object pyFunction) {
  TORCH_INTERNAL_ASSERT(type);
  functionAttributes_.emplace(
      std::move(name),
      ConcreteModuleTypeBuilder::FunctionAttribute{
          type->expect<FunctionType>(), std::move(pyFunction)});
}

void ConcreteModuleTypeBuilder::addBuiltinFunction(
    std::string name,
    const std::string& symbol_name) {
  builtinFunctions_.emplace(
      std::move(name), c10::Symbol::fromQualString(symbol_name));
}

void ConcreteModuleTypeBuilder::addModule(
    std::string name,
    std::shared_ptr<ConcreteModuleType> meta) {
  modules_.emplace_back(std::move(name), std::move(meta));
}

void ConcreteModuleTypeBuilder::addForwardHook(py::object hook) {
  forwardHooks_.emplace_back(std::move(hook));
}

void ConcreteModuleTypeBuilder::addForwardPreHook(py::object pre_hook) {
  forwardPreHooks_.emplace_back(std::move(pre_hook));
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

void ConcreteModuleTypeBuilder::addIgnoredAttribute(std::string name) {
  ignoredAttributes_.emplace(std::move(name));
}

void ConcreteModuleType::dump() const {
  std::cout << "ConcreteModuleType for: "
            << py::getattr(data_.pyClass_, "__name__") << "\n";
  std::cout << "Constants: \n";
  for (const auto& pr : data_.constants_) {
    std::cout << "\t" << pr.first << ": " << pr.second << "\n";
  }
  std::cout << "\nAttributes: \n";
  for (const auto& pr : data_.attributes_) {
    std::cout << "\t" << pr.key() << ": " << pr.value().type_->annotation_str()
              << "\n";
  }
  std::cout << "\nSubmodules: \n";
  for (const auto& info : data_.modules_) {
    std::cout << "\t" << info.name_ << ": "
              << info.meta_->getJitType()->annotation_str() << "\n";
  }
  std::cout << "\nForward Pre-Hooks: \n";
  for (const auto& pre_hook_id : data_.forwardPreHooks_) {
    std::cout << "\t"
              << "pre_hook id: " << pre_hook_id << "\n";
  }
  std::cout << "\nForward Hooks: \n";
  for (const auto& hook_id : data_.forwardHooks_) {
    std::cout << "\t"
              << "hook id: " << hook_id << "\n";
  }
  std::cout << "\nOverloads: \n";
  for (const auto& pr : data_.overloads_) {
    std::cout << "\t" << pr.first << ": " << pr.second << "\n";
  }
  std::string isPoisoned = data_.isPoisoned_ ? "true" : "false";
  std::cout << "isPoisoned: " << isPoisoned << "\n";
  if (jitType_) {
    std::cout << "jit type: " << jitType_->annotation_str() << "\n";
  }
}

std::unordered_map<std::string, py::object> ConcreteModuleType::getConstantsPy()
    const {
  // Convert to a more pybind-friendly representation, so we don't
  // need to bind ConcreteModuleType::Constant as well.
  std::unordered_map<std::string, py::object> ret;
  for (const auto& pr : data_.constants_) {
    ret.emplace(pr.first, toPyObject(pr.second));
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
        pr.key(),
        std::pair<TypePtr, bool>(pr.value().type_, pr.value().isParam_));
  }
  return ret;
}

std::vector<std::pair<std::string, std::shared_ptr<ConcreteModuleType>>>
ConcreteModuleType::getModulesPy() const {
  std::vector<std::pair<std::string, std::shared_ptr<ConcreteModuleType>>> ret;

  ret.reserve(data_.modules_.size());
  for (const auto& info : data_.modules_) {
    ret.emplace_back(info.name_, info.meta_);
  }
  return ret;
}

} // namespace torch::jit
