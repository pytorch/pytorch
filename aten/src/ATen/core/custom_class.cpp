#include <torch/custom_class.h>
#include <ATen/record_function.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/functional.h>
#include <ATen/core/type_factory.h>

#include <atomic>
#include <unordered_map>

namespace c10 {

ska::flat_hash_map<std::type_index, c10::ClassTypePtr>& getCustomClassTypeMap() {
  static ska::flat_hash_map<std::type_index, c10::ClassTypePtr> tmap;
  return tmap;
}

std::unordered_map<std::string, std::function<PyObject*(void*)>>&
getClassConverter() {
  static std::unordered_map<std::string, std::function<PyObject*(void*)>>
      classConverter;
  return classConverter;
}

} // namespace c10

namespace torch {

namespace detail {

void record_custom_class(std::string name) {
  RECORD_FUNCTION_WITH_SCOPE(at::RecordScope::CUSTOM_CLASS, name, c10::ArrayRef<const c10::IValue>{});
}

} // namespace detail

std::unordered_map<std::string, at::ClassTypePtr>& customClasses() {
  static std::unordered_map<std::string, at::ClassTypePtr> customClasses;
  return customClasses;
}

void registerCustomClass(at::ClassTypePtr class_type) {
  TORCH_INTERNAL_ASSERT(class_type->name());
  auto name = class_type->name()->qualifiedName();
  TORCH_CHECK(
      !customClasses().count(name),
      "Custom class with name ",
      name,
      " is already registered. Ensure that registration with torch::class_ is only called once.");
  customClasses()[name] = std::move(class_type);
}

at::ClassTypePtr getCustomClass(const std::string& class_name) {
  auto ret = customClasses().count(class_name) ? customClasses()[class_name] : nullptr;
  if (ret) {
    RECORD_CUSTOM_CLASS(class_name);
  }
  return ret;
}

const std::unordered_set<std::string> getAllCustomClassesNames() {
  std::unordered_set<std::string> ret;
  for (const auto& kv: customClasses()) {
    ret.insert(kv.first);
  }
  return ret;
}

bool isCustomClass(const c10::IValue& v) {
  return v.isObject() && v.toObject()->type()->name() &&
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

std::vector<std::unique_ptr<jit::Function>>& customClassMethods() {
  static std::vector<std::unique_ptr<jit::Function>> customClassMethods;
  return customClassMethods;
}

void registerCustomClassMethod(std::unique_ptr<jit::Function> fn) {
  customClassMethods().emplace_back(std::move(fn));
}

std::vector<c10::FunctionSchema> customClassSchemasForBCCheck() {
    auto& methods = customClassMethods();
    return c10::fmap(methods, [](const std::unique_ptr<jit::Function>& fn) {
      return fn->getSchema();
    });
}

namespace detail {
class_base::class_base(
  const std::string& namespaceName,
  const std::string& className,
  std::string doc_string,
  const std::type_info& intrusivePtrClassTypeid,
  const std::type_info& taggedCapsuleClassTypeid)
    : qualClassName("__torch__.torch.classes." + namespaceName + '.' + className),
      classTypePtr(at::ClassType::create(
                       c10::QualifiedName(qualClassName),
                       std::weak_ptr<jit::CompilationUnit>(),
                       /*is_module=*/false,
                       std::move(doc_string)))
{
    detail::checkValidIdent(namespaceName, "Namespace name");
    detail::checkValidIdent(className, "Class name");
    classTypePtr->addAttribute("capsule", c10::TypeFactory::get<c10::CapsuleType>());
    c10::getCustomClassTypeMap().insert(
        {std::type_index(intrusivePtrClassTypeid), classTypePtr});
    c10::getCustomClassTypeMap().insert(
        {std::type_index(taggedCapsuleClassTypeid), classTypePtr});

    registerCustomClass(classTypePtr);
}

c10::FunctionSchema class_base::withNewArguments(
    const c10::FunctionSchema& schema,
    std::initializer_list<arg> default_args) {
  const auto& old_args = schema.arguments();
  std::vector<c10::Argument> new_args;
  new_args.reserve(old_args.size());

  new_args.emplace_back(old_args[0]);
  // Skip self.
  size_t argIdx = 1;
  for (const auto& default_arg : default_args) {
    auto& old_arg = old_args[argIdx++];
    new_args.emplace_back(
        default_arg.name_,
        old_arg.type(),
        old_arg.N(),
        default_arg.value_);
  }
  return schema.cloneWithArguments(std::move(new_args));
}

} // namespace detail
} // namespace torch
