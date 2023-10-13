#pragma once

#include <ATen/core/ivalue.h>

#include <utility>

namespace c10 {

struct EnumType;
using EnumTypePtr = std::shared_ptr<EnumType>;
using EnumNameValue = std::pair<std::string, IValue>;
struct TORCH_API EnumType : public NamedType {
  friend struct Type;
  static const TypeKind Kind = TypeKind::EnumType;

  static EnumTypePtr create(
      const c10::QualifiedName& qualified_class_name,
      TypePtr value,
      std::vector<EnumNameValue> enum_names_values,
      std::weak_ptr<::torch::jit::CompilationUnit> cu) {
    switch (value->kind()) {
      case TypeKind::IntType:
      case TypeKind::FloatType:
      case TypeKind::StringType:
        return EnumTypePtr(new EnumType(
            qualified_class_name,
            std::move(value),
            std::move(enum_names_values),
            std::move(cu)));
      default:
        AT_ERROR(
            "Cannot create Enum with value type '",
            value->str(),
            "', only int, float and string are supported");
    }
  }

  std::string str() const override {
    return "Enum<" + annotation_str() + ">";
  }

  std::string repr_str() const override {
    return str();
  }

  const TypePtr& getValueType() const {
    return value_type_;
  }

  bool equals(const Type& rhs) const override {
    if (auto* enum_rhs = rhs.castRaw<EnumType>()) {
      return name().value() == enum_rhs->name().value() &&
          *getValueType() == *(enum_rhs->getValueType()) &&
          this->compilation_unit() == enum_rhs->compilation_unit();
    }
    return false;
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  std::shared_ptr<const ::torch::jit::CompilationUnit> compilation_unit()
      const {
    auto cu = cu_.lock();
    return cu;
  }

  const QualifiedName& qualifiedClassName() const {
    return name().value();
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return value_type_;
  }

  const at::ArrayRef<EnumNameValue> enumNamesValues() const {
    return enum_names_values_;
  }

 private:
  EnumType(
      c10::QualifiedName qualified_class_name,
      TypePtr value_type,
      std::vector<EnumNameValue> enum_names_values,
      std::weak_ptr<torch::jit::CompilationUnit> cu)
      : NamedType(TypeKind::EnumType, std::move(qualified_class_name)),
        value_type_(std::move(value_type)),
        enum_names_values_(std::move(enum_names_values)),
        cu_(std::move(cu)) {}

  std::string annotation_str_impl(
      TypePrinter printer = nullptr) const override {
    (void)printer; // Suppress unused variable warning
    const auto& n = name().value();
    return n.qualifiedName();
  }

  TypePtr value_type_;
  std::vector<EnumNameValue> enum_names_values_;
  std::weak_ptr<::torch::jit::CompilationUnit> cu_;
};

} // namespace c10
