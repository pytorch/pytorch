#pragma once

#include <memory>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <optional>

namespace torch::jit {
struct CompilationUnit;
struct Function;
} // namespace torch::jit

namespace c10 {

struct FunctionSchema;

// This enumerator represents the 'kind' of an attribute - a buffer, a
// parameter, or neither. This state is mutually exclusive. Buffers and
// Parameters can only appear on modules.
enum class AttributeKind { BUFFER, PARAMETER, REGULAR_ATTRIBUTE };

// This structure represents all notional booking entities in a class attribute:
// name, kind (see: AttributeKind), and type (see: TypePtr). Note: This
// structure does not represent the value of the attribute.
struct TORCH_API ClassAttribute {
 public:
  ClassAttribute(
      AttributeKind kind,
      TypePtr attributeType,
      std::string attributeName)
      : kind_(kind),
        attributeType_(std::move(attributeType)),
        attributeName_(std::move(attributeName)) {}

  AttributeKind getKind() const {
    return kind_;
  }

  const TypePtr& getType() const {
    return attributeType_;
  }

  const std::string& getName() const {
    return attributeName_;
  }

 private:
  AttributeKind kind_;
  TypePtr attributeType_;
  std::string attributeName_;
};

/**
 * User Defined Types
 */

struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;
using ::torch::jit::CompilationUnit;

// This represents a class in TorchScript.
struct TORCH_API ClassType : public NamedType {
  // This represents an attribute of a class; a name associated with an
  // attribute, and a getter and (optional) setter for that attribute.
  struct Property {
    std::string name;
    torch::jit::Function* getter;
    torch::jit::Function* setter;
  };

  // Create a class type with name `name` and its methods stored in `cu`.
  static ClassTypePtr create(
      std::optional<QualifiedName> qualifiedName,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module = false,
      std::string doc_string = "",
      std::vector<std::string> unresolved_class_attributes = {});

  bool equals(const Type& rhs) const override {
    if (this == &rhs) {
      return true;
    }
    if (auto user_rhs = rhs.castRaw<ClassType>()) {
      const auto& lhs_name = name();
      const auto& rhs_name = user_rhs->name();
      return lhs_name.has_value() && lhs_name == rhs_name &&
          this->compilation_unit() == user_rhs->compilation_unit();
    }
    return false;
  }

  std::string str() const override {
    return annotation_str();
  }

  std::string repr_str() const override {
    std::stringstream ss;
    ss << str()
       << " (of Python compilation unit at: " << compilation_unit().get()
       << ")";
    return ss.str();
  }

  const std::vector<torch::jit::Function*>& methods() const;

  TypePtr findAttribute(const std::string& name) const {
    size_t pos = 0;
    for (const auto& attr : attributes_) {
      if (name == attr.getName()) {
        break;
      }
      ++pos;
    }

    if (pos >= attributes_.size()) {
      return nullptr;
    }
    return attributes_[pos].getType();
  }

  const TypePtr& getAttribute(const std::string& name) const {
    auto slot = findAttributeSlot(name);
    TORCH_CHECK(
        slot, repr_str(), " does not have an attribute with name '", name, "'");
    return attributes_[*slot].getType();
  }

  size_t numAttributes() const {
    return attributes_.size();
  }

  const TypePtr& getAttribute(size_t slot) const {
    AT_ASSERT(slot < attributes_.size());
    return attributes_.at(slot).getType();
  }

  const std::string getAttributeName(size_t slot) const {
    AT_ASSERT(slot < attributes_.size());
    return attributes_[slot].getName();
  }

  void checkNotExist(const std::string& name, const std::string& what) const;

  // Attributes are stored in a specific slot at runtime for effiency.
  // When emitting instructions we specify the slot so that attribute access is
  // a constant lookup
  std::optional<size_t> findAttributeSlot(const std::string& name) const {
    size_t slot = 0;
    for (const auto& attr : attributes_) {
      if (name == attr.getName()) {
        return slot;
      }
      slot++;
    }
    return std::nullopt;
  }
  size_t getAttributeSlot(const std::string& name) const {
    if (auto r = findAttributeSlot(name)) {
      return *r;
    }
    TORCH_CHECK(
        false,
        repr_str(),
        " does not have an attribute with name '",
        name,
        "'");
  }

  bool hasAttribute(const std::string& name) const {
    return std::find_if(
               attributes_.cbegin(),
               attributes_.cend(),
               [&](const ClassAttribute& attr) {
                 return attr.getName() == name;
               }) != attributes_.cend();
  }

  bool isUnresolvedClassAttribute(const std::string& name) const;

  at::ArrayRef<TypePtr> containedTypes() const override {
    return attributeTypes_;
  }

  size_t addAttribute(
      const std::string& name,
      TypePtr type,
      bool is_parameter = false,
      bool is_buffer = false);

  // [Internal Only] Remove attribute from the ClassType,
  // caller is responsible to make sure the modification is safe:
  // it is unsafe to having existing allocations
  // of this object around anymore, and any code that works on
  // the attribute is now invalid. Only newly created code is
  // valid again.
  void unsafeRemoveAttribute(const std::string& name);

  // [Internal Only] Change the type of an attribute of the ClassType,
  // The caller is responsible to make sure the modification is safe:
  // it is unsafe to maintain uses of the old type of the attribute,
  // and any code that works on the attribute is now invalid.
  // Only newly created code is valid again.
  void unsafeChangeAttributeType(
      const std::string& name,
      const TypePtr& new_ty);

  // Add attribute \p NAME if it doesn't exist or verify that it has a
  // compatible type otherwise.
  size_t addOrCheckAttribute(
      const std::string& name,
      TypePtr ty,
      bool is_parameter = false,
      bool is_buffer = false) {
    auto slot_idx = findAttributeSlot(name);
    if (!slot_idx) {
      return addAttribute(name, std::move(ty), is_parameter, is_buffer);
    }

    TORCH_CHECK(
        is_parameter == this->is_parameter(*slot_idx),
        "Parameter field mismatch for the field '",
        name,
        "'");
    const TypePtr& atype = getAttribute(*slot_idx);
    TORCH_CHECK(
        ty->isSubtypeOf(*atype),
        ty->repr_str(),
        " is not compatible with the type ",
        atype->repr_str(),
        " for the field '",
        name,
        "'");
    return *slot_idx;
  }

  // Get the property with the given \p name, if it exists on the class.
  std::optional<ClassType::Property> getProperty(const std::string& name);
  // Add a property named \p name with \p getter and \p setter as its getter and
  // setter.
  void addProperty(
      const std::string& name,
      torch::jit::Function* getter,
      torch::jit::Function* setter);
  // Get a list of all properties.
  const std::vector<Property>& properties() const {
    return properties_;
  }

  bool hasConstant(const std::string& name) const {
    return std::find_if(
               constantNames_.cbegin(),
               constantNames_.cend(),
               [&](const std::string& constant) { return constant == name; }) !=
        constantNames_.cend();
  }

  size_t addConstant(const std::string& name, const IValue& value);

  std::optional<size_t> findConstantSlot(const std::string& name) const;

  size_t getConstantSlot(const std::string& name) const {
    if (auto r = findConstantSlot(name)) {
      return *r;
    }
    TORCH_CHECK(
        false,
        repr_str(),
        " does not have constant field with the name '",
        name,
        "'");
  }

  const std::string& getConstantName(size_t slot) const;

  const std::string& doc_string() const {
    return doc_string_;
  }

  IValue getConstant(const std::string& name) const;

  IValue getConstant(size_t slot) const;

  std::optional<IValue> findConstant(const std::string& name) const;

  size_t numConstants() const;

  at::ArrayRef<std::string> constantNames() const {
    return constantNames_;
  }

  at::ArrayRef<IValue> constantValues() const;

  // [Internal Only] Remove constant from the ClassType
  // caller is responsible to make sure the modification is safe:
  // it is unsafe to having existing allocations
  // of this object around anymore, and any code that works on
  // the attribute is now invalid. Only newly created code is
  // valid again.
  void unsafeRemoveConstant(const std::string& name);

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    auto ptr = ClassType::create(name(), compilation_unit_, is_module());
    AT_ASSERT(numAttributes() == contained_types.size());
    for (size_t i = 0; i < attributes_.size(); ++i) {
      AT_ASSERT(attributes_[i].getType()->isSubtypeOf(*contained_types[i]));
      ptr->addAttribute(
          attributes_[i].getName(), std::move(contained_types[i]));
    }
    // Copy methods over
    for (const auto& method : methods()) {
      ptr->addMethod(method);
    }
    return ptr;
  }

  bool is_module() const override {
    return isModule_;
  }

  const std::vector<ClassAttribute>& getAttributes() const {
    return attributes_;
  }

  bool is_parameter(size_t slot) const {
    TORCH_INTERNAL_ASSERT(
        is_module(), "asking for parameterSlots of non-Module");
    return attributes_.at(slot).getKind() == AttributeKind::PARAMETER;
  }

  bool is_buffer(size_t slot) const {
    TORCH_INTERNAL_ASSERT(
        is_module(), "asking for bufferWrittenSlots of non-Module");
    return attributes_.at(slot).getKind() == AttributeKind::BUFFER;
  }

  void addForwardPreHook(torch::jit::Function* pre_hook_ptr);
  void addForwardHook(torch::jit::Function* hook_ptr);
  torch::jit::Function* findForwardPreHook(const std::string& name) const;
  torch::jit::Function* findForwardHook(const std::string& name) const;
  const std::vector<torch::jit::Function*>& getForwardHooks() const;
  const std::vector<torch::jit::Function*>& getForwardPreHooks() const;

  void checkForwardPreHookSchema(
      size_t pre_hook_idx,
      const FunctionSchema& pre_hook_schema) const;
  void checkForwardHookSchema(
      size_t hook_idx,
      const FunctionSchema& hook_schema) const;

  void addMethod(torch::jit::Function* method);
  torch::jit::Function* findMethod(const std::string& name) const;
  torch::jit::Function& getMethod(const std::string& name) const;
  torch::jit::Function* findHook(const std::string& name) const;
  torch::jit::Function& getHook(const std::string& name) const;
  bool hasMethod(const std::string& name) const;

  torch::jit::Function* findStaticMethod(const std::string& name) const;
  void addStaticMethod(torch::jit::Function* method);

  // [Internal Only] Remove method from the ClassType
  // caller is responsible to make sure the modification is safe:
  // it is unsafe to having existing allocations
  // of this object around anymore, and any code that works on
  // the attribute is now invalid. Only newly created code is
  // valid again.
  // Note this method is intended for freezing only.
  void unsafeRemoveMethod(const std::string& name);

  std::shared_ptr<CompilationUnit> compilation_unit();

  std::shared_ptr<const CompilationUnit> compilation_unit() const;

  // generate a refined version of this class.
  // It has the same name but the slot Types are subtypes of
  // the original slots. It is only valid to refine a class type in a context
  // where it is know that there are not assignments to the objects slots
  // that would invalidate the refinement.
  // These variants are not registered in the global class table.
  ClassTypePtr refine(at::ArrayRef<TypePtr> refined_slots) const;

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  static const TypeKind Kind = TypeKind::ClassType;

 private:
  ClassType(
      std::optional<QualifiedName> name,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module = false,
      std::string doc_string = "",
      std::vector<std::string> unresolved_class_attributes = {});

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return name()->qualifiedName();
  }

  void addAttribute(ClassAttribute classAttribute);
  std::string getForwardPreHookErrorMessage(size_t pre_hook_idx) const;
  std::string getForwardHookErrorMessage(size_t hook_idx) const;

  // Mapping of attribute names -> their type.
  // NOTE: this does not contain methods, which are stored in the module
  // TODO: once modules support arbitrary ivalue attributes, we don't need this
  // anymore.
  // TODO: This is better represented as an OrderedDict, but alas it is not yet
  // available from c10

  // Mapping of constant names -> their value.
  std::vector<std::string> constantNames_;
  std::vector<IValue> constantValues_;
  // Holds method attributes
  std::weak_ptr<CompilationUnit> compilation_unit_;

  // Holds all atrributes, attribute details are found on ClassAttribute
  std::vector<ClassAttribute> attributes_;
  // Construct mirroring attributes_, only around due to the fact that
  // `containedTypes()` method returns an ArrayRef. Never fill this without
  // using the appropriate provideNewClassAttribute method
  std::vector<TypePtr> attributeTypes_;

  // List of methods associated with this class.
  std::vector<torch::jit::Function*> methods_;
  std::vector<torch::jit::Function*> staticmethods_;

  // List of hooks to be run before/after forward.
  std::vector<torch::jit::Function*> forward_hooks_;
  std::vector<torch::jit::Function*> forward_pre_hooks_;

  // List of properties exposed by this class.
  std::vector<Property> properties_;

  bool isModule_ = false;

  // Doc string of class.
  std::string doc_string_;

  // For error reporting accesses to class level attributes.
  std::vector<std::string> unresolved_class_attributes_;
};

} // namespace c10
