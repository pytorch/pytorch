#pragma once
#include <ATen/core/jit_type.h>
#include <c10/util/variant.h>

namespace torch {
namespace jit {
namespace script {

struct ShadowClassType;
using ShadowClassTypePtr = std::shared_ptr<ShadowClassType>;

using MutableTypePtr = c10::variant<c10::TypePtr, ShadowClassTypePtr>;

// TODO: maybe we should refactor the common parts of ClassType and
// ShadowClassType to a separate class
struct TORCH_API ShadowClassType : c10::ClassType {
  static ShadowClassTypePtr create(c10::ClassTypePtr shadowed) {
    return ShadowClassTypePtr(new ShadowClassType(shadowed));
  }

  size_t addMutableAttribute(
      const std::string& name,
      MutableTypePtr type,
      bool is_parameter = false);

  size_t numAttributes() const override {
    AT_ASSERT(attributeNames_.size() == mutableAttributeTypes_.size());
    return attributeNames_.size();
  }

  c10::optional<MutableTypePtr> getMutableAttribute(const std::string& name) const {
    AT_ASSERT(attributeNames_.size() == mutableAttributeTypes_.size());
    size_t pos = 0;
    for (const auto& attr : attributeNames_) {
      if (name == attr) {
        break;
      }
      ++pos;
    }

    if (pos >= attributeNames_.size()) {
      return c10::nullopt;
    }
    return mutableAttributeTypes_[pos];
  }

  const MutableTypePtr& getMutableAttribute(size_t slot) const {
    AT_ASSERT(attributeNames_.size() == mutableAttributeTypes_.size());
    AT_ASSERT(slot < mutableAttributeTypes_.size());
    return mutableAttributeTypes_[slot];
  }

  c10::optional<size_t> findMutableAttributeSlot(const std::string& name) const {
    AT_ASSERT(attributeNames_.size() == mutableAttributeTypes_.size());
    size_t slot = 0;
    for (const auto& attr : attributeNames_) {
      if (name == attr) {
        return slot;
      }
      slot++;
    }
    return c10::nullopt;
  }

  size_t getMutableAttributeSlot(const std::string& name) const {
    if (auto r = findMutableAttributeSlot(name)) {
      return *r;
    }
    TORCH_CHECK(
        false,
        python_str(),
        " does not have a field with the name '",
        name,
        "'");
  }

  std::string python_str() const override {
    return "shadowed " + name()->qualifiedName();
  }


  void removeAttribute(const std::string& name);

private:
  explicit ShadowClassType(c10::ClassTypePtr shadowed) : ClassType(shadowed->name(), shadowed->compilation_unit(), shadowed->is_module()) {
    const auto& names = shadowed->attributeNames();
    const auto& types = shadowed->containedTypes();
    for(size_t i = 0; i < names.size(); ++i) {
      if (types[i]->kind() == c10::TypeKind::ClassType) {
        addMutableAttribute(names[i], ShadowClassType::create(types[i]->expect<ClassType>()));
      } else {
        addMutableAttribute(names[i], types[i]);
      }
    }
    // Copy methods over
    // TODO: validate the type here
    for (const auto& method : shadowed->methods()) {
      addMethod(method);
    }
  }

  std::vector<MutableTypePtr> mutableAttributeTypes_;
};

} // namespace torch
} // namespace jit
} // namespace torch
