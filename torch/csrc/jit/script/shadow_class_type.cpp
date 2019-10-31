#include <torch/csrc/jit/script/shadow_class_type.h>
#include <c10/util/Exception.h>

namespace torch {
namespace jit {
namespace script {

size_t ShadowClassType::addMutableAttribute(
    const std::string& name,
    MutableTypePtr type,
    bool is_parameter) {
  const char* what = is_parameter ? "parameter" : "attribute";
  for (size_t i = 0; i < attributeNames_.size(); ++i) {
    auto mt = mutableAttributeTypes_[i];
    std::string py_str = c10::holds_alternative<c10::TypePtr>(mt) ? c10::get<c10::TypePtr>(mt)->python_str() : c10::get<ShadowClassTypePtr>(mt)->python_str();
    TORCH_CHECK(
        name != attributeNames_[i],
        "attempting to add ",
        what,
        " '",
        name,
        "' to ",
        python_str(),
        " but a field of the same name already exists with type ",
        py_str);
  }

  if (c10::holds_alternative<c10::TypePtr>(type)) {
    checkNoAny(*this, what, name, c10::get<c10::TypePtr>(type));
  }

  size_t slot = attributeNames_.size();
  attributeNames_.push_back(name);
  mutableAttributeTypes_.push_back(type);
  if (is_parameter) {
    TORCH_INTERNAL_ASSERT(is_module(), "adding a parameter to a non module");
  }
  if (is_module()) {
    parameterSlots_->push_back(is_parameter);
  }
  return slot;
}

void ShadowClassType::removeAttribute(const std::string& name) {
  std::vector<std::string> attributeNames;
  std::vector<MutableTypePtr> attributeTypes;
  auto& names = attributeNames_;
  auto& types = mutableAttributeTypes_;
  TORCH_CHECK(std::find(names.begin(), names.end(), name) != names.end(), "Can't remove a non-existent attribute");
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] != name) {
      attributeNames.push_back(names[i]);
      attributeTypes.push_back(types[i]);
    }
  }
  attributeNames_ = attributeNames;
  mutableAttributeTypes_ = attributeTypes;
}

} // namespace script
} // namespace jit
} // namespace torch
