#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>

namespace c10 {

void ShadowClassType::removeAttribute(const std::string& name) {
  std::vector<std::string> attributeNames;
  std::vector<TypePtr> attributeTypes;
  auto& names = attributeNames_;
  auto& types = attributeTypes_;
  TORCH_CHECK(std::find(names.begin(), names.end(), name) != names.end(), "Can't remove a non-existent attribute");
  for (int i = 0; i < names.size(); ++i) {
    if (names[i] != name) {
      attributeNames.push_back(names[i]);
      attributeTypes.push_back(types[i]);
    }
  }
  attributeNames_ = attributeNames;
  attributeTypes_ = attributeTypes;
}

} // namespace c10
