#include <torch/csrc/jit/serialization/type_name_uniquer.h>

namespace torch::jit {

c10::QualifiedName TypeNameUniquer::getUniqueName(c10::ConstNamedTypePtr t) {
  auto it = name_map_.find(t);
  if (it != name_map_.cend()) {
    // We already have a unique name for this type
    return it->second;
  }

  auto qualifiedName = t->name().value();
  if (!used_names_.count(qualifiedName)) {
    // We haven't used this qualified name yet, so assign it to this type.
    used_names_.insert(qualifiedName);
    name_map_.emplace(std::move(t), qualifiedName);
    return qualifiedName;
  }

  // The qualified name for this type is already in use by another type being
  // serialized. Mangle the name so that we can get a unique name for this type.
  auto mangled = mangler_.mangle(qualifiedName);
  while (used_names_.count(mangled)) {
    mangled = mangler_.mangle(qualifiedName);
  }

  name_map_.emplace(std::move(t), mangled);
  used_names_.insert(mangled);
  return mangled;
}

} // namespace torch::jit
