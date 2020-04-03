#include <torch/csrc/jit/frontend/name_mangler.h>

namespace torch {
namespace jit {

c10::QualifiedName NameMangler::mangle(const c10::QualifiedName& name) {
  static const std::string manglePrefix = "___torch_mangle_";
  std::vector<std::string> atoms = name.atoms();

  // Search for an already-existing mangle namespace.
  // If the name is already mangled, just bump the integer.
  for (auto& atom : atoms) {
    auto pos = atom.find(manglePrefix);
    if (pos != std::string::npos) {
      auto num = atom.substr(pos + manglePrefix.size());
      // current mangle index in the name
      size_t num_i = c10::stoi(num);
      // bump the mangleIndex_ to num_i + 1
      mangleIndex_ = std::max(mangleIndex_, num_i + 1);
      std::string newAtomPrefix;
      newAtomPrefix.reserve(atom.size());
      // Append the part of the name up to the end of the prefix
      newAtomPrefix.append(atom, 0, pos);
      newAtomPrefix.append(manglePrefix);
      atom = newAtomPrefix + c10::to_string(mangleIndex_++);
      // increment mangleIndex_ until the type is not defined
      return c10::QualifiedName(atoms);
    }
  }

  // Otherwise add a mangle namespace right before the basename
  TORCH_INTERNAL_ASSERT(!atoms.empty());
  atoms.insert(atoms.end() - 1, manglePrefix + c10::to_string(mangleIndex_++));
  return c10::QualifiedName(atoms);
}

} // namespace jit
} // namespace torch
