#pragma once

#include <ATen/core/qualified_name.h>
#include <torch/csrc/Export.h>

namespace torch::jit {

/**
 * class NameMangler
 *
 * Utility to mangle qualified names in order to make them unique. We use this
 * in various places where we to de-duplicate qualified names.
 */
class TORCH_API NameMangler {
 public:
  // Given a qualified name, return a mangled version that is guaranteed to be
  // unique with respect to previous/future calls of `mangled()` on this name
  // mangler instance.
  c10::QualifiedName mangle(const c10::QualifiedName& name);

 private:
  size_t mangleIndex_ = 0;
};

} // namespace torch::jit
