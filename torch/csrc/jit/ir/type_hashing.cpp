#include <torch/csrc/jit/ir/type_hashing.h>
#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/hash.h>

namespace torch {
namespace jit {

size_t HashType::operator()(const TypePtr& type) const {
  if (auto named_type = type->cast<ClassType>()) {
    return get_hash(named_type->name().value());
  }
  auto hashes = fmap(type->containedTypes(), [](const TypePtr& elem) {
    HashType hash;
    return hash(elem);
  });
  auto typekind_hash = type->kind();
  return get_hash(typekind_hash, hashes);
};

bool EqualType::operator()(const TypePtr& a, const TypePtr& b) const {
  return *a == *b;
};

} // namespace jit
} // namespace torch
