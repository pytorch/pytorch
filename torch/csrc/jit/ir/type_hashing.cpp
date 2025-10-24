#include <torch/csrc/jit/ir/type_hashing.h>

#include <c10/util/hash.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

namespace {
size_t hashType(const Type& type) {
  if (auto named_type = type.castRaw<ClassType>()) {
    return c10::get_hash(
        named_type->name().value(), named_type->compilation_unit());
  }
  size_t hash = 0;
  for (const auto& containedType : type.containedTypes()) {
    hash = at::hash_combine(hash, hashType(*containedType));
  }
  hash = at::hash_combine(hash, get_hash(type.kind()));
  return hash;
}
} // namespace

size_t HashType::operator()(const TypePtr& type) const {
  return hashType(*type);
}

size_t HashType::operator()(const c10::ConstTypePtr& type) const {
  return hashType(*type);
}

bool EqualType::operator()(const TypePtr& a, const TypePtr& b) const {
  return *a == *b;
}

bool EqualType::operator()(
    const c10::ConstTypePtr& a,
    const c10::ConstTypePtr& b) const {
  return *a == *b;
}

} // namespace torch::jit
