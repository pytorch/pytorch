#include <torch/csrc/lazy/core/ops/generic.h>

namespace torch {
namespace lazy {

Generic::Generic(
    OpKind op,
    OpList operands,
    Shape shape,
    size_t num_outputs,
    hash_t hash_seed)
    : Node(op, operands, {std::move(shape)}, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(
    OpKind op,
    OpList operands,
    const std::function<Shape()>& shape_fn,
    size_t num_outputs,
    hash_t hash_seed)
    : Node(op, operands, shape_fn, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(
    OpKind op,
    OpList operands,
    size_t num_outputs,
    hash_t hash_seed)
    : Node(op, operands, num_outputs, hash_seed), hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : Node(op, std::move(shape), num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

} // namespace lazy
} // namespace torch
