#pragma once

#include <ostream>
#include <vector>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/lazy/core/hash.h>

namespace torch {
namespace lazy {

class TORCH_API Shape {
 public:
  Shape() = default;

  Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes);

  std::string to_string() const;

  c10::ScalarType scalar_type() const {
    return scalar_type_;
  }
  void set_scalar_type(at::ScalarType value) {
    scalar_type_ = value;
  }

  int64_t dim() const {
    return sizes_.size();
  }
  c10::ArrayRef<int64_t> sizes() const {
    return sizes_;
  }
  int64_t size(int64_t dim) const {
    return sizes_.at(dim);
  }
  void set_size(int64_t dim, int64_t size) {
    sizes_.at(dim) = size;
  }

  const at::optional<std::vector<bool>>& is_symbolic() const {
    return is_symbolic_;
  }
  c10::SymbolicShape get_symbolic_shape() const;

  // Only sets is_symbolic, doesn't set concrete sizes
  // Also throws away any information about which symbolic dims
  // are the same size
  void set_from_symbolic(c10::SymbolicShape&);

  size_t numel() const;
  hash_t hash(bool bakeInSizes) const;

  bool operator==(const Shape& other) const;

 private:
  c10::ScalarType scalar_type_{c10::ScalarType::Undefined};

  // Stores which dimmensions are symbolic
  // If nullopt, either it hasn't been initialized or the symbolic
  // dimmensions are not calculatable
  at::optional<std::vector<bool>> is_symbolic_ = c10::nullopt;
  // Sizes are the upper bound sizes for a tensor, used by XLA.
  std::vector<int64_t> sizes_;
};

TORCH_API std::ostream& operator<<(std::ostream& out, const Shape& shape);

bool symbolicShapeEnabled();
// Calculate and applies symbolic shapes onto the
// Shape objects passed to result_shapes
void applySymbolicShapesOnLT(
    const char* schema_str,
    std::vector<c10::IValue> args,
    std::vector<Shape>& result_shapes);
} // namespace lazy
} // namespace torch
