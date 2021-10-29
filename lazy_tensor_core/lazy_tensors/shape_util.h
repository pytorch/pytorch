#pragma once
#include <c10/util/Optional.h>

#include <complex>

#include "lazy_tensors/primitive_types.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensors/util.h"
#include "torch/csrc/jit/tensorexpr/types.h"
#include "torch/csrc/lazy/core/hash.h"

namespace torch {
namespace lazy {
    // Adapters that provide torch::lazy Hash functions for lazy_tensors types
    torch::lazy::hash_t Hash(const lazy_tensors::Shape& shape);
}
}

namespace lazy_tensors {

class ShapeIndex {
 public:
  ShapeIndex() = default;
  ShapeIndex(std::initializer_list<int64_t> init) : indices_(init) {}

  bool empty() const { return indices_.empty(); }
  size_t size() const { return indices_.size(); }
  void push_back(int64_t value) { indices_.push_back(value); }
  void pop_back() { indices_.pop_back(); }

  const int64_t& operator[](size_t i) const { return indices_[i]; }
  int64_t& operator[](size_t i) { return indices_[i]; }

 private:
  std::vector<int64_t> indices_;
};

class ShapeUtil {
 public:
  static int64_t ElementsIn(const Shape& shape) {
    return util::Multiply<int64_t>(shape.dimensions());
  }
  static bool SameDimensions(const Shape& lhs, const Shape& rhs) {
    return lhs.dimensions() == rhs.dimensions();
  }

  static bool Compatible(const Shape& lhs, const Shape& rhs) {
    return lhs == rhs;
  }

  static Shape ChangeElementType(const Shape& original, c10::ScalarType type) {
    if (original.IsTuple()) {
      std::vector<Shape> new_operands;
      new_operands.reserve(original.tuple_shapes_size());
      for (const Shape& operand : original.tuple_shapes()) {
        new_operands.push_back(ChangeElementType(operand, type));
      }
      return MakeTupleShape(new_operands);
    } else {
      Shape new_shape = original;
      new_shape.set_element_type(type);
      return new_shape;
    }
  }

  static Shape MakeTupleShape(c10::ArrayRef<Shape> shapes) {
    return Shape(shapes);
  }

  static Shape MakeShape(c10::ScalarType element_type,
                         c10::ArrayRef<int64_t> dimensions) {
    return lazy_tensors::Shape(element_type, dimensions);
  }

  // Returns the number of elements in the given tuple shape.
  // Precondition: IsTuple(shape)
  static int64_t TupleElementCount(const Shape& shape);

  static const Shape& GetTupleElementShape(const Shape& shape, int64_t index) {
    LOG(FATAL) << "Not implemented yet.";
  }

  // Calls the given visitor function for each subshape of the given shape.
  // Subshapes are visited in DFS pre-order starting with the entire shape
  // (index {}).
  using VisitorFunction = std::function<void(const Shape& /*subshape*/,
                                             const ShapeIndex& /*index*/)>;
  static void ForEachSubshape(const Shape& shape, const VisitorFunction& func);
  using MutatingVisitorFunction =
      std::function<void(Shape* /*subshape*/, const ShapeIndex& /*index*/)>;
  static void ForEachMutableSubshape(Shape* shape,
                                     const MutatingVisitorFunction& func) {
    if (!shape->IsTuple()) {
      return;
    }
    for (size_t i = 0; i < shape->tuple_shapes_size(); ++i) {
      func(shape, {static_cast<int64_t>(i)});
    }
  }

  static bool ElementIsIntegral(const Shape& shape) {
    return isIntegralType(shape.at_element_type(), /* include_bool */ true);
  }

  // Variants of ForEach(Mutable)Subshape which propagate Status from the
  // visitor function.
  using StatusVisitorFunction = std::function<Status(
      const Shape& /*subshape*/, const ShapeIndex& /*index*/)>;

  // Compute a hash for `shape`.
  static size_t Hash(const Shape& shape);
};

}  // namespace lazy_tensors
