#pragma once

#include "absl/types/optional.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/primitive_types.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/util.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace lazy_tensors {

class ShapeIndex {
 public:
  ShapeIndex(std::initializer_list<int64> init) : indices_(init) {}

  bool empty() const { return indices_.empty(); }
  size_t size() const { return indices_.size(); }

  const int64& operator[](size_t i) const { return indices_[i]; }
  int64& operator[](size_t i) { return indices_[i]; }

 private:
  std::vector<int64> indices_;
};

class ShapeUtil {
 public:
  static int64 ElementsIn(const Shape& shape) {
    return util::Multiply<lazy_tensors::int64>(shape.dimensions());
  }

  static int64 ByteSizeOfPrimitiveType(PrimitiveType primitive_type) {
    switch (primitive_type) {
      case PrimitiveType::PRED:
        return sizeof(int8);
      case PrimitiveType::S8:
        return sizeof(int8);
      case PrimitiveType::S16:
        return sizeof(int16);
      case PrimitiveType::S32:
        return sizeof(int32);
      case PrimitiveType::S64:
        return sizeof(int64);
      case PrimitiveType::U8:
        return sizeof(uint8);
      case PrimitiveType::U16:
        return sizeof(uint16);
      case PrimitiveType::U32:
        return sizeof(uint32);
      case PrimitiveType::U64:
        return sizeof(uint64);
      case PrimitiveType::BF16:
        return sizeof(float) / 2;
      case PrimitiveType::F16:
        return sizeof(float) / 2;
      case PrimitiveType::F32:
        return sizeof(float);
      case PrimitiveType::F64:
        return sizeof(double);
      case PrimitiveType::C64:
        return sizeof(complex64);
      case PrimitiveType::C128:
        return sizeof(complex128);
      default:
        LTC_LOG(FATAL) << "Unhandled primitive type " << primitive_type;
    }
  }

  static bool SameDimensions(const Shape& lhs, const Shape& rhs) {
    return lhs.dimensions() == rhs.dimensions();
  }

  static bool Compatible(const Shape& lhs, const Shape& rhs) {
    return lhs == rhs;
  }

  static Shape ChangeElementType(const Shape& original, PrimitiveType type) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  static Shape MakeTupleShape(lazy_tensors::Span<const Shape> shapes) {
    return Shape(shapes);
  }

  static Shape MakeShape(PrimitiveType element_type,
                         lazy_tensors::Span<const int64> dimensions) {
    return MakeShapeWithDescendingLayout(element_type, dimensions);
  }

  static Shape MakeShapeWithLayout(
      PrimitiveType element_type, lazy_tensors::Span<const int64> dimensions,
      lazy_tensors::Span<const int64> minor_to_major,
      lazy_tensors::Span<const Tile> tiles = {}, int64 element_size_in_bits = 0,
      int64 memory_space = 0) {
    LTC_CHECK(tiles.empty());
    LTC_CHECK_EQ(element_size_in_bits, 0);
    LTC_CHECK_EQ(memory_space, 0);
    LTC_CHECK_EQ(dimensions.size(), minor_to_major.size());
    LTC_CHECK(element_type != PrimitiveType::INVALID &&
              element_type != PrimitiveType::TUPLE);
    Layout layout;
    for (int64 dimension_number : minor_to_major) {
      layout.add_minor_to_major(dimension_number);
    }
    Shape shape(element_type, dimensions);
    *shape.mutable_layout() = layout;
    return shape;
  }

  static Shape MakeShapeWithDescendingLayout(
      PrimitiveType element_type, lazy_tensors::Span<const int64> dimensions) {
    std::vector<int64> layout(dimensions.size());
    std::iota(layout.rbegin(), layout.rend(), static_cast<int64>(0));
    return MakeShapeWithLayout(element_type, dimensions, layout);
  }

  static const Shape& GetTupleElementShape(const Shape& shape, int64 index) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  using MutatingVisitorFunction =
      std::function<void(Shape* /*subshape*/, const ShapeIndex& /*index*/)>;
  static void ForEachMutableSubshape(Shape* shape,
                                     const MutatingVisitorFunction& func) {
    if (!shape->IsTuple()) {
      return;
    }
    for (size_t i = 0; i < shape->tuple_shapes_size(); ++i) {
      func(shape, {static_cast<int64>(i)});
    }
  }

  static bool ElementIsIntegral(const Shape& shape) {
    return primitive_util::IsIntegralType(shape.element_type());
  }
};

inline torch::jit::tensorexpr::ScalarType PrimitiveToScalarType(
    lazy_tensors::PrimitiveType scalar_type) {
  switch (scalar_type) {
    case lazy_tensors::PrimitiveType::S8: {
      return torch::jit::tensorexpr::ScalarType::Char;
    }
    case lazy_tensors::PrimitiveType::S16: {
      return torch::jit::tensorexpr::ScalarType::Short;
    }
    case lazy_tensors::PrimitiveType::S32: {
      return torch::jit::tensorexpr::ScalarType::Int;
    }
    case lazy_tensors::PrimitiveType::S64: {
      return torch::jit::tensorexpr::ScalarType::Long;
    }
    case lazy_tensors::PrimitiveType::U8: {
      return torch::jit::tensorexpr::ScalarType::Byte;
    }
    case lazy_tensors::PrimitiveType::F32: {
      return torch::jit::tensorexpr::ScalarType::Float;
    }
    case lazy_tensors::PrimitiveType::F64: {
      return torch::jit::tensorexpr::ScalarType::Double;
    }
    case lazy_tensors::PrimitiveType::PRED: {
      return torch::jit::tensorexpr::ScalarType::Bool;
    }
    default: { LTC_LOG(FATAL) << "Not implemented yet."; }
  }
}

}  // namespace lazy_tensors
