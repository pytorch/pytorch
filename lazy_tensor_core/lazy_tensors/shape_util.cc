#include "lazy_tensors/shape_util.h"

#include "lazy_tensors/core/platform/errors.h"
#include "lazy_tensors/core/platform/hash.h"
#include "lazy_tensors/layout_util.h"

namespace torch {
namespace lazy {

torch::lazy::hash_t SingleShapeHash(const lazy_tensors::Shape& shape, torch::lazy::hash_t seed) {
  for (auto dim : shape.layout().minor_to_major()) {
    seed = HashCombine(seed, (uint64_t)dim);
  }
  for (auto dim : shape.dimensions()) {
    seed = HashCombine(seed, (uint64_t)dim);
  }
  return HashCombine(seed, static_cast<int>(shape.at_element_type()));
}

// The hash is deterministic to enable easier debugging between separate runs.
torch::lazy::hash_t ShapeHash(const lazy_tensors::Shape& shape) {
  torch::lazy::hash_t hash = (uint32_t)0xa5d2d6916;
  lazy_tensors::ShapeUtil::ForEachSubshape(shape,
                             [&](const lazy_tensors::Shape& subshape, const lazy_tensors::ShapeIndex&) {
                               hash = torch::lazy::SingleShapeHash(subshape, hash);
                             });
  return hash;
}


torch::lazy::hash_t Hash(const lazy_tensors::Shape& shape) {
  return ShapeHash(shape);
}
}  // namespace lazy
}  // namespace torch

namespace lazy_tensors {

/* static */ int64 ShapeUtil::TupleElementCount(const Shape& shape) {
  LTC_CHECK(shape.IsTuple()) << shape;
  return shape.tuple_shapes_size();
}

namespace {

// Helper for ForEachSubshape which visits the subshapes of the given shape in
// DFS pre-order starting with the index.
Status ForEachSubshapeHelper(const Shape& shape,
                             const ShapeUtil::StatusVisitorFunction& func,
                             ShapeIndex* index) {
  TF_RETURN_IF_ERROR(func(shape, *index));
  if (shape.IsTuple()) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      // Track the sub-shape position, which can be used by the visitor
      // function.
      index->push_back(i);
      TF_RETURN_IF_ERROR(ForEachSubshapeHelper(
          ShapeUtil::GetTupleElementShape(shape, i), func, index));
      index->pop_back();
    }
  }
  return Status::OK();
}

}  // namespace

void ShapeUtil::ForEachSubshape(const Shape& shape,
                                const VisitorFunction& func) {
  ShapeIndex index;
  // Can safely ignore error since the visitor closure always returns
  // Status::OK().
  ForEachSubshapeHelper(
      shape,
      [&func](const Shape& subshape, const ShapeIndex& index) {
        func(subshape, index);
        return Status::OK();
      },
      &index)
      .IgnoreError();
}

/*static*/ size_t ShapeUtil::Hash(const Shape& shape) {
  using lazy_tensors::hash;
  using lazy_tensors::Hash64Combine;

  size_t hash_value = hash<c10::ScalarType>()(shape.at_element_type());

  if (shape.tuple_shapes().empty()) {
    for (int i = 0; i < shape.dimensions_size(); ++i) {
      hash_value =
          Hash64Combine(hash_value, hash<int64>()(shape.dimensions(i)));
      hash_value = Hash64Combine(hash_value,
                                 hash<bool>()(shape.is_dynamic_dimension(i)));
    }

    hash_value = Hash64Combine(hash_value, LayoutUtil::Hash(shape.layout()));
  } else {
    hash_value = 0;
    for (const Shape& subshape : shape.tuple_shapes()) {
      hash_value = Hash64Combine(hash_value, ShapeUtil::Hash(subshape));
    }
  }

  return hash_value;
}

}  // namespace lazy_tensors
