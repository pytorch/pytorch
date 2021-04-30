#include "lazy_tensors/shape_util.h"

#include "lazy_tensors/core/platform/errors.h"

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
  ForEachSubshapeHelper(
      shape,
      [&func](const Shape& subshape, const ShapeIndex& index) {
        func(subshape, index);
        return Status::OK();
      },
      &index)
      .IgnoreError();
}

}  // namespace lazy_tensors
