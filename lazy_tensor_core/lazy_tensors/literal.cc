#include "lazy_tensors/literal.h"

#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/core/platform/hash.h"
#include "lazy_tensors/shape_util.h"

namespace lazy_tensors {

Literal::Literal(const Shape& shape) : shape_(shape) {
  std::vector<int64_t> dimensions = util::ToVector<int64_t>(shape.dimensions());
  at::TensorOptions options(shape.at_element_type());
  value_ = at::empty(dimensions, options);
}

const Shape& Literal::shape() const { return shape_; }

size_t Literal::Hash() const {
  size_t hash_value = ShapeUtil::Hash(shape());

  ShapeUtil::ForEachSubshape(
      shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }

        hash_value = Hash64Combine(
            hash_value, Hash64(static_cast<const char*>(untyped_data(index)),
                               size_bytes(index)));
      });

  return hash_value;
}

}  // namespace lazy_tensors
