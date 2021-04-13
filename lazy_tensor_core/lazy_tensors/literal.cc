#include "lazy_tensors/literal.h"

#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace lazy_tensors {

Literal::Literal(const Shape& shape) : shape_(shape) {
  std::vector<int64_t> dimensions = util::ToVector<int64_t>(shape.dimensions());
  at::TensorOptions options(
      static_cast<at::ScalarType>(PrimitiveToScalarType(shape.element_type())));
  value_ = at::empty(dimensions, options);
}

const Shape& Literal::shape() const { return shape_; }

}  // namespace lazy_tensors
