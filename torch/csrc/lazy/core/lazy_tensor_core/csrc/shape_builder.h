#pragma once

#include <vector>

#include "lazy_tensors/shape.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {

class ShapeBuilder {
 public:
  explicit ShapeBuilder(lazy_tensors::PrimitiveType type) : type_(type) {}

  ShapeBuilder& Add(const lazy_tensors::Shape& shape, lazy_tensors::int64 dim);

  ShapeBuilder& Add(const lazy_tensors::Shape& shape,
                    lazy_tensors::Span<const lazy_tensors::int64> dimensions);

  ShapeBuilder& Add(lazy_tensors::int64 size);

  lazy_tensors::Shape Build() const;

 private:
  struct ShapeDim {
    const lazy_tensors::Shape* shape = nullptr;
    lazy_tensors::int64 dim_or_size = -1;
  };

  lazy_tensors::PrimitiveType type_;
  std::vector<ShapeDim> dims_;
};

}  // namespace torch_lazy_tensors
