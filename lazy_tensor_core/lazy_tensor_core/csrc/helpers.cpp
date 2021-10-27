#include "lazy_tensor_core/csrc/helpers.h"

#include <c10/util/Half.h>

#include <limits>

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {

std::vector<int64_t> Helpers::DropDimensions(c10::ArrayRef<int64_t> sizes,
                                             c10::ArrayRef<int64_t> drop_dims) {
  std::vector<int64_t> new_dims;
  size_t drop_index = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (drop_index < drop_dims.size() && i == drop_dims[drop_index]) {
      ++drop_index;
    } else {
      new_dims.push_back(sizes[i]);
    }
  }
  CHECK_EQ(drop_index, drop_dims.size());
  return new_dims;
}

int64_t Helpers::GetCanonicalDimensionIndex(int64_t dim, int64_t rank) {
  int64_t min_shape_dim = -rank;
  int64_t max_shape_dim = rank - 1;
  CHECK(min_shape_dim <= dim && dim <= max_shape_dim)
      << "Value out of range (expected to be in range of [" << min_shape_dim
      << ", " << max_shape_dim << "], but got " << dim << ")";
  int64_t dim_index = dim < 0 ? rank + dim : dim;
  CHECK_GE(dim_index, 0);
  CHECK_LT(dim_index, rank);
  return dim_index;
}

std::vector<int64_t> Helpers::GetCanonicalDimensionIndices(
    c10::ArrayRef<int64_t> dimensions, int64_t rank) {
  std::vector<int64_t> canonical_dim_indices;
  for (int64_t dim : dimensions) {
    canonical_dim_indices.push_back(GetCanonicalDimensionIndex(dim, rank));
  }
  return canonical_dim_indices;
}

int64_t Helpers::GetCanonicalPosition(c10::ArrayRef<int64_t> dimensions,
                                      int64_t dim, int64_t pos) {
  dim = GetCanonicalDimensionIndex(dim, dimensions.size());
  if (pos < 0) {
    pos = GetCanonicalDimensionIndex(pos, dimensions[dim]);
  } else {
    pos = std::min<int64_t>(pos, dimensions[dim]);
  }
  return pos;
}

int64_t Helpers::GetDynamicDimension(const lazy_tensors::Shape& shape) {
  int64_t dynamic_dimension = -1;
  for (int64_t i = 0; i < shape.rank(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      CHECK(dynamic_dimension < 0)
          << "Only one dynamic dimension is supported: " << i << " and "
          << dynamic_dimension << " in " << shape;
      dynamic_dimension = i;
    }
  }
  return dynamic_dimension;
}

Helpers::MinMax Helpers::MinMaxValues(lazy_tensors::PrimitiveType type) {
  switch (type) {
    case lazy_tensors::PrimitiveType::S8:
      return {std::numeric_limits<int8_t>::lowest(),
              std::numeric_limits<int8_t>::max()};
    case lazy_tensors::PrimitiveType::U8:
      return {std::numeric_limits<uint8_t>::lowest(),
              std::numeric_limits<uint8_t>::max()};
    case lazy_tensors::PrimitiveType::S16:
      return {std::numeric_limits<int16_t>::lowest(),
              std::numeric_limits<int16_t>::max()};
    case lazy_tensors::PrimitiveType::U16:
      return {std::numeric_limits<uint16_t>::lowest(),
              std::numeric_limits<uint16_t>::max()};
    case lazy_tensors::PrimitiveType::S32:
      return {static_cast<int64_t>(std::numeric_limits<int32_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<int32_t>::max())};
    case lazy_tensors::PrimitiveType::U32:
      return {static_cast<int64_t>(std::numeric_limits<uint32_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max())};
    case lazy_tensors::PrimitiveType::S64:
      return {static_cast<int64_t>(std::numeric_limits<int64_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<int64_t>::max())};
    case lazy_tensors::PrimitiveType::U64:
      return {static_cast<int64_t>(std::numeric_limits<uint64_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<uint64_t>::max())};
    case lazy_tensors::PrimitiveType::F16:
      return {static_cast<float>(std::numeric_limits<c10::Half>::lowest()),
              static_cast<float>(std::numeric_limits<c10::Half>::max())};
    case lazy_tensors::PrimitiveType::BF16:
    case lazy_tensors::PrimitiveType::F32:
      return {std::numeric_limits<float>::lowest(),
              std::numeric_limits<float>::max()};
    case lazy_tensors::PrimitiveType::F64:
      return {std::numeric_limits<double>::lowest(),
              std::numeric_limits<double>::max()};
    case lazy_tensors::PrimitiveType::PRED:
      return {0, 1};
    default:
      LOG(ERROR) << "Unsupported type " << type;
  }
}

c10::optional<Helpers::DynamicReshapeInfo> Helpers::GetDynamicReshapeInfo(
    const lazy_tensors::Shape& input_shape,
    c10::ArrayRef<int64_t> output_sizes) {
  int64_t input_dynamic_dimension = GetDynamicDimension(input_shape);
  if (input_dynamic_dimension < 0) {
    return c10::nullopt;
  }
  DynamicReshapeInfo info;
  info.output_shape = lazy_tensors::ShapeUtil::MakeShape(
      input_shape.at_element_type(), output_sizes);
  if (info.output_shape.rank() > 0) {
    int64_t size_at_dyndim = 1;
    for (int64_t i = 0; i <= input_dynamic_dimension; ++i) {
      size_at_dyndim *= input_shape.dimensions(i);
    }
    int64_t dynamic_dimension = -1;
    int64_t out_size = 1;
    for (int64_t i = 0; i < output_sizes.size(); ++i) {
      CHECK_LE(out_size,
               size_at_dyndim / input_shape.dimensions(input_dynamic_dimension))
          << "Unable to map dynamic dimension of shape " << input_shape
          << " to output sizes (" << c10::Join(", ", output_sizes) << ")";
      out_size *= output_sizes[i];
      if (out_size >= size_at_dyndim) {
        dynamic_dimension = i;
        break;
      }
    }
    CHECK(dynamic_dimension >= 0)
        << "Unable to map dynamic dimension of shape " << input_shape
        << " to output sizes (" << c10::Join(", ", output_sizes) << ")";
    info.dynamic_dimension = dynamic_dimension;
    info.output_shape.set_dynamic_dimension(info.dynamic_dimension, true);
  }
  return std::move(info);
}

lazy_tensors::Shape Helpers::GetDynamicReshape(
    const lazy_tensors::Shape& input_shape,
    c10::ArrayRef<int64_t> output_sizes) {
  auto info = GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return info->output_shape;
  }
  return lazy_tensors::ShapeUtil::MakeShape(input_shape.at_element_type(),
                                            output_sizes);
}

std::vector<int64_t> Helpers::MakeTransposePermutation(int64_t dim0,
                                                       int64_t dim1,
                                                       int64_t rank) {
  int64_t canonical_dim0 = GetCanonicalDimensionIndex(dim0, rank);
  int64_t canonical_dim1 = GetCanonicalDimensionIndex(dim1, rank);
  auto permute_dims = lazy_tensors::util::Iota<int64_t>(rank);
  std::swap(permute_dims[canonical_dim0], permute_dims[canonical_dim1]);
  return permute_dims;
}

std::vector<int64_t> Helpers::GetPromotedShape(
    c10::ArrayRef<int64_t> shape1_dims, c10::ArrayRef<int64_t> shape2_dims) {
  std::vector<int64_t> dimensions;
  // If the rank of a shape is bigger than then other, fill up the first
  // dimensions with the ones of the bigger.
  // Example:
  //   shape1 = [9, 7, 6, 5, 2]
  //   shape2 =       [6, 1, 2]
  // Insert [9, 7] into the dimensions vector.
  if (shape1_dims.size() > shape2_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape1_dims.begin(),
        shape1_dims.begin() + (shape1_dims.size() - shape2_dims.size()));
  } else if (shape2_dims.size() > shape1_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape2_dims.begin(),
        shape2_dims.begin() + (shape2_dims.size() - shape1_dims.size()));
  }
  // For the common dimensions, they must match, or one of them be 1.
  size_t min_size = std::min(shape1_dims.size(), shape2_dims.size());
  for (int64_t i = 0; i < min_size; ++i) {
    int64_t dim1 = shape1_dims[shape1_dims.size() - min_size + i];
    int64_t dim2 = shape2_dims[shape2_dims.size() - min_size + i];
    CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1)
        << "(" << c10::Join(", ", shape1_dims) << ") and ("
        << c10::Join(", ", shape1_dims) << ")";
    if (dim1 == 0 || dim2 == 0) {
      dimensions.push_back(0);
    } else {
      dimensions.push_back(std::max<int64_t>(dim1, dim2));
    }
  }
  return dimensions;
}

lazy_tensors::Shape Helpers::GetPromotedShape(
    const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2) {
  return lazy_tensors::ShapeUtil::MakeShape(
      shape1.at_element_type(),
      GetPromotedShape(shape1.dimensions(), shape2.dimensions()));
}

lazy_tensors::Shape Helpers::GetPromotedBinaryOpShape(
    const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2) {
  return lazy_tensors::ShapeUtil::MakeShape(
      promoteTypes(shape1.at_element_type(), shape2.at_element_type()),
      GetPromotedShape(shape1.dimensions(), shape2.dimensions()));
}

}  // namespace torch_lazy_tensors
