#include "lazy_tensor_core/csrc/helpers.h"

#include <limits>

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/ltc_logging.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/primitive_util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {

std::vector<lazy_tensors::int64> Helpers::DropDimensions(
    lazy_tensors::Span<const lazy_tensors::int64> sizes,
    lazy_tensors::Span<const lazy_tensors::int64> drop_dims) {
  std::vector<lazy_tensors::int64> new_dims;
  size_t drop_index = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (drop_index < drop_dims.size() && i == drop_dims[drop_index]) {
      ++drop_index;
    } else {
      new_dims.push_back(sizes[i]);
    }
  }
  LTC_CHECK_EQ(drop_index, drop_dims.size());
  return new_dims;
}

lazy_tensors::int64 Helpers::GetCanonicalDimensionIndex(
    lazy_tensors::int64 dim, lazy_tensors::int64 rank) {
  lazy_tensors::int64 min_shape_dim = -rank;
  lazy_tensors::int64 max_shape_dim = rank - 1;
  LTC_CHECK(min_shape_dim <= dim && dim <= max_shape_dim)
      << "Value out of range (expected to be in range of [" << min_shape_dim
      << ", " << max_shape_dim << "], but got " << dim << ")";
  lazy_tensors::int64 dim_index = dim < 0 ? rank + dim : dim;
  LTC_CHECK_GE(dim_index, 0);
  LTC_CHECK_LT(dim_index, rank);
  return dim_index;
}

std::vector<lazy_tensors::int64> Helpers::GetCanonicalDimensionIndices(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::int64 rank) {
  std::vector<lazy_tensors::int64> canonical_dim_indices;
  for (lazy_tensors::int64 dim : dimensions) {
    canonical_dim_indices.push_back(GetCanonicalDimensionIndex(dim, rank));
  }
  return canonical_dim_indices;
}

lazy_tensors::int64 Helpers::GetCanonicalPosition(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::int64 dim, lazy_tensors::int64 pos) {
  dim = GetCanonicalDimensionIndex(dim, dimensions.size());
  if (pos < 0) {
    pos = GetCanonicalDimensionIndex(pos, dimensions[dim]);
  } else {
    pos = std::min<lazy_tensors::int64>(pos, dimensions[dim]);
  }
  return pos;
}

lazy_tensors::int64 Helpers::GetDynamicDimension(
    const lazy_tensors::Shape& shape) {
  lazy_tensors::int64 dynamic_dimension = -1;
  for (lazy_tensors::int64 i = 0; i < shape.rank(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      LTC_CHECK(dynamic_dimension < 0)
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
      return {std::numeric_limits<lazy_tensors::int8>::lowest(),
              std::numeric_limits<lazy_tensors::int8>::max()};
    case lazy_tensors::PrimitiveType::U8:
      return {std::numeric_limits<lazy_tensors::uint8>::lowest(),
              std::numeric_limits<lazy_tensors::uint8>::max()};
    case lazy_tensors::PrimitiveType::S16:
      return {std::numeric_limits<lazy_tensors::int16>::lowest(),
              std::numeric_limits<lazy_tensors::int16>::max()};
    case lazy_tensors::PrimitiveType::U16:
      return {std::numeric_limits<lazy_tensors::uint16>::lowest(),
              std::numeric_limits<lazy_tensors::uint16>::max()};
    case lazy_tensors::PrimitiveType::S32:
      return {static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::int32>::lowest()),
              static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::int32>::max())};
    case lazy_tensors::PrimitiveType::U32:
      return {static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::uint32>::lowest()),
              static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::uint32>::max())};
    case lazy_tensors::PrimitiveType::S64:
      return {static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::int64>::lowest()),
              static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::int64>::max())};
    case lazy_tensors::PrimitiveType::U64:
      return {static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::uint64>::lowest()),
              static_cast<int64_t>(
                  std::numeric_limits<lazy_tensors::uint64>::max())};
    case lazy_tensors::PrimitiveType::F16:
      return {
          static_cast<float>(std::numeric_limits<lazy_tensors::half>::lowest()),
          static_cast<float>(std::numeric_limits<lazy_tensors::half>::max())};
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
      LTC_ERROR() << "Unsupported type " << type;
  }
}

absl::optional<Helpers::DynamicReshapeInfo> Helpers::GetDynamicReshapeInfo(
    const lazy_tensors::Shape& input_shape,
    lazy_tensors::Span<const lazy_tensors::int64> output_sizes) {
  lazy_tensors::int64 input_dynamic_dimension =
      GetDynamicDimension(input_shape);
  if (input_dynamic_dimension < 0) {
    return absl::nullopt;
  }
  DynamicReshapeInfo info;
  info.output_shape = lazy_tensors::ShapeUtil::MakeShape(
      input_shape.element_type(), output_sizes);
  if (info.output_shape.rank() > 0) {
    lazy_tensors::int64 size_at_dyndim = 1;
    for (lazy_tensors::int64 i = 0; i <= input_dynamic_dimension; ++i) {
      size_at_dyndim *= input_shape.dimensions(i);
    }
    lazy_tensors::int64 dynamic_dimension = -1;
    lazy_tensors::int64 out_size = 1;
    for (lazy_tensors::int64 i = 0; i < output_sizes.size(); ++i) {
      LTC_CHECK_LE(out_size, size_at_dyndim / input_shape.dimensions(
                                                  input_dynamic_dimension))
          << "Unable to map dynamic dimension of shape " << input_shape
          << " to output sizes (" << absl::StrJoin(output_sizes, ", ") << ")";
      out_size *= output_sizes[i];
      if (out_size >= size_at_dyndim) {
        dynamic_dimension = i;
        break;
      }
    }
    LTC_CHECK(dynamic_dimension >= 0)
        << "Unable to map dynamic dimension of shape " << input_shape
        << " to output sizes (" << absl::StrJoin(output_sizes, ", ") << ")";
    info.dynamic_dimension = dynamic_dimension;
    info.output_shape.set_dynamic_dimension(info.dynamic_dimension, true);
  }
  return std::move(info);
}

lazy_tensors::Shape Helpers::GetDynamicReshape(
    const lazy_tensors::Shape& input_shape,
    lazy_tensors::Span<const lazy_tensors::int64> output_sizes) {
  auto info = GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return info->output_shape;
  }
  return lazy_tensors::ShapeUtil::MakeShape(input_shape.element_type(),
                                            output_sizes);
}

std::vector<lazy_tensors::int64> Helpers::MakeTransposePermutation(
    lazy_tensors::int64 dim0, lazy_tensors::int64 dim1,
    lazy_tensors::int64 rank) {
  lazy_tensors::int64 canonical_dim0 = GetCanonicalDimensionIndex(dim0, rank);
  lazy_tensors::int64 canonical_dim1 = GetCanonicalDimensionIndex(dim1, rank);
  auto permute_dims = lazy_tensors::util::Iota<lazy_tensors::int64>(rank);
  std::swap(permute_dims[canonical_dim0], permute_dims[canonical_dim1]);
  return permute_dims;
}

lazy_tensors::PrimitiveType Helpers::PromoteType(
    lazy_tensors::PrimitiveType type1, lazy_tensors::PrimitiveType type2) {
  if (type1 == type2) {
    return type1;
  }
  lazy_tensors::int64 size1 =
      lazy_tensors::ShapeUtil::ByteSizeOfPrimitiveType(type1);
  lazy_tensors::int64 size2 =
      lazy_tensors::ShapeUtil::ByteSizeOfPrimitiveType(type2);
  if (lazy_tensors::primitive_util::IsComplexType(type1)) {
    return (!lazy_tensors::primitive_util::IsComplexType(type2) ||
            size1 >= size2)
               ? type1
               : type2;
  }
  if (lazy_tensors::primitive_util::IsComplexType(type2)) {
    return type2;
  }
  if (lazy_tensors::primitive_util::IsFloatingPointType(type1)) {
    return (!lazy_tensors::primitive_util::IsFloatingPointType(type2) ||
            size1 >= size2)
               ? type1
               : type2;
  }
  if (lazy_tensors::primitive_util::IsFloatingPointType(type2) ||
      size2 > size1) {
    return type2;
  }
  if (lazy_tensors::primitive_util::IsIntegralType(type1) &&
      lazy_tensors::primitive_util::IsIntegralType(type2)) {
    if (size1 > size2) {
      return type1;
    }
    if (size2 > size1) {
      return type2;
    }
    // At this point, they are not the same type, they are both integers, and
    // they have the same size. One of them must be unsigned and the other
    // signed, convert to unsigned.
    return lazy_tensors::primitive_util::UnsignedIntegralTypeForBitWidth(
        lazy_tensors::primitive_util::BitWidth(type1));
  }
  if (type1 == lazy_tensors::PrimitiveType::PRED) {
    return type2;
  }
  if (type2 == lazy_tensors::PrimitiveType::PRED) {
    return type1;
  }
  // If nothing matches the above logic, first operand wins.
  return type1;
}

std::vector<lazy_tensors::int64> Helpers::GetPromotedShape(
    lazy_tensors::Span<const lazy_tensors::int64> shape1_dims,
    lazy_tensors::Span<const lazy_tensors::int64> shape2_dims) {
  std::vector<lazy_tensors::int64> dimensions;
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
  for (lazy_tensors::int64 i = 0; i < min_size; ++i) {
    lazy_tensors::int64 dim1 = shape1_dims[shape1_dims.size() - min_size + i];
    lazy_tensors::int64 dim2 = shape2_dims[shape2_dims.size() - min_size + i];
    LTC_CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1)
        << "(" << absl::StrJoin(shape1_dims, ", ") << ") and ("
        << absl::StrJoin(shape1_dims, ", ") << ")";
    if (dim1 == 0 || dim2 == 0) {
      dimensions.push_back(0);
    } else {
      dimensions.push_back(std::max<lazy_tensors::int64>(dim1, dim2));
    }
  }
  return dimensions;
}

lazy_tensors::Shape Helpers::GetPromotedShape(
    const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2) {
  return lazy_tensors::ShapeUtil::MakeShape(
      shape1.element_type(),
      GetPromotedShape(shape1.dimensions(), shape2.dimensions()));
}

lazy_tensors::Shape Helpers::GetPromotedBinaryOpShape(
    const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2) {
  return lazy_tensors::ShapeUtil::MakeShape(
      PromoteType(shape1.element_type(), shape2.element_type()),
      GetPromotedShape(shape1.dimensions(), shape2.dimensions()));
}

}  // namespace torch_lazy_tensors
