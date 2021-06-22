#pragma once

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

#include <functional>
#include <tuple>
#include <vector>

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/core/lib/bfloat16/bfloat16.h"
#include "lazy_tensors/literal_util.h"
#include "lazy_tensors/permutation_util.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {

// Miscellaneous helpers for lowering.
class Helpers {
 public:
  struct MinMax {
    at::Scalar min;
    at::Scalar max;
  };

  struct DynamicReshapeInfo {
    lazy_tensors::Shape output_shape;
    lazy_tensors::int64 dynamic_dimension = -1;
  };

  template <class T>
  static lazy_tensors::Literal ScalarLiteral(T scalar_value,
                                             lazy_tensors::PrimitiveType type) {
    switch (type) {
      case lazy_tensors::PrimitiveType::F64:
        return lazy_tensors::LiteralUtil::CreateR0<double>(scalar_value);
      case lazy_tensors::PrimitiveType::F32:
        return lazy_tensors::LiteralUtil::CreateR0<float>(scalar_value);
      case lazy_tensors::PrimitiveType::BF16:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::bfloat16>(
            static_cast<lazy_tensors::bfloat16>(
                static_cast<float>(scalar_value)));
      case lazy_tensors::PrimitiveType::F16:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::half>(
            static_cast<lazy_tensors::half>(static_cast<float>(scalar_value)));
      case lazy_tensors::PrimitiveType::S64:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::int64>(
            scalar_value);
      case lazy_tensors::PrimitiveType::U64:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::uint64>(
            scalar_value);
      case lazy_tensors::PrimitiveType::S32:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::int32>(
            scalar_value);
      case lazy_tensors::PrimitiveType::U32:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::uint32>(
            scalar_value);
      case lazy_tensors::PrimitiveType::S16:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::int16>(
            scalar_value);
      case lazy_tensors::PrimitiveType::U16:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::uint16>(
            scalar_value);
      case lazy_tensors::PrimitiveType::S8:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::int8>(
            scalar_value);
      case lazy_tensors::PrimitiveType::U8:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::uint8>(
            scalar_value);
      case lazy_tensors::PrimitiveType::PRED:
        return lazy_tensors::LiteralUtil::CreateR0<bool>(scalar_value);
      case lazy_tensors::PrimitiveType::C64:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::complex64>(
            scalar_value);
      case lazy_tensors::PrimitiveType::C128:
        return lazy_tensors::LiteralUtil::CreateR0<lazy_tensors::complex128>(
            scalar_value);
      default:
        return lazy_tensors::LiteralUtil::CreateR0<T>(scalar_value);
    }
  }

  static std::vector<lazy_tensors::int64> GetAllDimensions(size_t rank) {
    return lazy_tensors::util::Iota<lazy_tensors::int64>(rank);
  }

  static std::vector<lazy_tensors::int64> GetAllDimensions(
      const lazy_tensors::Shape& shape) {
    return lazy_tensors::util::Iota<lazy_tensors::int64>(shape.rank());
  }

  static c10::optional<DynamicReshapeInfo> GetDynamicReshapeInfo(
      const lazy_tensors::Shape& input_shape,
      lazy_tensors::Span<const lazy_tensors::int64> output_sizes);

  static lazy_tensors::Shape GetDynamicReshape(
      const lazy_tensors::Shape& input_shape,
      lazy_tensors::Span<const lazy_tensors::int64> output_sizes);

  // Converts an iterable container to a vector of int64's.
  template <typename S>
  static std::vector<lazy_tensors::int64> I64List(const S& input) {
    return lazy_tensors::util::ToVector<lazy_tensors::int64>(input);
  }

  static c10::optional<lazy_tensors::int64> I64Optional(
      c10::optional<int64_t> opt) {
    return opt ? c10::optional<lazy_tensors::int64>(*opt) : c10::nullopt;
  }

  // Creates a set of dimension by dropping the drop_dims ones.
  static std::vector<lazy_tensors::int64> DropDimensions(
      lazy_tensors::Span<const lazy_tensors::int64> sizes,
      lazy_tensors::Span<const lazy_tensors::int64> drop_dims);

  // Get the canonical dimension index in the [0, rank) interval. Negative
  // indices are interpreted as follows: -1 is rank-1, -2 is rank-2 etc.
  static lazy_tensors::int64 GetCanonicalDimensionIndex(
      lazy_tensors::int64 dim, lazy_tensors::int64 rank);

  // Same as above, for multiple dimensions.
  static std::vector<lazy_tensors::int64> GetCanonicalDimensionIndices(
      lazy_tensors::Span<const lazy_tensors::int64> dimensions,
      lazy_tensors::int64 rank);

  // Returns the canonical position in the dim dimension, handling negative
  // values for the position.
  static lazy_tensors::int64 GetCanonicalPosition(
      lazy_tensors::Span<const lazy_tensors::int64> dimensions,
      lazy_tensors::int64 dim, lazy_tensors::int64 pos);

  // Retrieves the dynamic dimension of an input shape, or returns -1 if none.
  static lazy_tensors::int64 GetDynamicDimension(
      const lazy_tensors::Shape& shape);

  // Retrieves type's minimum and maximum values.
  static MinMax MinMaxValues(lazy_tensors::PrimitiveType type);

  // Gathers the input using the order specified by the permutation. For each i,
  // output[i] = input[permutation[i]]. The given permutation must be the same
  // size as the input.
  template <typename Container>
  static std::vector<typename Container::value_type> Permute(
      lazy_tensors::Span<const lazy_tensors::int64> permutation,
      const Container& input) {
    using T = typename Container::value_type;
    LTC_CHECK(input.size() == permutation.size() &&
              lazy_tensors::IsPermutation(permutation))
        << "Invalid permutation specified";
    std::vector<T> output(input.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      output[i] = input[permutation[i]];
    }
    return output;
  }

  // Creates a transposition from the given input and dimensions.
  static std::vector<lazy_tensors::int64> MakeTransposePermutation(
      lazy_tensors::int64 dim0, lazy_tensors::int64 dim1,
      lazy_tensors::int64 rank);

  static lazy_tensors::PrimitiveType PromoteType(
      lazy_tensors::PrimitiveType type1, lazy_tensors::PrimitiveType type2);

  // Calculates the protomoted shape to which the input shapes should be
  // broadcasted for an elementwise operation. The size of the common dimensions
  // (2,3,4 for shape1, and 0,1,2 for shape2) must either match, or either one
  // of the two be 1.
  // Example:
  //   shape1       = [9, 7, 6, 1, 2]
  //   shape2       =       [6, 5, 2]
  //   result_shape = [9, 7, 6, 5, 2]
  static std::vector<lazy_tensors::int64> GetPromotedShape(
      lazy_tensors::Span<const lazy_tensors::int64> shape1_dims,
      lazy_tensors::Span<const lazy_tensors::int64> shape2_dims);

  static lazy_tensors::Shape GetPromotedShape(
      const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2);

  static lazy_tensors::Shape GetPromotedBinaryOpShape(
      const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2);

  template <typename T>
  static lazy_tensors::Literal Range(T start, T end, T step) {
    return lazy_tensors::LiteralUtil::CreateR1<T>(
        lazy_tensors::util::Range<T>(start, end, step));
  }
};

}  // namespace torch_lazy_tensors
