#pragma once
#include <c10/core/Scalar.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>

#include <complex>
#include <functional>
#include <tuple>
#include <vector>

#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/literal_util.h"
#include "lazy_tensors/permutation_util.h"

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
    int64_t dynamic_dimension = -1;
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
        return lazy_tensors::LiteralUtil::CreateR0<c10::BFloat16>(
            static_cast<c10::BFloat16>(static_cast<float>(scalar_value)));
      case lazy_tensors::PrimitiveType::F16:
        return lazy_tensors::LiteralUtil::CreateR0<c10::Half>(
            static_cast<c10::Half>(static_cast<float>(scalar_value)));
      case lazy_tensors::PrimitiveType::S64:
        return lazy_tensors::LiteralUtil::CreateR0<int64_t>(scalar_value);
      case lazy_tensors::PrimitiveType::U64:
        return lazy_tensors::LiteralUtil::CreateR0<uint64_t>(scalar_value);
      case lazy_tensors::PrimitiveType::S32:
        return lazy_tensors::LiteralUtil::CreateR0<int32_t>(scalar_value);
      case lazy_tensors::PrimitiveType::U32:
        return lazy_tensors::LiteralUtil::CreateR0<uint32_t>(scalar_value);
      case lazy_tensors::PrimitiveType::S16:
        return lazy_tensors::LiteralUtil::CreateR0<int16_t>(scalar_value);
      case lazy_tensors::PrimitiveType::U16:
        return lazy_tensors::LiteralUtil::CreateR0<uint16_t>(scalar_value);
      case lazy_tensors::PrimitiveType::S8:
        return lazy_tensors::LiteralUtil::CreateR0<int8_t>(scalar_value);
      case lazy_tensors::PrimitiveType::U8:
        return lazy_tensors::LiteralUtil::CreateR0<uint8_t>(scalar_value);
      case lazy_tensors::PrimitiveType::PRED:
        return lazy_tensors::LiteralUtil::CreateR0<bool>(scalar_value);
      case lazy_tensors::PrimitiveType::C64:
        return lazy_tensors::LiteralUtil::CreateR0<std::complex<float>>(
            scalar_value);
      case lazy_tensors::PrimitiveType::C128:
        return lazy_tensors::LiteralUtil::CreateR0<std::complex<double>>(
            scalar_value);
      default:
        return lazy_tensors::LiteralUtil::CreateR0<T>(scalar_value);
    }
  }

  static std::vector<int64_t> GetAllDimensions(size_t rank) {
    return lazy_tensors::util::Iota<int64_t>(rank);
  }

  static std::vector<int64_t> GetAllDimensions(
      const lazy_tensors::Shape& shape) {
    return lazy_tensors::util::Iota<int64_t>(shape.rank());
  }

  // Converts an iterable container to a vector of int64's.
  template <typename S>
  static std::vector<int64_t> I64List(const S& input) {
    return lazy_tensors::util::ToVector<int64_t>(input);
  }

  static c10::optional<int64_t> I64Optional(c10::optional<int64_t> opt) {
    return opt ? c10::optional<int64_t>(*opt) : c10::nullopt;
  }

  // Creates a set of dimension by dropping the drop_dims ones.
  static std::vector<int64_t> DropDimensions(c10::ArrayRef<int64_t> sizes,
                                             c10::ArrayRef<int64_t> drop_dims);

  // Get the canonical dimension index in the [0, rank) interval. Negative
  // indices are interpreted as follows: -1 is rank-1, -2 is rank-2 etc.
  static int64_t GetCanonicalDimensionIndex(int64_t dim, int64_t rank);

  // Same as above, for multiple dimensions.
  static std::vector<int64_t> GetCanonicalDimensionIndices(
      c10::ArrayRef<int64_t> dimensions, int64_t rank);

  // Returns the canonical position in the dim dimension, handling negative
  // values for the position.
  static int64_t GetCanonicalPosition(c10::ArrayRef<int64_t> dimensions,
                                      int64_t dim, int64_t pos);


  // Retrieves type's minimum and maximum values.
  static MinMax MinMaxValues(lazy_tensors::PrimitiveType type);

  // Gathers the input using the order specified by the permutation. For each i,
  // output[i] = input[permutation[i]]. The given permutation must be the same
  // size as the input.
  template <typename Container>
  static std::vector<typename Container::value_type> Permute(
      c10::ArrayRef<int64_t> permutation, const Container& input) {
    using T = typename Container::value_type;
    CHECK(input.size() == permutation.size() &&
          lazy_tensors::IsPermutation(permutation))
        << "Invalid permutation specified";
    std::vector<T> output(input.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      output[i] = input[permutation[i]];
    }
    return output;
  }

  // Creates a transposition from the given input and dimensions.
  static std::vector<int64_t> MakeTransposePermutation(int64_t dim0,
                                                       int64_t dim1,
                                                       int64_t rank);

  // Calculates the protomoted shape to which the input shapes should be
  // broadcasted for an elementwise operation. The size of the common dimensions
  // (2,3,4 for shape1, and 0,1,2 for shape2) must either match, or either one
  // of the two be 1.
  // Example:
  //   shape1       = [9, 7, 6, 1, 2]
  //   shape2       =       [6, 5, 2]
  //   result_shape = [9, 7, 6, 5, 2]
  static std::vector<int64_t> GetPromotedShape(
      c10::ArrayRef<int64_t> shape1_dims, c10::ArrayRef<int64_t> shape2_dims);

  static lazy_tensors::Shape GetPromotedShape(
      const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2);

  static lazy_tensors::Shape GetPromotedBinaryOpShape(
      const lazy_tensors::Shape& shape1, const lazy_tensors::Shape& shape2);

};

}  // namespace torch_lazy_tensors
