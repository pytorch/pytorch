#pragma once

#include <c10/core/Scalar.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/util.h>

#include <complex>
#include <functional>
#include <tuple>
#include <vector>

namespace torch {
namespace lazy {

// Converts an iterable container to a vector of int64's.
template <typename S>
static std::vector<int64_t> ToI64Vector(const S& input) {
  return ToVector<int64_t>(input);
}

// Creates a set of dimension by dropping the drop_dims ones.
std::vector<int64_t> DropDimensions(
    c10::ArrayRef<int64_t> sizes,
    c10::ArrayRef<int64_t> drop_dims);

// Get the canonical dimension index in the [0, rank) interval. Negative
// indices are interpreted as follows: -1 is rank-1, -2 is rank-2 etc.
int64_t GetCanonicalDimensionIndex(int64_t dim, int64_t rank);

// Same as above, for multiple dimensions.
std::vector<int64_t> GetCanonicalDimensionIndices(
    c10::ArrayRef<int64_t> dimensions,
    int64_t rank);

// Returns the canonical position in the dim dimension, handling negative
// values for the position.
int64_t GetCanonicalPosition(
    c10::ArrayRef<int64_t> dimensions,
    int64_t dim,
    int64_t pos);

// Creates a transposition from the given input and dimensions.
std::vector<int64_t> MakeTransposePermutation(
    int64_t dim0,
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
std::vector<int64_t> GetPromotedShape(
    c10::ArrayRef<int64_t> shape1_dims,
    c10::ArrayRef<int64_t> shape2_dims);

torch::lazy::Shape GetPromotedBinaryOpShape(
    const torch::lazy::Shape& shape1,
    const torch::lazy::Shape& shape2);

std::vector<std::string> StrSplit(c10::string_view text, char delim);

} // namespace lazy
} // namespace torch
