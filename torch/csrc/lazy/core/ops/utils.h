#include <vector>

#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

namespace torch {
namespace lazy {

bool StrideIsSupported(c10::ArrayRef<int64_t> stride);

std::vector<int64_t> GetArrayStridePermutation(c10::ArrayRef<int64_t> stride);

Shape MakeDiagonalShape(
    const Shape& shape,
    int64_t offset,
    int64_t dim1,
    int64_t dim2);

Shape MakePermuteShape(
    const Shape& source_shape,
    c10::ArrayRef<int64_t> permutation);

Shape MakeSelectShape(
    const Shape& shape,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t stride);

int64_t GetStride(int64_t start, int64_t end, int64_t stride);

std::vector<int64_t> BuildSqueezedDimensions(c10::ArrayRef<int64_t> dimensions,
                                             int64_t squeeze_dim);

std::vector<int64_t> BuildUnsqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim);

} // namespace lazy
} // namespace torch
