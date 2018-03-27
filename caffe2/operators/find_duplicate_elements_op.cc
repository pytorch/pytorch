#include "caffe2/operators/find_duplicate_elements_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(
    FindDuplicateElements,
    FindDuplicateElementsOp<CPUContext>);

OPERATOR_SCHEMA(FindDuplicateElements)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Shrink the data tensor by removing data blocks with given zero-based indices in
the outermost dimension of the tensor. Indices are not assumed in any order or
unique but with the range [0, blocks_size). Indices could be empty.
  )DOC")
    .Input(0, "data", "a 1-D tensor.")
    .Output(
        0,
        "indices",
        "indices of duplicate elements in data, excluding first occurrences.");

SHOULD_NOT_DO_GRADIENT(FindDuplicateElements);
} // namespace
} // namespace caffe2
