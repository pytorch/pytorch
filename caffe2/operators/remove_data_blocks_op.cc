#include "caffe2/operators/remove_data_blocks_op.h"

namespace caffe2 {
namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(RemoveDataBlocks, RemoveDataBlocksOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(RemoveDataBlocks)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Shrink the data tensor by removing data blocks with given zero-based indices in
the outermost dimension of the tensor. Indices are not assumed in any order or
unique but with the range [0, blocks_size). Indices could be empty.
  )DOC")
    .Input(0, "data", "a N-D data tensor, N >= 1")
    .Input(1, "indices", "zero-based indices of blocks to be removed")
    .Output(
        0,
        "shrunk data",
        "data after removing data blocks indexed by 'indices'");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(RemoveDataBlocks);
} // namespace
} // namespace caffe2
