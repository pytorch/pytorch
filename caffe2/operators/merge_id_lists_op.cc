#include "caffe2/operators/merge_id_lists_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(MergeIdLists, MergeIdListsOp<CPUContext>);

OPERATOR_SCHEMA(MergeIdLists)
    .NumInputs([](int n) { return (n > 0 && n % 2 == 0); })
    .NumOutputs(2)
    .SetDoc(R"DOC(
MergeIdLists: Merge multiple ID_LISTs into a single ID_LIST.

An ID_LIST is a list of IDs (may be ints, often longs) that represents a single
feature. As described in https://caffe2.ai/docs/sparse-operations.html, a batch
of ID_LIST examples is represented as a pair of lengths and values where the
`lengths` (int32) segment the `values` or ids (int32/int64) into examples.

Given multiple inputs of the form lengths_0, values_0, lengths_1, values_1, ...
which correspond to lengths and values of ID_LISTs of different features, this
operator produces a merged ID_LIST that combines the ID_LIST features. The
final merged output is described by a lengths and values vector.

WARNING: The merge makes no guarantee about the relative order of ID_LISTs
within a batch. This can be an issue if ID_LIST are order sensitive.
)DOC")
    .Input(0, "lengths_0", "Lengths of the ID_LISTs batch for first feature")
    .Input(1, "values_0", "Values of the ID_LISTs batch for first feature")
    .Output(0, "merged_lengths", "Lengths of the merged ID_LISTs batch")
    .Output(1, "merged_values", "Values of the merged ID_LISTs batch");
NO_GRADIENT(MergeIdLists);
}
}
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    MergeIdLists,
    "_caffe2::MergeIdLists(Tensor[] lengths_and_values) -> (Tensor merged_lengths, Tensor merged_values)",
    caffe2::MergeIdListsOp<caffe2::CPUContext>);
