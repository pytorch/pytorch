#include "caffe2/operators/dense_vector_to_id_list_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(DenseVectorToIdList, DenseVectorToIdListOp<CPUContext>);

OPERATOR_SCHEMA(DenseVectorToIdList)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
DenseVectorToIdList: Convert a blob with dense feature into a ID_LIST.

An ID_LIST is a list of IDs (may be ints, often longs) that represents a single
feature. As described in https://caffe2.ai/docs/sparse-operations.html, a batch
of ID_LIST examples is represented as a pair of lengths and values where the
`lengths` (int32) segment the `values` or ids (int32/int64) into examples.

Input is a single blob where the first dimension is the batch size and the
second dimension is the length of dense vectors. This operator produces a
ID_LIST where out_values are the indices of non-zero entries
and out_lengths are the number of non-zeros entries in each row.

)DOC")
    .Input(0, "values", "A data blob of dense vectors")
    .Output(0, "out_lengths", "Lengths of the sparse feature")
    .Output(1, "out_values", "Values of the sparse feature");
NO_GRADIENT(DenseVectorToIdList);
} // namespace
} // namespace caffe2
