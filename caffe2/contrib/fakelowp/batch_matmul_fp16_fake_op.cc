#include "batch_matmul_fp16_fake_op.h"

#include "caffe2/core/operator_schema.h"

namespace caffe2 {

vector<TensorShape> TensorInferenceForBatchMatMul(
    const OperatorDef& def,
    const vector<TensorShape>& in);
OpSchema::Cost CostInferenceForBatchMatMul(
    const OperatorDef& def,
    const vector<TensorShape>& in);

REGISTER_CPU_OPERATOR(BatchMatMulFP16Fake, BatchMatMulFP16FakeOp<CPUContext>);

OPERATOR_SCHEMA(BatchMatMulFP16Fake)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Batch Matrix multiplication Yi = Ai * Bi, where A has shape (dim0, dim1, ... M, K),
B has shape (dim0, dim1, ... K, N), Y has shape (dim0, dim1, ... M, N) and i ranges
from 0 to (dim0 * dim1 ...) - 1. rank(A) == rank(B) >= 2. In case of A and B being
two diemnsional, it behaves like normal matrix multiplication.
)DOC")
    .Input(0, "A", "tensor of shape (dim0, dim1 ... M, K)")
    .Input(1, "B", "tensor of shpae (dim0, dim2 ... K, N)")
    .Output(0, "Y", "tensor of shape (dim0, dim1 ... M, N)")
    .Arg(
        "trans_a",
        "Pass 1 to transpose the last two dimensions of A before "
        "doing multiplication")
    .Arg(
        "trans_b",
        "Pass 1 to transpose the last two dimensions of B before "
        "doing multiplication")
    .Arg(
        "broadcast",
        "Pass 1 to allow broadcasting of dimensions. Behavior is the same as numpy.matmul. Gradient is currently not supported when running in broadcast mode.")
    .TensorInferenceFunction(TensorInferenceForBatchMatMul)
    .CostInferenceFunction(
        OpSchema::CostInferenceFunctionType(CostInferenceForBatchMatMul))
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(
    BatchMatMulFP16Acc16Fake,
    BatchMatMulFP16FakeOp<
        CPUContext,
        DefaultEngine,
        true /*use custom fp16 gemm acc16*/,
        false /*not using temp accmulator*/,
        false /*use math fp16 gemm acc 32*/>);

OPERATOR_SCHEMA(BatchMatMulFP16Acc16Fake).NumInputs(2).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    BatchMatMulFP16Acc32Fake,
    BatchMatMulFP16FakeOp<
        CPUContext,
        DefaultEngine,
        false /*use custom fp16 gemm acc16*/,
        false /*not using temp accmulator*/,
        true /*use custom fp16 gemm acc32*/>);

OPERATOR_SCHEMA(BatchMatMulFP16Acc32Fake).NumInputs(2).NumOutputs(1);

} // namespace caffe2
