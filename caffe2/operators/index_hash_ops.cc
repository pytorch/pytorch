#include "caffe2/operators/index_hash_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(IndexHash, IndexHashOp<CPUContext>);

OPERATOR_SCHEMA(IndexHash)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This operator translates a list of indices into a list of hashed indices.
A seed can be fed as an argument to change the behavior of the hash function.
If a modulo is specified, all the hashed indices will be modulo the
specified number. All input and output indices are enforced to be positive.
)DOC")
    .Input(0, "Indices", "Input feature indices.")
    .Output(0, "HashedIndices", "Hashed feature indices.")
    .AllowOneToOneInplace()
    .Arg("seed", "seed for the hash function")
    .Arg("modulo", "must be > 0, hashed ids will be modulo this number")
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      std::vector<int64_t> output_dims = GetDimsVector(in[0]);
      out[0] = CreateTensorShape(output_dims, in[0].data_type());
      return out;
    });

SHOULD_NOT_DO_GRADIENT(IndexHash);

} // namespace
} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    IndexHash,
    "_caffe2::IndexHash(Tensor indices, int seed, int modulo) -> Tensor hashed_indices",
    caffe2::IndexHashOp<caffe2::CPUContext>);
