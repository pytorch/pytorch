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
If a modulo is speicified, all the hashed indices will be modulo the
speicified number. All input and output indices are enforced to be positive.
)DOC")
    .Input(0, "Indices", "Input feature indices.")
    .Output(0, "HashedIndices", "Hashed feature indices.")
    .Arg("seed", "seed for the hash function")
    .Arg("modulo", "must be > 0, hashed ids will be modulo this number");

SHOULD_NOT_DO_GRADIENT(IndexHash);

} // namespace
} // namespace caffe2
