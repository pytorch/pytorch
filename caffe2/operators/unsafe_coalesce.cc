#include "caffe2/operators/unsafe_coalesce.h"

namespace caffe2 {

OPERATOR_SCHEMA(UnsafeCoalesce)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs + 1 == outputs;
    })
    .AllowInplace([](int input, int output) { return input == output; })
    .SetDoc(R"DOC(
Coalesce the N inputs into N outputs and a single coalesced output blob.
This allows operations that operate over multiple small kernels (e.g.
biases in a deep CNN) to be coalesced into a single larger operation,
amortizing the kernel launch overhead, synchronization costs for
distributed computation, etc.
The operator:
- computes the total size of the coalesced blob by summing the input sizes
- allocates the coalesced output blob as the total size
- copies the input vectors into the coalesced blob, at the correct offset.
- aliases each Output(i) to- point into the coalesced blob, at the corresponding offset for Input(i).
This is 'unsafe' as the output vectors are aliased, so use with
caution.
)DOC");

REGISTER_CPU_OPERATOR(UnsafeCoalesce, UnsafeCoalesceOp<CPUContext>);

} // namespace caffe2
