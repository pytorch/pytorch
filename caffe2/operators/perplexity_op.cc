#include "caffe2/operators/perplexity_op.h"

namespace caffe2 {

template <>
bool PerplexityOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);

  TORCH_DCHECK_EQ(X.dim(), 1);
  int N = X.dim32(0);

  auto* Y = Output(0, vector<int64_t>(), at::dtype<float>());
  const auto* Xdata = X.data<float>();

  float perplexity = 1.0;
  for (int i = 0; i < N; ++i) {
    perplexity *= pow(Xdata[i], -1.0/N);
  }
  *(Y->template mutable_data<float>()) = perplexity;
  return true;
}

REGISTER_CPU_OPERATOR(Perplexity, PerplexityOp<float, CPUContext>);

OPERATOR_SCHEMA(Perplexity).NumInputs(1).NumOutputs(1)
.SetDoc(R"DOC(
Perplexity calculates how well a probability distribution predicts a sample.
Perplexity takes a 1-D tensor containing a batch of probabilities. Each value
in the tensor belongs to a different sample and represents the probability of
the model predicting the true label for that sample. The operator returns a
single (float) perplexity value for the batch.
)DOC")
.Input(0, "probabilities", "The input data as Tensor. It contains a batch of"
       "true label or target probabilities")
.Output(0, "output", "The output- a single (float) perplexity value for the "
        "batch");

SHOULD_NOT_DO_GRADIENT(Perplexity);
}  // namespace caffe2
