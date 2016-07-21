#include "caffe2/operators/h_softmax_op.h"

namespace caffe2 {

template <>
float HSoftmaxOp<float, CPUContext>::RunForwardSingle(const float* X,
  const float* W, const float* b, int target, float* int_output,
  const float* bias_multiplier, int dim_out, int dim_in,
  int& int_output_offset) {

  // W * x
  float* fc_output_data = int_output + int_output_offset;

  math::Gemm<float, CPUContext>(CblasNoTrans, CblasTrans, 1, dim_out, dim_in, 1,
    X, W, 0, fc_output_data, &context_);
  math::Gemv<float, CPUContext>(CblasNoTrans, dim_out, 1, 1,
    b, bias_multiplier, 1, fc_output_data, &context_);

  int_output_offset += dim_out;

  //Softmax
  float* softmax_output_data = int_output + int_output_offset;

  if (scale_.size() != 1) {
    scale_.Resize(vector<TIndex>{1});
  }
  if (sum_multiplier_.size() != dim_out) {
    sum_multiplier_.Resize(vector<TIndex>{dim_out});
    math::Set<float, CPUContext>(dim_out, 1.f,
      sum_multiplier_.mutable_data<float>(), &context_);
  }
  math::RowwiseMax<float, CPUContext>(1, dim_out, fc_output_data,
    scale_.mutable_data<float>(), &context_);

  // Put the intermediate result X - max(X) into Y
  context_.template Copy<float, CPUContext, CPUContext>(dim_out, fc_output_data,
    softmax_output_data);
  // Subtract the scale
  math::Gemv<float, CPUContext>(CblasNoTrans, dim_out, 1, -1,
    sum_multiplier_.data<float>(), scale_.data<float>(), 1, softmax_output_data,
    &context_);

  // Exponentiation
  math::Exp<float, CPUContext>(dim_out, softmax_output_data,
    softmax_output_data, &context_);
  math::Gemv<float, CPUContext>(CblasNoTrans, 1, dim_out, 1,
    softmax_output_data, sum_multiplier_.data<float>(), 0,
    scale_.mutable_data<float>(), &context_);

  // Do division
  const float scale = *scale_.data<float>();
  for (int j = 0; j < dim_out; ++j) {
    softmax_output_data[j] /= scale;
  }

  int_output_offset += dim_out;

  //Return cross entropy loss
  return -log(std::max(softmax_output_data[target], kLOG_THRESHOLD()));
}

// Implementation for the CPU context.
template <>
bool HSoftmaxOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  const auto& W = Input(1);
  const auto& b = Input(2);
  auto& label = Input(3);
  auto* Y = Output(0);
  auto* intermediate_output = Output(1);

  // Batch size
  int M = X.ndim() > 1 ? X.dim32(0) : 1;
  // Input feature dimension
  int K = X.size() / M;
  CHECK_GE(W.ndim(), 2);//N*K
  CHECK_EQ(b.ndim(), 1);//N
  CHECK_EQ(K, W.size() / (W.dim32(0)));
  // Sum of output dimensions of all hierarchy nodes
  int N = W.dim32(0);
  CHECK_EQ(N, b.dim32(0));
  Y->Resize(vector<TIndex>{M});
  auto* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(M, 0.f, Ydata, &context_);
  const auto* labeldata = label.data<int>();

  std::unordered_map<int, PathProto> hierarchy = getHierarchyForLabels(M,
    labeldata, hierarchy_);
  int int_output_size = getIntermediateOutputSize(labeldata, M, hierarchy);
  intermediate_output->Resize(vector<TIndex>{int_output_size});
  float * int_output_data = intermediate_output->mutable_data<float>();
  int int_output_offset = 0;

  if (bias_multiplier_.size() != M) {
    bias_multiplier_.Resize(vector<TIndex>{M});
    math::Set<float, CPUContext>(M, static_cast<float>(1),
        bias_multiplier_.mutable_data<float>(), &context_);
  }

  for (int sample = 0; sample < M; ++sample) {
    int word_id = labeldata[sample];
    const PathProto& path = hierarchy[word_id];
    for (const PathNodeProto& node : path.path_nodes()) {
      //Offset of node's weight matrix in W
      int w_offset = node.index();
      //Number of output dimensions in node's weight matrix
      int w_length = node.length();
      int target = node.target();
      //Adding log probabilities
      Ydata[sample] += RunForwardSingle(X.data<float>() + sample*K,
        W.data<float>() + w_offset*K, b.data<float>() + w_offset, target,
        int_output_data, bias_multiplier_.data<float>()+sample, w_length, K,
        int_output_offset);
    }
  }
  return true;
}

template <>
void HSoftmaxGradientOp<float, CPUContext>::RunBackwardSingle(const float* X,
  const float* dY, const float* W, int target,
  const float* int_output, float* dX, float* dW, float* db, float* dint_output,
  int dim_in, int dim_out, int& int_output_offset) {

  //Cross entropy
  // dX_entropy is the dX for the cross entropy layer
  float* dX_entropy = dint_output + int_output_offset - dim_out;
  // X_entropy is the X for the cross entropy layer and Y for the softmax layer
  const float* X_entropy = int_output + int_output_offset - dim_out;

  math::Set<float, CPUContext>(dim_out, 0.f, dX_entropy, &context_);
  dX_entropy[target] = - (*dY) / std::max(X_entropy[target], kLOG_THRESHOLD());

  int_output_offset -= dim_out;

  //Softmax
  if (scale_.size() != 1) {
    scale_.Resize(vector<TIndex>{1});
  }
  float* scaledata = scale_.mutable_data<float>();

  if (sum_multiplier_.size() != dim_out) {
    sum_multiplier_.Resize(vector<TIndex>{dim_out});
    math::Set<float, CPUContext>(dim_out, 1.f,
      sum_multiplier_.mutable_data<float>(), &context_);
  }

  float* dX_softmax = dint_output + int_output_offset - dim_out;
  context_.Copy<float, CPUContext, CPUContext>(dim_out, dX_entropy, dX_softmax);

  math::Dot<float, CPUContext>(dim_out, X_entropy, dX_entropy, scaledata,
    &context_);
  math::Gemv<float, CPUContext>(CblasTrans, 1, dim_out, -1,
    sum_multiplier_.data<float>(), scaledata , 1, dX_softmax, &context_);
  math::Mul<float, CPUContext>(dim_out, dX_softmax, X_entropy, dX_softmax,
    &context_);

  int_output_offset -= dim_out;

  //FC
  if (bias_multiplier_.size() != 1) {
    // If the helper bias multiplier has not been created, reshape and fill
    // it with 1
    bias_multiplier_.Resize(vector<TIndex>{1});
    math::Set<float, CPUContext>(1, static_cast<float>(1),
        bias_multiplier_.template mutable_data<float>(), &context_);
  }

  // Compute dW and add incrementally
  // dW = dW + dX_softmax'*X
  math::Gemm<float, CPUContext>(CblasTrans, CblasNoTrans, dim_out, dim_in, 1, 1,
    dX_softmax, X, 1, dW, &context_);

  // Compute dB and add incrementally
  // db = db + dX_softmax*bias_multiplier_
  math::Gemv<float, CPUContext>(CblasTrans, 1, dim_out, 1, dX_softmax,
    bias_multiplier_.template data<float>(), 1, db, &context_);

  // Compute dX and add incrementally
  // dX = dX + W'dX_softmax
  math::Gemv<float, CPUContext>(CblasTrans, dim_out, dim_in,
    1, W, dX_softmax, 1, dX, &context_);
}

// Implementation for the CPU context.
template <>
bool HSoftmaxGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  const auto& W = Input(1);
  const auto& b = Input(2);
  auto& label = Input(3);
  auto& intermediate_output = Input(4);
  auto& dY = Input(5);
  auto* dX = Output(0);
  auto* dW = Output(1);
  auto* db = Output(2);
  auto* dX_intermediate_output = Output(3);
  dX->ResizeLike(X);
  dW->ResizeLike(W);
  db->ResizeLike(b);
  dX_intermediate_output->ResizeLike(intermediate_output);

  float* dX_data = dX->mutable_data<float>();
  float* dW_data = dW->mutable_data<float>();
  float* db_data = db->mutable_data<float>();
  float* dOutput_data = dX_intermediate_output->mutable_data<float>();

  math::Set<float, CPUContext>(X.size(), 0.f, dX_data, &context_);
  math::Set<float, CPUContext>(W.size(), 0.f, dW_data, &context_);
  math::Set<float, CPUContext>(b.size(), 0.f, db_data, &context_);
  math::Set<float, CPUContext>(intermediate_output.size(), 0.f, dOutput_data,
                               &context_);

  // Batch size
  int M = X.ndim() > 1 ? X.dim32(0) : 1;
  // Input feature dimension
  int K = X.size() / M;
  const auto* labeldata = label.data<int>();

  std::unordered_map<int, PathProto> hierarchy = getHierarchyForLabels(M,
    labeldata, hierarchy_);
  int output_offset = getIntermediateOutputSize(labeldata, M, hierarchy);

  //Traverse backward to access intermediate_output generated by HSoftmaxOp
  // sequentially in reverse order
  for (int sample = M-1; sample >= 0; sample--) {
    int word_id = labeldata[sample];
    PathProto path = hierarchy[word_id];
    for (auto node = path.path_nodes().rbegin();
      node != path.path_nodes().rend(); node++) {
      int w_offset = node->index();
      int w_length = node->length();
      int target = node->target();
      RunBackwardSingle(X.data<float>() + sample*K, dY.data<float>() + sample,
        W.data<float>() + w_offset*K, target, intermediate_output.data<float>(),
        dX_data + sample*K, dW_data + w_offset*K, db_data + w_offset,
        dOutput_data, K, w_length, output_offset);
    }
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(HSoftmax, HSoftmaxOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(HSoftmaxGradient,
  HSoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(HSoftmax)
  .NumInputs(4)
  .NumOutputs(2)
  .SetDoc(R"DOC(
Hierarchical softmax is an operator which approximates the softmax operator
while giving significant training speed gains and reasonably comparable
performance. In this operator, instead of calculating the probabilities of all
the classes, we calculate the probability of each step in the path from root to
the target word in the hierarchy.

The operator takes a 2-D tensor (Tensor<float>) containing a batch of layers, a
set of parameters represented by the weight matrix and bias terms, and a 1-D
tensor (Tensor<int>) holding labels, or the indices of the target class. The
hierarchy has to be specified as an argument to the operator.

The operator returns a 1-D tensor holding the computed log probability of the
target class and a 2-D tensor of intermediate outputs (from the weight matrix
and softmax from each step in the path from root to target class) which will be
used by the gradient operator to compute gradients for all samples in the batch.
)DOC")
  .Arg("hierarchy", "Serialized HierarchyProto string containing list of "
  "vocabulary words and their paths from root of hierarchy to the leaf")
  .Input(0, "X", "Input data from previous layer")
  .Input(1, "W", "2D blob containing 'stacked' fully connected weight "
  "matrices. Each node in the hierarchy contributes one FC weight matrix if "
  "it has children nodes. Dimension is N*D, D is input dimension of data (X), "
  "N is sum of all output dimensions, or total number of nodes (excl root)")
  .Input(2, "b", "1D blob with N parameters")
  .Input(3, "labels", "int word_id of the target word")
  .Output(0, "Y", "1-D of log probability outputs, one per sample")
  .Output(1, "intermediate_output", "Extra blob to store the intermediate "
  "FC and softmax outputs for each node in the hierarchical path of a word. "
  "The outputs from samples are stored in consecutive blocks in the forward "
  "pass and are used in reverse order in the backward gradientOp pass");

OPERATOR_SCHEMA(HSoftmaxGradient).NumInputs(6).NumOutputs(4);

class GetHSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "HSoftmaxGradient", "",
        //X, W, b, label, intermediate output, dY
        vector<string>{I(0), I(1), I(2), I(3), O(1), GO(0)},
        //dX, dW, db, dintermediate_output
        vector<string>{GI(0), GI(1), GI(2), GO(1)});
  }
};
REGISTER_GRADIENT(HSoftmax, GetHSoftmaxGradient);
}  // namespace
}  // namespace caffe2
