#include "caffe2/operators/batch_matmul_op.h"

#include "caffe2/core/operator_schema.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(BatchMatMul, BatchMatMulOp<CPUContext>);

vector<TensorShape> TensorInferenceForBatchMatMul(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  ArgumentHelper helper(def);
  bool broadcast = helper.GetSingleArgument<int>("broadcast", 0);
  if (!broadcast) {
    const auto ndim = in[0].dims_size();
    CAFFE_ENFORCE_GE(ndim, 2);
    CAFFE_ENFORCE_GE(in[1].dims_size(), 2);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int a_dim0;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int b_dim1;
    if (helper.GetSingleArgument<int>("trans_a", 0)) {
      a_dim0 = in[0].dims(ndim - 1);
    } else {
      a_dim0 = in[0].dims(ndim - 2);
    }

    if (helper.GetSingleArgument<int>("trans_b", 0)) {
      b_dim1 = in[1].dims(ndim - 2);
    } else {
      b_dim1 = in[1].dims(ndim - 1);
    }

    auto output_dims =
        vector<int64_t>{in[0].dims().begin(), in[0].dims().end()};
    output_dims[ndim - 2] = a_dim0;
    output_dims[ndim - 1] = b_dim1;

    return vector<TensorShape>{
        CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
  } else {
    auto ndims_A = in[0].dims_size();
    auto ndims_B = in[1].dims_size();
    std::vector<int64_t> dims_A(ndims_A), dims_B(ndims_B);
    for (int i = 0; i < ndims_A; ++i) {
      dims_A[i] = in[0].dims(i);
    }
    for (int i = 0; i < ndims_B; ++i) {
      dims_B[i] = in[1].dims(i);
    }
    bool A_broadcasted = false, B_broadcasted = false;
    if (ndims_A == 1) {
      dims_A.insert(dims_A.begin(), 1);
      ndims_A = 2;
      A_broadcasted = true;
    }
    if (ndims_B == 1) {
      dims_B.push_back(1);
      ndims_B = 2;
      B_broadcasted = true;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t M, N;
    if (helper.GetSingleArgument<int>("trans_a", 0)) {
      M = dims_A[ndims_A - 1];
    } else {
      M = dims_A[ndims_A - 2];
    }
    if (helper.GetSingleArgument<int>("trans_b", 0)) {
      N = dims_B[ndims_B - 2];
    } else {
      N = dims_B[ndims_B - 1];
    }

    std::vector<int64_t> new_dims;
    if (ndims_A >= ndims_B) {
      new_dims.assign(dims_A.begin(), dims_A.end() - 2);
    } else {
      new_dims.assign(dims_B.begin(), dims_B.end() - 2);
    }
    if (!A_broadcasted) {
      new_dims.push_back(M);
    }
    if (!B_broadcasted) {
      new_dims.push_back(N);
    }
    if (A_broadcasted && B_broadcasted) {
      new_dims.push_back(1);
    }
    return vector<TensorShape>{
        CreateTensorShape(vector<int64_t>{new_dims}, in[0].data_type())};
  }
}

OpSchema::Cost CostInferenceForBatchMatMul(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  CAFFE_ENFORCE_EQ(in.size(), 2U, "BatchMatMul requires two inputs");

  ArgumentHelper helper(def);
  struct OpSchema::Cost c;
  const auto& A = in[0];
  const auto& B = in[1];
  const TensorShape Y = TensorInferenceForBatchMatMul(def, in)[0];

  uint64_t nElemA = nElemFromDim(A);
  uint64_t nElemB = nElemFromDim(B);
  uint64_t nElemY = nElemFromDim(Y);

  auto ndims_A = A.dims_size();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t K;
  if (helper.GetSingleArgument<int>("trans_a", 0)) {
    K = in[0].dims(ndims_A - 2);
  } else {
    K = in[0].dims(ndims_A - 1);
  }

  c.flops = 2 * nElemY * K;
  c.bytes_read = (nElemA + nElemB) * sizeof(A.data_type());
  c.bytes_written = nElemY * sizeof(Y.data_type());
  c.params_bytes = 0;
  return c;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(BatchMatMul)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Batch Matrix multiplication Yi = Ai * Bi, where A has shape (dim0, dim1, ... M, K),
B has shape (dim0, dim1, ... K, N), Y has shape (dim0, dim1, ... M, N) and i ranges
from 0 to (dim0 * dim1 ...) - 1. rank(A) == rank(B) >= 2. In case of A and B being
two dimensional, it behaves like normal matrix multiplication.
)DOC")
    .Input(0, "A", "tensor of shape (dim0, dim1 ... M, K)")
    .Input(1, "B", "tensor of shape (dim0, dim1 ... K, N)")
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

class GetBatchMatMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);

    bool broadcast = false;
    if (ArgumentHelper::HasArgument(Def(), "broadcast")) {
      broadcast = GetArgument(Def(), "broadcast").i();
    }
    CAFFE_ENFORCE(
        !broadcast,
        "Gradient is currently not supported with "
        "broadcast=1 for BatchMatMul.");

    // NOLINTNEXTLINE(modernize-use-bool-literals)
    bool trans_a = 0;
    // NOLINTNEXTLINE(modernize-use-bool-literals)
    bool trans_b = 0;

    if (ArgumentHelper::HasArgument(Def(), "trans_a")) {
      trans_a = GetArgument(Def(), "trans_a").i();
    }
    if (ArgumentHelper::HasArgument(Def(), "trans_b")) {
      trans_b = GetArgument(Def(), "trans_b").i();
    }

    auto no_trans_arg = vector<Argument>();
    auto trans_a_arg = vector<Argument>{MakeArgument<int>("trans_a", 1)};
    auto trans_b_arg = vector<Argument>{MakeArgument<int>("trans_b", 1)};
    auto trans_both_arg = vector<Argument>{MakeArgument<int>("trans_a", 1),
                                           MakeArgument<int>("trans_b", 1)};

    if (trans_a) {
      if (trans_b) {
        // A'B':
        // dA = B'G', dB = G'A'
        return vector<OperatorDef>{CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{I(1), GO(0)},
                                       vector<string>{GI(0)},
                                       trans_both_arg),
                                   CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{GO(0), I(0)},
                                       vector<string>{GI(1)},
                                       trans_both_arg)};
      } else {
        // A'B:
        // dA = BG', dB = AG
        return vector<OperatorDef>{CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{I(1), GO(0)},
                                       vector<string>{GI(0)},
                                       trans_b_arg),
                                   CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{I(0), GO(0)},
                                       vector<string>{GI(1)},
                                       no_trans_arg)};
      }
    } else {
      if (trans_b) {
        // AB':
        // dA = GB, dB = G'A
        return vector<OperatorDef>{CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{GO(0), I(1)},
                                       vector<string>{GI(0)},
                                       no_trans_arg),
                                   CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{GO(0), I(0)},
                                       vector<string>{GI(1)},
                                       trans_a_arg)};
      } else {
        // AB:
        // dA = GB', dB = A'G
        return vector<OperatorDef>{CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{GO(0), I(1)},
                                       vector<string>{GI(0)},
                                       trans_b_arg),
                                   CreateOperatorDef(
                                       "BatchMatMul",
                                       "",
                                       vector<string>{I(0), GO(0)},
                                       vector<string>{GI(1)},
                                       trans_a_arg)};
      }
    }
  }

  bool CopyArguments() const override {
    return false;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(BatchMatMul, GetBatchMatMulGradient);

} // namespace caffe2
