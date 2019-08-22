#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

inline ideep::tensor::dims CanonicalDims_(
    ideep::tensor::dims adims, int32_t axis) {
  CAFFE_ENFORCE(axis < (int32_t)adims.size(), "Invalid axis!");
  CAFFE_ENFORCE(axis >= (int32_t)-adims.size(), "Invalid axis!");
  if (adims.size() == 2)
    return adims;
  if (axis < 0) {
    axis += (int32_t)adims.size();
  }

  auto dim0 = std::accumulate(adims.begin(), adims.begin() + axis, 1,
                              std::multiplies<ideep::tensor::dim_t>());
  auto dim1 = std::accumulate(adims.begin() + axis, adims.end(), 1,
                              std::multiplies<ideep::tensor::dim_t>());
  return ideep::tensor::dims({dim0, dim1});
}

class IDEEPMatMulWithBiasOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPMatMulWithBiasOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        axis_A_(OperatorBase::GetSingleArgument<int32_t>("axis_A", 1)),
        axis_B_(OperatorBase::GetSingleArgument<int32_t>("axis_B", 1)),
        trans_a_(OperatorBase::GetSingleArgument<int32_t>("trans_a", 0)),
        trans_b_(OperatorBase::GetSingleArgument<int32_t>("trans_b", 0)) {}
  ~IDEEPMatMulWithBiasOp() override {}

  bool RunOnDevice() override {
    const auto& A = Input(INPUT_A);
    const auto& B = Input(INPUT_B);
    auto* Y = Output(OUTPUT);

    itensor A_in = A;
    auto A_dims = A.get_dims();
    if (A.ndims() > 2) {
      auto A_in_dims = CanonicalDims_(A_dims, axis_A_);
      A_in.reshape(A_in_dims);
    }

    itensor A_transposed;
    if (trans_a_ == 1) {
      itensor::dims dims{A_in.get_dim(1), A_in.get_dim(0)};
      std::vector<int> axes{1, 0};
      A_transposed.init({dims, A_in.get_data_type()});
      A_transposed.transpose_from(A_in, axes);
    }

    itensor B_;
    auto B_dims = B.get_dims();
    if (B.ndims() > 2) {
      B_dims = CanonicalDims_(B_dims, axis_B_);
    }

    itensor::descriptor desc;
    if (trans_b_ == 1) {
      desc = itensor::descriptor(B_dims, B.get_data_type(), ideep::format::oi);
    } else {
      // MKL-DNN uses OI as canonical order for 2D so we have to swap the dims in the case of IO.
      itensor::dims dims_asif_OI{B_dims[1], B_dims[0]};
      desc = itensor::descriptor(
          dims_asif_OI, B.get_data_type(), ideep::format::io);
    }
    if (B.is_public_format()) {
      B_.init(desc, B.get_data_handle());
    }
    else {
      B_.init(desc);
      B.to_public(B_.get_data_handle());
    }

    itensor bias_;
    if (InputSize() > BIAS) {
      bias_ = Input(BIAS);
      CAFFE_ENFORCE_EQ(
          bias_.get_dim(0), B_.get_dim(0), "Bias shape mismatched!");
    }

    CAFFE_ENFORCE_EQ(
        trans_a_ == 1 ? A_transposed.get_dim(1) : A_in.get_dim(1),
        B_.get_dim(1),
        "A shape mismatch B shape");

    if (InputSize() > BIAS) {
      ideep::inner_product_forward::compute(
          trans_a_ == 1 ? A_transposed : A_in, B_, bias_, *Y);
    } else {
      ideep::inner_product_forward::compute(trans_a_ == 1 ? A_transposed : A_in, B_, *Y);
    }

    if (A_dims.size() > 2) {
      std::vector<int> dims;
      if (trans_a_ == 1) {
        dims.insert(dims.begin(), A_dims.begin() + axis_A_, A_dims.end());
        dims.push_back(B_.get_dim(0));
      } else {
        dims.insert(dims.begin(), A_dims.begin(), A_dims.begin() + axis_A_);
        dims.push_back(B_.get_dim(0));
      }
      Y->reshape(dims);
    }

    return true;
  }

 private:
  size_t axis_A_{1};
  size_t axis_B_{1};
  int trans_a_{1};
  int trans_b_{1};

  INPUT_TAGS(INPUT_A, INPUT_B, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPMatMulWithBiasGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPMatMulWithBiasGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        axis_A_(OperatorBase::GetSingleArgument<int32_t>("axis_A", 1)),
        axis_B_(OperatorBase::GetSingleArgument<int32_t>("axis_B", 1)),
        trans_a_(OperatorBase::GetSingleArgument<int32_t>("trans_a", 0)),
        trans_b_(OperatorBase::GetSingleArgument<int32_t>("trans_b", 0)) {}
  ~IDEEPMatMulWithBiasGradientOp() override {}

  bool RunOnDevice() override {
    const auto& A = Input(INPUT_A);
    const auto& B = Input(INPUT_B);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dA = Output(A_GRAD);
    auto* dB = Output(B_GRAD);

    itensor A_in = A;
    auto A_dims = A.get_dims();
    if (A.ndims() > 2) {
      A_dims = CanonicalDims_(A_dims, axis_A_);
      A_in.reshape(A_dims);
    }

    itensor A_transposed;
    if (trans_a_ == 1) {
      itensor::dims dims{A_dims[1], A_dims[0]};
      std::vector<int> axes{1, 0};
      A_transposed.init({dims, A_in.get_data_type()});
      A_transposed.transpose_from(A_in, axes);
    }

    itensor B_;
    auto B_dims = B.get_dims();
    if (B.ndims() > 2) {
      B_dims = CanonicalDims_(B_dims, axis_B_);
    }

    itensor::descriptor desc;
    if (trans_b_ == 1) {
      desc = itensor::descriptor(B_dims, B.get_data_type(), ideep::format::oi);
    } else {
      // MKL-DNN uses OI as canonical order for 2D so we have to swap the dims in the case of IO.
      itensor::dims dims_asif_OI{B_dims[1], B_dims[0]};
      desc = itensor::descriptor(
          dims_asif_OI, B.get_data_type(), ideep::format::io);
    }
    if (B.is_public_format()) {
      B_.init(desc, B.get_data_handle());
    }
    else {
      B_.init(desc);
      B.to_public(B_.get_data_handle());
    }

    itensor dY_ = dY;
    if (dY_.ndims() > 2) {
      auto dY_dims = dY_.get_dims();
      auto dY_in_dims = CanonicalDims_(dY_dims, dY_.ndims() - 1);
      dY_.reshape(dY_in_dims);
    }

    itensor dB_;
    if (OutputSize() > BIAS_GRAD) {
      auto* dbias = Output(BIAS_GRAD);
      ideep::inner_product_backward_weights::compute(
          trans_a_ == 1 ? A_transposed : A_in, dY_, trans_b_ == 1 ? *dB : dB_, *dbias);
    } else {
      ideep::inner_product_backward_weights::compute(
          trans_a_ == 1 ? A_transposed : A_in, dY_, trans_b_ == 1 ? *dB : dB_);
    }

    if (trans_b_ != 1) {
      dB->reinit({B_dims, B.get_data_type(), ideep::format::io});
      std::vector<int> axes{1, 0};
      dB->transpose_from(dB_, axes);
    }

    if (B.ndims() > 2) {
      dB->reshape(B.get_dims());
    }

    itensor dA_;
    ideep::inner_product_backward_data::compute(
        dY_,
        B_,
        trans_a_ == 1 ? A_transposed.get_dims() : A_in.get_dims(),
        trans_a_ == 1 ? dA_ : *dA);

    if (trans_a_ == 1) {
      dA->reinit({A_dims, A.get_data_type(), dA_.get_internal_format()});
      std::vector<int> axes{1, 0};
      dA->transpose_from(dA_, axes);
    }

    if (A.ndims() > 2) {
      dA->reshape(A.get_dims());
    }

    return true;
  }

 private:
  size_t axis_A_{1};
  size_t axis_B_{1};
  int trans_a_{1};
  int trans_b_{1};

  INPUT_TAGS(INPUT_A, INPUT_B, OUTPUT_GRAD);
  OUTPUT_TAGS(A_GRAD, B_GRAD, BIAS_GRAD);
};

REGISTER_IDEEP_OPERATOR(MatMulWithBias, IDEEPMatMulWithBiasOp);
REGISTER_IDEEP_OPERATOR(MatMulWithBiasGradient, IDEEPMatMulWithBiasGradientOp);

OPERATOR_SCHEMA(MatMulWithBias)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Matrix multiplication with bias $Y = A * B + Bias$, where `A` has size (M x K) or (Batch_size x M x k), `B` has size
(K x N), `Bias` has size (n), and `Y` will have a size (M x N) or (Batch_size x M x N). To transpose `A` or `B` before
multiplication, pass 1 to the `trans_a` and/or `trans_b` arguments, which
separate the first and second dimensions of the respective matrices using
`axis_a` and `axis_b`.

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()
use_bias = True
op = core.CreateOperator(
    "MatMulWithBias",
    ["A", "B", "Bias"] if use_bias else ["A", "B"],
    ["Y"],
)

workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.float32))
workspace.FeedBlob("B", np.random.randint(10, size=(3,3)).astype(np.float32))
if use_bias:
    workspace.FeedBlob("Bias", np.random.randint(10, size=(3)).astype(np.float32))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
```

</details>

)DOC")
    .Input(
        0,
        "A",
        "*(type: Tensor`<float>`)* 2D or 3D matrix of size (M x K) or (Batch_size x M x K).")
    .Input(1, "B", "*(type: Tensor`<float>`)* 2D matrix of size (K x N).")
    .Input(2, "Bias", "*(type: Tensor`<float>`)* D matrix of size ( N).")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* 2D or 3D matrix of size (M x N) or (Batch_size x M x N).")
    .Arg(
        "axis_a",
        "*(type: int; default: 1)* Exclusive axis that divides the first and "
        "second dimension of matrix `A`.")
    .Arg(
        "axis_b",
        "*(type: int; default: 1)* Exclusive axis that divides the first and "
        "second dimension of matrix `B`.")
    .Arg(
        "trans_a",
        "*(type: int; default: 0)* Pass 1 to transpose `A` before multiplication and "
        "after the dimension adjustment using `axis_a`.")
    .Arg(
        "trans_b",
        "*(type: int; default: 0)* Pass 1 to transpose `B` before multiplication and "
        "after the dimension adjustment using `axis_b`.");

GRADIENT_OPERATOR_SCHEMA(MatMulWithBiasGradient).NumInputs(3).NumOutputs(2, 3);

namespace {

class GetMatMulWithBiasGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 2 || def_.input_size() == 3);
    CAFFE_ENFORCE(def_.type() == "MatMulWithBias");
    if (def_.input_size() == 2) {
      return SingleGradientDef(
          def_.type() + "Gradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(0), GI(1)});
    } else {
      return SingleGradientDef(
          def_.type() + "Gradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(0), GI(1), GI(2)});
    }
  }
};

REGISTER_GRADIENT(MatMulWithBias, GetMatMulWithBiasGradient);

} // namespace
} // namespace caffe2
