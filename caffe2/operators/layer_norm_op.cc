#include "caffe2/operators/layer_norm_op.h"
#include "caffe2/utils/eigen_utils.h"
#include <ATen/core/opschema/layer_norm.h>
#include <ATen/core/dispatch/KernelRegistration.h>
#include <c10/core/Tensor.h>

namespace caffe2 {

template <>
template <typename T>
void LayerNormOp<CPUContext>::ComputeStdDevAndFusedParams(
    const int N,
    const T* mean,
    const T* var,
    T* stddev,
    T* scale,
    T* bias,
    float epsilon,
    CPUContext* /*context*/) {
  ConstEigenVectorArrayMap<T> var_arr(var, N);
  EigenVectorArrayMap<T> stddev_arr(stddev, N);
  EigenVectorArrayMap<T> scale_arr(scale, N);
  scale_arr = (var_arr + static_cast<T>(epsilon)).rsqrt();
  stddev_arr = scale_arr * (var_arr + static_cast<T>(epsilon));
  EigenVectorArrayMap<T>(bias, N) =
      -scale_arr * ConstEigenVectorArrayMap<T>(mean, N);
}

template <>
template <typename T>
void LayerNormOp<CPUContext>::LayerNormForward(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y,
    CPUContext* context) {
  EigenArrayMap<T>(Y, N, M) =
      (ConstEigenArrayMap<T>(X, N, M).rowwise() *
       ConstEigenVectorArrayMap<T>(scale, M).transpose())
          .rowwise() +
      ConstEigenVectorArrayMap<T>(bias, M).transpose();
}

REGISTER_CPU_OPERATOR(LayerNorm, LayerNormOp<CPUContext>);

template <>
template <typename T>
void LayerNormGradientOp<CPUContext>::ComputeInternalGradients(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    T* ds,
    T* db) {
  ConstEigenArrayMap<T> dY_arr(dY, N, M);
  EigenVectorArrayMap<T>(ds, M) =
      (dY_arr * ConstEigenArrayMap<T>(X, N, M)).colwise().sum();
  EigenVectorArrayMap<T>(db, M) = dY_arr.colwise().sum();
}

template <>
template <typename T>
void LayerNormGradientOp<CPUContext>::ComputeFusedParams(
    const int M,
    const int N,
    const T* mean,
    const T* sig,
    const T* ds,
    const T* db,
    T* dY_scale,
    T* X_scale,
    T* bias) {
  const T scale = T(1) / static_cast<T>(N);
  ConstEigenVectorArrayMap<T> mean_arr(mean, M);
  ConstEigenVectorArrayMap<T> ds_arr(ds, M);
  ConstEigenVectorArrayMap<T> db_arr(db, M);
  EigenVectorArrayMap<T> rsig_arr(dY_scale, M);
  EigenVectorArrayMap<T> X_scale_arr(X_scale, M);
  rsig_arr = ConstEigenVectorArrayMap<T>(sig, M).inverse();
  X_scale_arr = (db_arr * mean_arr - ds_arr) * rsig_arr.cube() * scale;
  EigenVectorArrayMap<T>(bias, M) =
      -X_scale_arr * mean_arr - db_arr * rsig_arr * scale;
}

template <>
template <typename T>
void LayerNormGradientOp<CPUContext>::LayerNormBackward(
    const int M,
    const int N,
    const T* dY_scale,
    const T* dY,
    const T* X_scale,
    const T* X,
    const T* bias,
    T* dX) {
  EigenArrayMap<T>(dX, N, M) =
      (ConstEigenArrayMap<T>(dY, N, M).rowwise() *
           ConstEigenVectorArrayMap<T>(dY_scale, M).transpose() +
       ConstEigenArrayMap<T>(X, N, M).rowwise() *
           ConstEigenVectorArrayMap<T>(X_scale, M).transpose())
          .rowwise() +
      ConstEigenVectorArrayMap<T>(bias, M).transpose();
}

OPERATOR_SCHEMA(LayerNormGradient).NumInputs(5).NumOutputs(1);

REGISTER_CPU_OPERATOR(LayerNormGradient, LayerNormGradientOp<CPUContext>);

namespace {

class GetLayerNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LayerNormGradient",
        "",
        std::vector<std::string>{GO(0), O(0), O(1), O(2), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(LayerNorm, GetLayerNormGradient);

OPERATOR_SCHEMA(LayerNorm)
    .NumInputs(1)
    .NumOutputs(3)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(3);
      auto input_dims_long = GetDimsVector(in[0]);
      std::vector<int> input_dims(
          input_dims_long.begin(), input_dims_long.end());
      out[0] = CreateTensorShape(input_dims, TensorProto::FLOAT);

      ArgumentHelper helper(def);

      auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const auto canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      std::vector<int> stat_dims(
          input_dims.begin(), input_dims.begin() + canonical_axis);
      stat_dims.push_back(1);
      out[1] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      out[2] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
Computes layer normalization as described in https://arxiv.org/pdf/1607.06450.pdf.
Given an input vector x \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}],
this op treats dimensions a_k through a_{n-1} as feature vectors. For each
feature vector, the op contains the mean and standard deviation. Then,
it returns the normalized values (with respect to the feature vector).

Note that this op does not contain the scale an bias terms described in the
paper. Simply follow this op with an FC op to add those. Concretely, this op
implements:

h = \frac{1}{\sigma}(a - \mu)
where \mu = \frac{1}{H}\sum_{i=1}^{H} a_i
and \sigma = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_i - \mu)^2}
where H is the number of hidden units (i.e. product of dimensions from 'axis'
to the end.)
)DOC")
    .Arg(
        "axis",
        "(int) default to 1; Describes axis of the inputs. Defaults to one "
        "because the 0th axis most likely describes the batch size")
    .Arg(
        "epsilon",
        "(float) default to 0.001. Small value to be added to the stdev when"
        " dividing out by that value. This prevents division by zero.")
    .Input(
        0,
        "input",
        "Input tensor which layer normalization will be applied to")
    .Output(0, "output", "Normalized values")
    .Output(1, "mean", "Mean values for each feature vector")
    .Output(2, "stddev", "Standard deviations for each feature vector");

} // namespace caffe2


// Register layer norm with c10
namespace {
struct Cache final : public c10::KernelCache {
    at::optional<at::Tensor> scale = at::nullopt;
    at::optional<at::Tensor> bias = at::nullopt;
};

template <class DataType>
void layer_norm_c10(c10::Stack* stack, c10::KernelCache* cache_) { // TODO Pass in correct cache type
  c10::ArrayRef<c10::IValue> inputs = torch::jit::peekSlice(*stack, 0, 3, 6);
  c10::ArrayRef<c10::IValue> outputs = torch::jit::peekSlice(*stack, 3, 3, 6);


  caffe2::Tensor X{inputs[0].toTensor()};
  int64_t axis = inputs[1].toInt();
  float epsilon = inputs[2].toDouble();

  auto device = X.GetDevice();

  caffe2::Tensor Y, mean, sig;
  if (outputs[0].isTensor()) {
    Y = caffe2::Tensor(std::move(torch::jit::peek(*stack, 0, 3)).toTensor());
  }
  if (outputs[1].isTensor()) {
    mean = caffe2::Tensor(std::move(torch::jit::peek(*stack, 1, 3)).toTensor());
  }
  if (outputs[2].isTensor()) {
    sig = caffe2::Tensor(std::move(torch::jit::peek(*stack, 2, 3)).toTensor());
  }
  if (!Y.defined()) {
    Y = caffe2::empty({0}, device);
  }
  if (!mean.defined()) {
    mean = caffe2::empty({0}, device);
  }
  if (!sig.defined()) {
    sig = caffe2::empty({0}, device);
  }

  caffe2::CPUContext context;
  Cache* cache = static_cast<Cache*>(cache_);
  if (!cache->scale.has_value()) {
    cache->scale = at::Tensor(caffe2::empty({0}, at::dtype<float>()));
  }
  if (!cache->bias.has_value()) {
    cache->bias = at::Tensor(caffe2::empty({0}, at::dtype<float>()));
  }
  caffe2::Tensor scale(*cache->scale);
  caffe2::Tensor bias(*cache->bias);

  const int canonical_axis = X.canonical_axis_index(axis);
  std::vector<int64_t> moments_dims(
      X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
  moments_dims.push_back(1);
  mean.Resize(moments_dims);
  sig.Resize(moments_dims);
  caffe2::LayerNormOp<caffe2::CPUContext>::runLayerNorm<DataType>(
    X, &Y, &mean, &sig, canonical_axis, epsilon, &scale, &bias, static_cast<caffe2::CPUContext*>(&context)
  );

  torch::jit::drop(*stack, 6);
  torch::jit::push(*stack,
    at::Tensor(std::move(Y)),
    at::Tensor(std::move(mean)),
    at::Tensor(std::move(sig))
  );

  return;
}
}
namespace c10 {
C10_REGISTER_KERNEL(c10::core::opschema::LayerNorm)
    .withCache<Cache>()
    .kernel<&layer_norm_c10<float>>()
    .dispatchKey(CPUTensorId());
} // namespace c10
