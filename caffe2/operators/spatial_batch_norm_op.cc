#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

template <>
bool SpatialBNOp<CPUContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);

  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  const int N = X.dim32(0);
  const int C =
      (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(X.ndim() - 1));
  const int H = (order_ == StorageOrder::NCHW ? X.dim32(2) : X.dim32(1));
  const int W = X.ndim() > 3
      ? (order_ == StorageOrder::NCHW ? X.dim32(3) : X.dim32(2))
      : 1;
  const int D = X.ndim() > 4
      ? (order_ == StorageOrder::NCHW ? X.dim32(4) : X.dim32(3))
      : 1;

  const int sample_size = H * W * D;
  CAFFE_ENFORCE_EQ(scale.ndim(), 1);
  CAFFE_ENFORCE_EQ(bias.ndim(), 1);
  CAFFE_ENFORCE_EQ(scale.dim32(0), C);
  CAFFE_ENFORCE_EQ(bias.dim32(0), C);

  ConstEigenVectorArrayMap<float> scale_arr(scale.data<float>(), C);
  ConstEigenVectorArrayMap<float> bias_arr(bias.data<float>(), C);

  auto* Y = Output(OUTPUT);
  Y->ResizeLike(X);

  if (!is_test_) {
    // training mode
    // Get the mean and variance.
    // Note that, to be consistent with cudnn, we will output saved inverse
    // std as output 5, but we will still use the same storage place to
    // compute var as well. The inverse is going to be carried out at the end
    // of the op.
    Output(SAVED_MEAN)->Resize(C);
    Output(SAVED_INV_VAR)->Resize(C);
    EigenVectorArrayMap<float> mean(
        Output(SAVED_MEAN)->mutable_data<float>(), C);
    EigenVectorArrayMap<float> var(
        Output(SAVED_INV_VAR)->mutable_data<float>(), C);

    if (num_batches_ > 1) {
      ConstEigenVectorArrayMap<float> sums(Input(SUMS).data<float>(), C);
      ConstEigenVectorArrayMap<float> sumsq(Input(SUMSQ).data<float>(), C);
      const auto multi_batch_size = N * num_batches_ * sample_size;
      mean = sums / multi_batch_size;
      var = (sumsq - (sums * sums) / multi_batch_size) / multi_batch_size;
    } else {
      mean.setZero();
      var.setZero();
      switch (order_) {
        case StorageOrder::NCHW: {
          ConstEigenArrayMap<float> X_arr(X.data<float>(), sample_size, N * C);
          for (int nc = 0; nc < N * C; ++nc) {
            mean(nc % C) += X_arr.col(nc).sum();
          }
          mean /= N * sample_size;
          for (int nc = 0; nc < N * C; ++nc) {
            var(nc % C) +=
                (X_arr.col(nc) - mean(nc % C)).matrix().squaredNorm();
          }
          var /= N * sample_size;
          break;
        }
        case StorageOrder::NHWC: {
          ConstEigenArrayMap<float> X_arr(X.data<float>(), C, N * sample_size);
          for (int i = 0; i < N * sample_size; ++i) {
            mean += X_arr.col(i);
          }
          mean /= N * sample_size;
          for (int i = 0; i < N * sample_size; ++i) {
            var += (X_arr.col(i) - mean) * (X_arr.col(i) - mean);
          }
          var /= N * sample_size;
          break;
        }
        default:
          CAFFE_THROW("Unknown storage order: ", order_);
      }
    }

    // Compute the running mean and running inv variance.
    auto* running_mean = Output(RUNNING_MEAN);
    auto* running_var = Output(RUNNING_VAR);
    // Check if they are initialized
    if (!running_mean->size()) {
      running_mean->Resize(C);
      EigenVectorArrayMap<float> running_mean_map(
          running_mean->mutable_data<float>(), C);
      running_mean_map.setZero();
    }
    if (!running_var->size()) {
      running_var->Resize(C);
      EigenVectorArrayMap<float> running_var_map(
          running_var->mutable_data<float>(), C);
      running_var_map.setZero();
    }
    EigenVectorArrayMap<float> running_mean_arr(
        running_mean->mutable_data<float>(), C);
    EigenVectorArrayMap<float> running_var_arr(
        running_var->mutable_data<float>(), C);
    running_mean_arr = running_mean_arr * momentum_ + mean * (1. - momentum_);
    running_var_arr = running_var_arr * momentum_ + var * (1. - momentum_);
  }

  // Regardless of training or testing, we will apply the estimated mean
  // and standard deviation to the input. For testing, they are
  // specified directly by the input, and for training, they are computed
  // by the op.
  Eigen::Array<float, Eigen::Dynamic, 1> inv_std(C);
  if (is_test_) {
    ConstEigenVectorArrayMap<float> var_arr(Input(EST_VAR).data<float>(), C);
    inv_std = (var_arr + epsilon_).sqrt().inverse();
  } else {
    EigenVectorArrayMap<float> saved_inv_std(
        Output(SAVED_INV_VAR)->mutable_data<float>(), C);
    saved_inv_std = (saved_inv_std + epsilon_).inverse().sqrt();
    inv_std = saved_inv_std;
  }
  ConstEigenVectorArrayMap<float> mean_arr(
      is_test_ ? Input(EST_MEAN).data<float>()
               : Output(SAVED_MEAN)->data<float>(),
      C);
  // We can fuse the output computation as follows:
  //   ((x - est_mean) * (inv_var) * scale + bias
  // to
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  Eigen::Array<float, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<float, Eigen::Dynamic, 1> new_bias =
      bias_arr - mean_arr * inv_std * scale_arr;
  switch (order_) {
    case StorageOrder::NHWC: {
      EigenArrayMap<float>(Y->mutable_data<float>(), C, N * sample_size) =
          (ConstEigenArrayMap<float>(X.data<float>(), C, N * sample_size)
               .colwise() *
           new_scale)
              .colwise() +
          new_bias;
      break;
    }
    case StorageOrder::NCHW: {
      EigenArrayMap<float> Y_arr(Y->mutable_data<float>(), sample_size, N * C);
      ConstEigenArrayMap<float> X_arr(X.data<float>(), sample_size, N * C);
      for (int nc = 0; nc < N * C; ++nc) {
        Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
  return true;
}

namespace {
OpSchema::Cost CostInferenceForSpatialBN(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<4>(def, in);
  ArgumentHelper helper(def);
  auto order =
      StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
  const TensorShape X = in[0];
  const int C =
      (order == StorageOrder::NCHW ? X.dims(1) : X.dims(X.dims_size() - 1));
  cost.params_bytes = 2 * C * sizeof(float);
  return cost;
}
} // namespace

REGISTER_CPU_OPERATOR(SpatialBN, SpatialBNOp<CPUContext>);

OPERATOR_SCHEMA(SpatialBN)
    .NumInputs({5, 7})
    .NumOutputs({1, 5})
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForSpatialBN)
    .EnforceInplace({{3, 1}, {4, 2}})
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          ArgumentHelper helper(def);
          bool is_test = helper.GetSingleArgument<int>(OpSchema::Arg_IsTest, 0);

          if (!is_test) {
            vector<TensorShape> out;
            StorageOrder order = StringToStorageOrder(
                helper.GetSingleArgument<string>("order", "NCHW"));
            const TensorShape& X = in[0];
            const int C =
                (order == StorageOrder::NCHW ? X.dims(1)
                                             : X.dims(X.dims_size() - 1));

            out.push_back(in[0]);
            TensorShape meanvar_tp =
                CreateTensorShape(vector<int>{C}, TensorProto::FLOAT);
            out.push_back(meanvar_tp); // RUNNING_MEAN
            out.push_back(meanvar_tp); // RUNNING_MEAN
            out.push_back(meanvar_tp); // SAVED_MEAN
            out.push_back(meanvar_tp); // SAVED_VAR
            return out;
          } else {
            return vector<TensorShape>{in[0]};
          }
        })
    .SetDoc(R"DOC(
Carries out spatial batch normalization as described in the paper
https://arxiv.org/abs/1502.03167 . Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:


Output case #1:
  Y, mean, var, saved_mean, saved_var (training mode)


Output case #2:
  Y (test mode)
)DOC")
    .ArgIsTest(
        "If set to nonzero, run spatial batch normalization in test mode.")
    .Arg("epsilon", "The epsilon value to use to avoid division by zero.")
    .Arg("order", "A StorageOrder string.")
    .Arg(
        "momentum",
        "Factor used in computing the running mean and variance."
        "e.g., running_mean = running_mean * momentum + mean * (1 - momentum)")
    .Arg(
        "num_batches",
        "(Optional) Specifies the number of batches to apply normalization on. "
        "Requires specifying the optional sums and sumsq inputs that provide "
        "statistics across multiple batches from which mean and variance can "
        "be determined.")
    .Input(
        0,
        "X",
        "The input 4-dimensional tensor of shape NCHW or NHWC depending "
        "on the order parameter.")
    .Input(
        1,
        "scale",
        "The scale as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Input(
        2,
        "bias",
        "The bias as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Input(
        3,
        "mean",
        "The running mean (training) or the estimated mean (testing) "
        "as a 1-dimensional tensor of size C.")
    .Input(
        4,
        "var",
        "The running variance (training) or the estimated "
        "variance (testing) as a 1-dimensional tensor of size C.")
    .Input(
        5,
        "sums",
        "(optional) Per-channel sums of elements to be used to determine the "
        "mean and variance for this batch")
    .Input(
        6,
        "sumsq",
        "(optional) Per-channel sum of elements squared per channel to be used "
        "to determine the variance for this batch")

    .Output(0, "Y", "The output 4-dimensional tensor of the same shape as X.")
    .Output(
        1,
        "mean",
        "The running mean after the spatial BN operator. Must be in-place "
        "with the input mean. Should not be used for testing.")
    .Output(
        2,
        "var",
        "The running variance after the spatial BN operator. Must be "
        "in-place with the input var. Should not be used for testing.")
    .Output(
        3,
        "saved_mean",
        "Saved mean used during training to speed up gradient "
        "computation. Should not be used for testing.")
    .Output(
        4,
        "saved_var",
        "Saved variance used during training to speed up "
        "gradient computation. Should not be used for testing.")
    .InheritOnnxSchema("BatchNormalization");

} // namespace caffe2
