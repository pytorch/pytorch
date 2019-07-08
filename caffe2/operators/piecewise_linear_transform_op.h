#ifndef CAFFE2_OPERATORS_PIECEWISE_LINEAR_TRANSFORM_OP_H_
#define CAFFE2_OPERATORS_PIECEWISE_LINEAR_TRANSFORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class PiecewiseLinearTransformOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit PiecewiseLinearTransformOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    binary_ = this->template GetSingleArgument<bool>("binary", false);

    // Retrieve transform params (i.e., the linear functions).
    bounds_from_arg_ = this->template GetRepeatedArgument<T>("bounds");
    slopes_from_arg_ = this->template GetRepeatedArgument<T>("slopes");
    intercepts_from_arg_ = this->template GetRepeatedArgument<T>("intercepts");
    transform_param_from_arg_ = CheckTransParamFromArg();
  }

  bool RunOnDevice() override {
    return binary_ ? TransformBinary() : TransformGeneral();
  }

 private:
  // num_func_per_group is the number of pieces of linear functions of
  // each group.
  // num_group: The number of groups of linear functions. Each group is for
  // transforming one column of predictions.
  void InferNumFunctionsPerGroup(
      const int64_t num_bounds,
      const int64_t num_slopes,
      const int64_t num_intercepts,
      int64_t* num_func_per_group,
      int64_t* num_group) {
    CAFFE_ENFORCE_EQ(num_slopes, num_intercepts);

    // This is based on the facts:
    // 1. in each group, the num of bounds minus the num of slopes is 1;
    // 2. each group has the same number of pieces.
    *num_group = num_bounds - num_slopes;
    CAFFE_ENFORCE_GT(*num_group, 0);
    if (binary_) {
      CAFFE_ENFORCE_EQ(*num_group, 1);
    }
    *num_func_per_group = num_slopes / *num_group;
    CAFFE_ENFORCE_GT(*num_func_per_group, 0);
    CAFFE_ENFORCE_EQ(num_slopes % *num_group, 0);
  }

  bool CheckBoundsSorted(
      const T* bounds,
      const int64_t num_bounds_per_group,
      const int64_t num_group) {
    const T* start = bounds;
    for (int64_t i = 0; i < num_group; i++) {
      if (!std::is_sorted(start, start + num_bounds_per_group)) {
        return false;
      }
      start += num_bounds_per_group;
    }
    return true;
  }

  // Returns true if the transform params from arg are valid.
  // Otherwise, we will assume the transform params will pass from Input blobs.
  bool CheckTransParamFromArg() {
    int good_param = 0;
    good_param += bounds_from_arg_.size() > 0;
    good_param += slopes_from_arg_.size() > 0;
    good_param += intercepts_from_arg_.size() > 0;
    CAFFE_ENFORCE(
        good_param == 0 || good_param == 3,
        "bounds, slopes, intercepts must be all set or all not set");
    if (good_param == 3) {
      int64_t num_func_per_group;
      int64_t num_group;
      InferNumFunctionsPerGroup(
          bounds_from_arg_.size(),
          slopes_from_arg_.size(),
          intercepts_from_arg_.size(),
          &num_func_per_group,
          &num_group);
      CAFFE_ENFORCE(
          CheckBoundsSorted(
              bounds_from_arg_.data(), num_func_per_group + 1, num_group),
          "bounds must be sorted for each group");
    }

    return good_param == 3;
  }

  void setUpTensors(int64_t& num_func_per_group, int64_t& num_group, int64_t M);

  void GetTransParamData(
      const T** bounds,
      const T** slopes,
      const T** intercepts,
      int64_t* num_func_per_group,
      int64_t* num_group) {
    int64_t num_bounds;
    int64_t num_slopes;
    int64_t num_intercepts;

    if (transform_param_from_arg_) {
      CAFFE_ENFORCE_EQ(InputSize(), 1);
      *bounds = bounds_from_arg_.data();
      *slopes = slopes_from_arg_.data();
      *intercepts = intercepts_from_arg_.data();
      num_bounds = bounds_from_arg_.size();
      num_slopes = slopes_from_arg_.size();
      num_intercepts = intercepts_from_arg_.size();
    } else {
      CAFFE_ENFORCE_EQ(InputSize(), 4);
      auto& bounds_input = Input(BOUNDS);
      auto& slopes_input = Input(SLOPES);
      auto& intercepts_input = Input(INTERCEPTS);
      *bounds = bounds_input.template data<T>();
      *slopes = slopes_input.template data<T>();
      *intercepts = intercepts_input.template data<T>();
      num_bounds = bounds_input.numel();
      num_slopes = slopes_input.numel();
      num_intercepts = intercepts_input.numel();
    }
    InferNumFunctionsPerGroup(
        num_bounds, num_slopes, num_intercepts, num_func_per_group, num_group);
  }

  bool TransformGeneral() {
    auto& X = Input(0);

    CAFFE_ENFORCE_EQ(X.dim(), 2);
    int64_t N = X.dim32(0);
    int64_t M = X.dim32(1);
    auto* Y = Output(0, X.sizes(), at::dtype<T>());
    const auto* Xdata = X.template data<T>();
    T* Ydata = Y->template mutable_data<T>();

    const T* bounds;
    const T* slopes;
    const T* intercepts;
    int64_t num_func_per_group;
    int64_t num_group;
    GetTransParamData(
        &bounds, &slopes, &intercepts, &num_func_per_group, &num_group);
    CAFFE_ENFORCE_EQ(num_group, M);

    for (int64_t j = 0; j < M; ++j) {
      const T* bounds_group = bounds + j * (num_func_per_group + 1);
      const T* slopes_group = slopes + j * num_func_per_group;
      const T* intercepts_group = intercepts + j * num_func_per_group;
      for (int64_t i = 0; i < N; ++i) {
        Ydata[i * M + j] = PiecewiseLinearTransform(
            Xdata[i * M + j],
            bounds_group,
            slopes_group,
            intercepts_group,
            num_func_per_group);
      }
    }
    return true;
  }

  bool TransformBinary() {
    auto& X = Input(PREDICTIONS);

    CAFFE_ENFORCE(X.dim() == 1 || X.dim() == 2);
    int64_t N = X.dim32(0);
    int64_t M = X.dim() == 2 ? X.dim32(1) : 1;
    CAFFE_ENFORCE(
        M == 1 || M == 2,
        "If binary is set to true, the input must be Nx2 or Nx1 tensor");
    auto* Y = Output(0, X.sizes(), at::dtype<T>());
    const auto* Xdata = X.template data<T>();
    T* Ydata = Y->template mutable_data<T>();

    const T* bounds;
    const T* slopes;
    const T* intercepts;
    int64_t num_func_per_group;
    int64_t num_group;
    GetTransParamData(
        &bounds, &slopes, &intercepts, &num_func_per_group, &num_group);
    CAFFE_ENFORCE_EQ(num_group, 1);

    if (M == 1) {
      for (int64_t i = 0; i < N; ++i) {
        Ydata[i] = PiecewiseLinearTransform(
            Xdata[i], bounds, slopes, intercepts, num_func_per_group);
      }
    } else {
      for (int64_t i = 0; i < N; ++i) {
        Ydata[i * M + 1] = PiecewiseLinearTransform(
            Xdata[i * M + 1], bounds, slopes, intercepts, num_func_per_group);
        Ydata[i * M] = 1.0f - Ydata[i * M + 1];
      }
    }

    return true;
  }

  T PiecewiseLinearTransform(
      const T x,
      const T* bounds,
      const T* slopes,
      const T* intercepts,
      const int64_t num_func_per_group) {
    T y = 0;
    // deal with samples out of bounds
    // make it the same as the upper/lower bound value
    if (x <= bounds[0]) {
      y = slopes[0] * bounds[0] + intercepts[0];
    } else if (x >= bounds[num_func_per_group]) {
      y = slopes[num_func_per_group - 1] * bounds[num_func_per_group] +
          intercepts[num_func_per_group - 1];
    } else {
      auto low_bound =
          std::lower_bound(bounds, bounds + num_func_per_group + 1, x);
      int bounds_idx = low_bound - bounds - 1;
      // compute the piecewise linear transformation as Y
      y = slopes[bounds_idx] * x + intercepts[bounds_idx];
    }
    return y;
  }

 private:
  bool binary_;
  vector<T> bounds_from_arg_;
  vector<T> slopes_from_arg_;
  vector<T> intercepts_from_arg_;

  Tensor bounds_device_{Context::GetDeviceType()};
  Tensor intercepts_device_{Context::GetDeviceType()};
  Tensor slopes_device_{Context::GetDeviceType()};
  bool gpu_copied_ = false;

  // If true, the piecewise linear functions are passed through args,
  // otherwise, they are passed through Input blobs.
  bool transform_param_from_arg_;

  INPUT_TAGS(PREDICTIONS, BOUNDS, SLOPES, INTERCEPTS);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PIECEWISE_LINEAR_TRANSFORM_OP_H_
