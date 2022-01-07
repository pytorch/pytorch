#ifndef CAFFE2_OPERATORS_TILE_OP_H_
#define CAFFE2_OPERATORS_TILE_OP_H_

#include <array>
#include <string>
#include <type_traits>
#include <vector>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Copy a Blob n times along a specified axis.
template <class Context>
class TileOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit TileOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(std::int32_t, "tiles", tiles_, 1),
        OP_SINGLE_ARG(std::int32_t, "axis", axis_, 0) {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<std::int32_t, std::int64_t, float, double>>::
        call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    if (InputSize() > 1) {
      // We potentially have tiles and/or axis specified as inputs
      // as well. We will check for them in that order. In other words:
      // InputSize() == 2: tiles is specified
      // InputSize() == 3: tiles is specified and axis.
      // Anything specified as input will override the arguments
      CAFFE_ENFORCE(
          Input(1).dim() == 1 && Input(1).numel() == 1,
          "Input `tiles` should be a vector of size 1.");
      tiles_ = GetArgFromTensor(Input(1));

      // Because of a bug in original code, temporarily adds this part to keep
      // backward compatibility.
      // TODO(yangxm): Remove this part when prod runtime upgraded with fixed
      // model config.
      if (Input(1).template IsType<std::int64_t>()) {
        axis_ = 0;
      }

      if (InputSize() > 2) {
        CAFFE_ENFORCE(
            Input(2).dim() == 1 && Input(2).numel() == 1,
            "Input `axis` should be a vector of size 1.");
        axis_ = GetArgFromTensor(Input(2));
      } else {
        CAFFE_ENFORCE(
            OperatorBase::HasArgument("axis"),
            "Argument `axis` is missing and was not specified as input.");
      }
    } else {
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("tiles"),
          "Argument `tiles` is missing and was not specified as input.");
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("axis"),
          "Argument `axis` is missing and was not specified as input.");
    }

    const auto& X = Input(0);
    auto* Y = Output(0);
    const int axis = X.canonical_axis_index(axis_);

    // reshape output to be input tiled along the axis
    std::vector<std::int64_t> Y_dims = X.sizes().vec();
    Y_dims[axis] *= tiles_;
    Y->Resize(Y_dims);

    // size up to (and not including) axis
    const int outer_size = X.size_to_dim(axis);
    // size from axis up
    const int inner_size = X.size_from_dim(axis);

    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    return DoTile<T>(outer_size, inner_size, X_data, Y_data);
  }

 private:
  std::int32_t GetArgFromTensor(const Tensor& tensor) {
    CAFFE_ENFORCE(
        tensor.IsType<std::int32_t>() || tensor.IsType<std::int64_t>());
    std::int32_t val = -1;
    if (tensor.IsType<std::int32_t>()) {
      context_.template CopyToCPU<std::int32_t>(
          1, tensor.data<std::int32_t>(), &val);
    } else if (tensor.IsType<std::int64_t>()) {
      std::int64_t val_int64;
      context_.template CopyToCPU<std::int64_t>(
          1, tensor.data<std::int64_t>(), &val_int64);
      val = static_cast<std::int32_t>(val_int64);
    }
    return val;
  }

  template <typename T>
  bool DoTile(const int outer_size, const int inner_size, const T* X, T* Y) {
    if (inner_size == 1) {
      EigenArrayMap<T> Y_arr(Y, tiles_, outer_size);
      for (const auto i : c10::irange(outer_size)) {
        Y_arr.col(i) = X[i];
      }
    } else {
      ConstEigenArrayMap<T> X_arr(X, inner_size, outer_size);
      for (const auto i : c10::irange(outer_size)) {
        EigenArrayMap<T>(Y + i * tiles_ * inner_size, inner_size, tiles_)
            .colwise() = X_arr.col(i);
      }
    }
    return true;
  }

  std::int32_t tiles_;
  std::int32_t axis_;
};

template <class Context>
class TileGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit TileGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(std::int32_t, "tiles", tiles_, 1),
        OP_SINGLE_ARG(std::int32_t, "axis", axis_, 0) {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<std::int32_t, std::int64_t, float, double>>::
        call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    if (InputSize() > 1) {
      // We potentially have tiles and/or axis specified as inputs
      // as well. We will check for them in that order. In other words:
      // InputSize() == 2: tiles is specified
      // InputSize() == 3: tiles is specified and axis.
      // Anything specified as input will override the arguments
      CAFFE_ENFORCE(
          Input(1).dim() == 1 && Input(1).numel() == 1,
          "Input `tiles` should be a vector of size 1.");
      tiles_ = GetArgFromTensor(Input(1));
      if (InputSize() > 2) {
        CAFFE_ENFORCE(
            Input(2).dim() == 1 && Input(2).numel() == 1,
            "Input `axis` should be a vector of size 1.");
        axis_ = GetArgFromTensor(Input(2));
      } else {
        CAFFE_ENFORCE(
            OperatorBase::HasArgument("axis"),
            "Argument `axis` is missing and was not specified as input.");
      }
    } else {
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("tiles"),
          "Argument `tiles` is missing and was not specified as input.");
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("axis"),
          "Argument `axis` is missing and was not specified as input.");
    }

    const auto& dY = Input(0);
    auto* dX = Output(0);
    const int axis = dY.canonical_axis_index(axis_);

    // reshape output to be input "untiled" along the axis
    std::vector<std::int64_t> X_dims = dY.sizes().vec();
    CAFFE_ENFORCE_EQ(X_dims[axis] % tiles_, 0);
    X_dims[axis] /= tiles_;
    dX->Resize(X_dims);

    // size up to (and not including) axis
    const int outer_size = dX->size_to_dim(axis);
    // size from axis up
    const int inner_size = dX->size_from_dim(axis);

    /**
     * How this works:
     * Imagine a 2D tensor (matrix) of size 3x10, tiled 2 times along axis 1
     * (column).
     * This is equivalent to multiplying by a vector of 1s transposed.
     * The gradient of this is all 1s in the shape of the input matrix
     * (call it X).
     * So the output gradient should be the matrix multiplication result
     * of input gradient (gradient of tiled tensor output) and X.
     */
    const T* dY_data = dY.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    return DoTileGradient<T>(outer_size, inner_size, dY_data, dX_data);
  }

 private:
  std::int32_t GetArgFromTensor(const Tensor& tensor) {
    CAFFE_ENFORCE(
        tensor.IsType<std::int32_t>() || tensor.IsType<std::int64_t>());
    std::int32_t val = -1;
    if (tensor.IsType<std::int32_t>()) {
      context_.template CopyToCPU<std::int32_t>(
          1, tensor.data<std::int32_t>(), &val);
    } else if (tensor.IsType<std::int64_t>()) {
      std::int64_t val_int64;
      context_.template CopyToCPU<std::int64_t>(
          1, tensor.data<std::int64_t>(), &val_int64);
      val = static_cast<std::int32_t>(val_int64);
    }
    return val;
  }

  template <typename T>
  bool DoTileGradient(
      const int outer_size,
      const int inner_size,
      const T* dY,
      T* dX) {
    if (inner_size == 1) {
      const std::array<int, 2> dY_dims = {outer_size, tiles_};
      const std::array<int, 2> dX_dims = {outer_size, 1};
      math::ReduceSum<T, Context>(
          2, dY_dims.data(), dX_dims.data(), T(1), dY, dX, &context_);
    } else {
      math::CopyMatrix<T, Context>(
          outer_size,
          inner_size,
          dY,
          inner_size * tiles_,
          dX,
          inner_size,
          &context_);
      for (const auto i : c10::irange(outer_size)) {
        const T* dY_ptr = dY + i * tiles_ * inner_size;
        T* dX_ptr = dX + i * inner_size;
        for (const auto j : c10::irange(1, tiles_)) {
          math::Add<T, Context>(
              inner_size, dX_ptr, dY_ptr + j * inner_size, dX_ptr, &context_);
        }
      }
    }
    return true;
  }

  std::int32_t tiles_;
  std::int32_t axis_;

  Tensor ones_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TILE_OP_H_
