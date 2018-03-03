/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_H_
#define CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_H_

#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/locally_connected_op_util.h"

namespace caffe2 {

template <typename T, class Context>
class LocallyConnectedOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);

  LocallyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
    // Since this is the default locally connected implementation, we will
    // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group locally connected only supports NCHW order right now.");
  }

  ~LocallyConnectedOp() = default;

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  void RunOnDeviceWithOrderNCHWImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* bias_data,
      T* Y_data,
      Tensor<Context>* column_buffer,
      Tensor<Context>* column_transposed_buffer,
      Tensor<Context>* output_buffer);

  void RunOnDeviceWithOrderNHWCImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* bias_data,
      T* Y_data,
      Tensor<Context>* column_buffer,
      Tensor<Context>* column_transposed_buffer,
      Tensor<Context>* Y_transposed_buffer);

  void SetColumnBufferShape(
      const int N,
      const int C,
      const int kernel_dim,
      const std::vector<int>& output_image_dims,
      std::vector<int>* column_dims,
      std::vector<int>* column_transposed_dims);

  void SetYTranposedBufferShape(
      const std::vector<int>& Y_dims,
      std::vector<int>* Y_transposed_dims);

  Tensor<Context> bias_multiplier_;

  // Buffer.
  Tensor<Context> column_buffer_;
  Tensor<Context> column_transposed_buffer_;
  Tensor<Context> Y_transposed_buffer_;

  // Dims devices.
  Tensor<Context> X_dims_device_;
  Tensor<Context> column_dims_device_;
  Tensor<Context> column_transposed_dims_device_;
  Tensor<Context> column_axes_device_;
  Tensor<Context> Y_dims_device_;
  Tensor<Context> Y_transposed_dims_device_;
  Tensor<Context> Y_transposed_axes_device_;

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

template <typename T, class Context>
class LocallyConnectedGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);

  LocallyConnectedGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group locally connected only supports NCHW order right now.");
  }

  ~LocallyConnectedGradientOp() = default;

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  void RunOnDeviceWithOrderNCHWImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* dY_data,
      T* dfilter_data,
      T* dX_data,
      T* dbias_data,
      Tensor<Context>* column_buffer,
      Tensor<Context>* column_transposed_buffer,
      Tensor<Context>* dY_transposed_buffer);

  void RunOnDeviceWithOrderNHWCImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* dY_data,
      T* dfilter_data,
      T* dX_data,
      T* dbias_data,
      Tensor<Context>* column_buffer,
      Tensor<Context>* column_transposed_buffer,
      Tensor<Context>* dY_transposed_buffer);

  void SetColumnBufferShape(
      const int N,
      const int C,
      const int kernel_dim,
      const std::vector<int>& output_image_dims,
      std::vector<int>* column_dims,
      std::vector<int>* column_transposed_dims);

  void SetDYTranposedBufferShape(
      const std::vector<int>& dY_dims,
      std::vector<int>* dY_transposed_dims);

  const bool no_bias_;

  Tensor<Context> bias_multiplier_;

  // Buffer.
  Tensor<Context> column_buffer_;
  Tensor<Context> column_transposed_buffer_;
  Tensor<Context> dY_transposed_buffer_;

  // Dims devices.
  Tensor<Context> X_dims_device_;
  Tensor<Context> column_dims_device_;
  Tensor<Context> column_transposed_dims_device_;
  Tensor<Context> column_axes_device_;
  Tensor<Context> column_transposed_axes_device_;
  Tensor<Context> dY_dims_device_;
  Tensor<Context> dY_transposed_dims_device_;
  Tensor<Context> dY_axes_device_;

  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_H_
