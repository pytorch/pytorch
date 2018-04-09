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

#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPConvOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
  }
  virtual ~IDEEPConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);
    auto Y_dims = CalcOutputDims(X, filter.get_dim(0));

    CAFFE_ENFORCE(4 == X.ndims());
    CAFFE_ENFORCE(4 == filter.ndims());
    CAFFE_ENFORCE(filter.get_dim(2) == kernel_h());
    CAFFE_ENFORCE(filter.get_dim(3) == kernel_w());
    CAFFE_ENFORCE(
        X.get_dim(1) == filter.get_dim(1) * group_,
        "Convolution op: input channels does not match: # of input channels ",
        X.get_dim(1),
        " is not equal to kernel channels * group:",
        filter.get_dim(1),
        "*",
        group_);

    if (InputSize() > BIAS) {
      ideep::convolution_forward::compute(
          X, filter, Input(BIAS), Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_);
    } else {
      ideep::convolution_forward::compute(
          X, filter, Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_);
    }

    return true;
  }

 private:

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(Conv, IDEEPConvOp);

}  // namespace caffe2
