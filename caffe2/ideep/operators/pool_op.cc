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

class IDEEPPoolOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {
    CAFFE_ENFORCE(
        (dilation_h() == 1) && (dilation_w() == 1),
        "Pooling op does not support dilation right now.");
    if (!global_pooling_) {
      CAFFE_ENFORCE(
          pad_t() < kernel_h() && pad_b() < kernel_h() &&
              pad_l() < kernel_w() && pad_r() < kernel_w(),
          "Pad should be smaller than kernel.");
    }
    // Figure out the pooling descriptor.
    if (operator_def.type().substr(0, 7) == "MaxPool") {
      algo_ = ialgo::pooling_max;
    } else if (operator_def.type().substr(0, 11) == "AveragePool") {
      algo_ = ialgo::pooling_avg;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
    }
  }
  virtual ~IDEEPPoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);
    auto Y_dims = CalcOutputDims(X, X.get_dim(1));

    ideep::pooling_forward::compute(X, Y_dims, *Y,
        stride_, kernel_, pad_tl(), pad_br(), algo_);

    return true;
  }

 private:
  ialgo algo_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(MaxPool, IDEEPPoolOp);
REGISTER_IDEEP_OPERATOR(AveragePool, IDEEPPoolOp);

} // namespace caffe2
