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

#include "caffe2/video/video_input_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(VideoInput, VideoInputOp<CPUContext>);

OPERATOR_SCHEMA(VideoInput)
    .NumInputs(0, 1)
    .NumOutputs(2)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<
                                    TensorShape>& /* unused */ /*in*/) {
      vector<TensorShape> out(2);
      ArgumentHelper helper(def);
      int batch_size = helper.GetSingleArgument<int>("batch_size", 0);
      int crop = helper.GetSingleArgument<int>("crop", -1);
      int length = helper.GetSingleArgument<int>("length", -1);
      int multiple_label = helper.GetSingleArgument<int>("multiple_label", 0);
      CHECK_GT(crop, 0);
      out[0] = CreateTensorShape(
          vector<int>{batch_size, 3, length, crop, crop}, TensorProto::FLOAT);
      if (!multiple_label) {
        out[1] =
            CreateTensorShape(vector<int>{1, batch_size}, TensorProto::INT32);
      } else {
        int num_of_labels = helper.GetSingleArgument<int>("num_of_labels", 0);
        out[1] = CreateTensorShape(
            vector<int>{batch_size, num_of_labels}, TensorProto::INT32);
      }
      return out;
    });

NO_GRADIENT(VideoInput);

} // namespace caffe2
