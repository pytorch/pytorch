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

#include "caffe2/operators/shape_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Shape, ShapeOp<CPUContext>);

OPERATOR_SCHEMA(Shape)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims().size());
      out[0].set_data_type(TensorProto::INT32);
      return out;
    })
    .SetDoc("Produce a 1D int64 tensor with the shape of the input tensor.");

SHOULD_NOT_DO_GRADIENT(Shape);

} // namespace caffe2
