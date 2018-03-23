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

#include "caffe2/core/common.h"
#include "caffe2/onnx/backend_rep.h"

#include <iostream>

namespace caffe2 { namespace onnx {

void Caffe2BackendRep::CheckInit() {
  if (!predictor_) {
    predictor_ = caffe2::make_unique<caffe2::Predictor>(init_net_, pred_net_);
    init_net_.Clear();
    pred_net_.Clear();
  }
}


void Caffe2BackendRep::Run(
    const caffe2::Predictor::TensorVector& inputs,
    caffe2::Predictor::TensorVector* outputs) {
  CheckInit();
  predictor_->run(inputs, outputs);
}

void Caffe2BackendRep::RunMap(
    const caffe2::Predictor::TensorMap& inputs,
    caffe2::Predictor::TensorVector* outputs) {
  CheckInit();
  predictor_->run_map(inputs, outputs);
}

}}
