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

#pragma once

#include "caffe2/core/predictor.h"
#include "caffe2/proto/caffe2.pb.h"

#include <memory>
#include <string>
#include <vector>

namespace caffe2 { namespace onnx {
class Caffe2BackendRep {
 public:
  void Run(
      const caffe2::Predictor::TensorVector& inputs,
      caffe2::Predictor::TensorVector* outputs);
  void RunMap(
      const caffe2::Predictor::TensorMap& inputs,
      caffe2::Predictor::TensorVector* outputs);

  caffe2::NetDef& init_net() {
    return init_net_;
  }
  caffe2::NetDef& pred_net() {
    return pred_net_;
  }
  std::vector<std::string>& uninitialized_inputs() {
    return uninitialized_inputs_;
  }

  const caffe2::NetDef& init_net() const {
    return init_net_;
  }
  const caffe2::NetDef& pred_net() const {
    return pred_net_;
  }
  const std::vector<std::string>& uninitialized_inputs() const {
    return uninitialized_inputs_;
  }

 private:
  void CheckInit();

  caffe2::NetDef init_net_;
  caffe2::NetDef pred_net_;
  std::vector<std::string> uninitialized_inputs_;
  std::unique_ptr<caffe2::Predictor> predictor_{nullptr};
};
}}
