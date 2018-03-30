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

#include "caffe2/core/common.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "onnx/onnx_pb.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace caffe2 {
  class TensorRTTransformer {
   public:
    TensorRTTransformer(
        size_t max_batch_size,
        size_t max_workspace_size,
        int verbosity,
        bool debug_builder)
        : max_batch_size_(max_batch_size),
          max_workspace_size_(max_workspace_size),
          verbosity_(verbosity),
          debug_builder_(debug_builder) {}

    OperatorDef BuildTrtOp(
        const std::string& onnx_model_str,
        const std::unordered_map<std::string, std::vector<int>>&
            output_size_hints);

    void Transform(
        NetDef* init_net,
        NetDef* pred_net,
        const std::unordered_map<std::string, TensorShape>& shape_hints);

   private:
    void ClusterToTrtOp(
        const NetDef& init_net,
        const NetDef& pred_net,
        int start,
        int end,
        const std::unordered_set<std::string>& weights,
        const std::unordered_map<std::string, TensorShape>& shape_hints,
        ::ONNX_NAMESPACE::ModelProto* model,
        std::vector<OperatorDef>* new_ops);

    // TensorRT params
    size_t max_batch_size_{50};
    size_t max_workspace_size_{1024 * 1024 * 2};
    int verbosity_{2};
    bool debug_builder_{true};
  };
}
