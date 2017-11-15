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

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/net_test_util.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(NetTestDummy, NetTestDummyOp<CPUContext>);
REGISTER_CUDA_OPERATOR(NetTestDummy, NetTestDummyOp<CUDAContext>);

OPERATOR_SCHEMA(NetTestDummy)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

TEST(NetTest, ChainingForDifferentDevices) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out"
          type: "NetTestDummy"
          device_option {
            device_type: 1
          }
        }
        op {
          input: "out"
          output: "out2"
          type: "NetTestDummy"
          device_option {
            device_type: 1
          }
        }
        op {
          input: "out2"
          output: "out3"
          type: "NetTestDummy"
          device_option {
            device_type: 1
            cuda_gpu_id: 1
          }
        }
)DOC";
  if (HasCudaRuntime()) {
    checkChainingAndRun(spec, {{0, {0, 1, 2}}, {3, {3}}});
  }
}

} // namespace caffe2
