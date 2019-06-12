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

#include <functional>

#include "fully_connected_fake_lowp_op.h"

namespace caffe2 {

// IEEE FP16
REGISTER_CPU_OPERATOR_WITH_ENGINE(FC, FAKE_FP16,
  FullyConnectedFakeLowpFPOp<fp32_to_fp16, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(FCGradient, FAKE_FP16,
  FullyConnectedGradientFakeLowpFPOp<fp32_to_fp16, CPUContext>);

// BFLOAT 16
REGISTER_CPU_OPERATOR_WITH_ENGINE(FC, FAKE_BFP_16,
  FullyConnectedFakeLowpFPOp<fp32_to_bfp16, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(FCGradient, FAKE_BFP_16,
  FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp16, CPUContext>);

// BFLOAT 24 (chop the least significant 8 bits)
REGISTER_CPU_OPERATOR_WITH_ENGINE(FC, FAKE_BFP_24,
  FullyConnectedFakeLowpFPOp<fp32_to_bfp24, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(FCGradient, FAKE_BFP_24,
  FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp24, CPUContext>);

// BFLOAT 14 (chop 2 extra bits from BFLOAT 16)
REGISTER_CPU_OPERATOR_WITH_ENGINE(FC, FAKE_BFP_14,
  FullyConnectedFakeLowpFPOp<fp32_to_bfp14, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(FCGradient, FAKE_BFP_14,
  FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp14, CPUContext>);

// BFLOAT16 with rounding
REGISTER_CPU_OPERATOR_WITH_ENGINE(FC, FAKE_BFP_16_ROUND,
  FullyConnectedFakeLowpFPOp<fp32_to_bfp16_round, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(FCGradient, FAKE_BFP_16_ROUND,
  FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp16_round, CPUContext>);

} // namespace caffe2
