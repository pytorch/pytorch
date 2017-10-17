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
#include "caffe2/core/net.h"

namespace caffe2 {
static constexpr const char* kMPSCNNReadCountArg = "__mpscnn_read_count__";
static constexpr const char* kMPSCNNOutputIsTempImageArg = "__mpscnn_output_is_temp_img__";
static constexpr const int kMetalMaxTextureArrLength = 2048;
// We currently only try to convert a fixed set of operators that handle a subset of a full
// CNN. We also only run when MPSCNN is available, provides a speedup.
// On failure, returns false. On success, returns true, and sets the MPSCNN net in the output
// parameter.

bool tryConvertToMPSCNN(const NetDef& initNet, const NetDef& predictNet, NetDef* mpscnnPredictNet);

// Exposed for testing.
NetDef annotateDefWithReadCounts(const NetDef& net);
NetDef rewriteForMetal(const NetDef& net);
NetDef runMPSCNNFusion(const NetDef& net);
void dumpDef(const NetDef& d);
void mpscnnRecordExecutionFinish();
} // namespace caffe2
