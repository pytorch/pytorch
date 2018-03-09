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
#include "caffe2/mobile/contrib/arm-compute/core/net_gl.h"
#include <unordered_set>

namespace caffe2 {
bool tryConvertToOpenGL(const NetDef& predictNet,
                        NetDef* glPredictNet,
                        bool runFusion,
                        std::unordered_set<std::string> cpuOps);

// Exposed for testing
NetDef rewritePredictNetForOpenGL(const NetDef& predictNet,
                                  bool runFusion,
                                  std::unordered_set<std::string> cpuOps);
void dumpDefForOpenGL(const NetDef& net);
} // namespace caffe2
