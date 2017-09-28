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
#include "GLPredictor.h"
#include "caffe2/core/predictor.h"

namespace caffe2 {
bool tryConvertToOpenGL(const NetDef& initNet,
                        const NetDef& predictNet,
                        NetDef* glPredictNet,
                        bool useTextureInput = false,
                        bool useTiling       = false,
                        bool runFusion       = true);

// Exposed for testing
NetDef rewritePredictNetForOpenGL(const NetDef& predictNet,
                                  bool useTextureInput = false,
                                  bool useTiling       = false,
                                  bool runFusion       = true);
void dumpDefForOpenGL(const NetDef& net);
} // namespace caffe2
