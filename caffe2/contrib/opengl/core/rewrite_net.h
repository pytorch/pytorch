// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "GLPredictor.h"
#include "caffe2/core/predictor.h"

namespace caffe2 {
bool tryConvertToOpenGL(const NetDef& initNet,
                        const NetDef& predictNet,
                        NetDef* glPredictNet,
                        bool useTextureInput = false);

// Exposed for testing
NetDef rewritePredictNetForOpenGL(const NetDef& predictNet,
                                  bool useTextureInput = false,
                                  bool useTiling = false);
void dumpDefForOpenGL(const NetDef& net);
} // namespace caffe2
