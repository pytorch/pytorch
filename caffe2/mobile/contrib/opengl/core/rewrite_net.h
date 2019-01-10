
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
