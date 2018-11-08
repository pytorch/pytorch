
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
