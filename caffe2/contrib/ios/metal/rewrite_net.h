// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "caffe2/core/predictor.h"

namespace caffe2 {
bool tryConvertToMetal(const NetDef &initNet, const NetDef &predictNet, NetDef *metalInitNet, NetDef *metalPredictNet);

// Exposed for testing
NetDef rewritePredictNetForMetal(const NetDef &predictNet, const std::string engine);
NetDef rewriteInitNetForMetal(const NetDef &initNet, const NetDef &predictNet, const std::string engine);
void dumpDef(NetDef &net);
}
