
#include "caffe2/core/net.h"
#pragma once

namespace caffe2 {

void testMPSCNN();
void compareModels(const NetDef& initNet, NetDef predictNet);
void verifyRewrite(const NetDef& initNet, const NetDef& net, std::vector<int> inputDims);
} // namespace caffe2
