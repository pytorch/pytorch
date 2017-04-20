// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/net.h"
#pragma once

namespace caffe2 {

void testMPSCNN();
void compareModels(const NetDef& initNet, NetDef predictNet);
}
