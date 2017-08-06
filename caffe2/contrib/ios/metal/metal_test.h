// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
  void testMetal();
  void compareModels(const NetDef& initNet, NetDef predictNet);
}
