// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
void testOpenGL();
void compareModelsForOpenGL(const NetDef& initNet, NetDef predictNet);
int runModelBenchmarks(caffe2::NetDef& init_net,
                       caffe2::NetDef& predict_net,
                       int warm_up_runs,
                       int main_runs,
                       int channel,
                       int height,
                       int width,
                       std::string input_type,
                       std::string input_order,
                       std::string engine,
                       bool run_individual = false,
                       bool use_texture_input = false);
} // namespace caffe2
