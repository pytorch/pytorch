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


#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
void testOpenGL();
void compareModelsForOpenGL(std::string name,
                            const NetDef& initNet,
                            NetDef predictNet,
                            int width,
                            int height,
                            int channel,
                            std::string input_type,
                            std::string input_order);

void compareBatchedToTiledModels(std::string name,
                                 const NetDef& initNet,
                                 NetDef predictNet,
                                 int width,
                                 int height,
                                 int channel,
                                 std::string input_type,
                                 std::string input_order);

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
                       bool run_individual    = false,
                       bool use_texture_input = false,
                       bool use_tiling        = false,
                       bool run_fusion        = true);
} // namespace caffe2
