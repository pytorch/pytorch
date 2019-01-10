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

#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"

CAFFE2_DEFINE_string(init_net, "", "The given path to the init protobuffer.");
CAFFE2_DEFINE_string(
    predict_net,
    "",
    "The given path to the predict protobuffer.");

namespace caffe2 {

void run() {
  if (FLAGS_init_net.empty()) {
    LOG(FATAL) << "No init net specified. Use --init_net=/path/to/net.";
  }
  if (FLAGS_predict_net.empty()) {
    LOG(FATAL) << "No predict net specified. Use --predict_net=/path/to/net.";
  }
  caffe2::NetDef init_net, predict_net;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));
  // Can be large due to constant fills
  VLOG(1) << "Init net: " << ProtoDebugString(init_net);
  LOG(INFO) << "Predict net: " << ProtoDebugString(predict_net);
  auto predictor = caffe2::make_unique<Predictor>(init_net, predict_net);
  LOG(INFO) << "Checking that a null forward-pass works";
  Predictor::TensorVector inputVec, outputVec;
  predictor->run(inputVec, &outputVec);
  CAFFE_ENFORCE_GT(outputVec.size(), 0);
}
}

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  // This is to allow us to use memory leak checks.
  caffe2::ShutdownProtobufLibrary();
  return 0;
}
