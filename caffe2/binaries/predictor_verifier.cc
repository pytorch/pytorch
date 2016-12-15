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
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
