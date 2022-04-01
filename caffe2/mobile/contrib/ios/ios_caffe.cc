
#include "ios_caffe.h"
#include "caffe2/core/tensor.h"
#include "caffe2/mobile/contrib/ios/ios_caffe_predictor.h"
#include "caffe2/predictor/predictor.h"

Caffe2IOSPredictor* MakeCaffe2Predictor(const std::string& init_net_str,
                                        const std::string& predict_net_str,
                                        bool disableMultithreadProcessing,
                                        bool allowMetalOperators,
                                        std::string& errorMessage) {
  caffe2::NetDef init_net, predict_net;
  init_net.ParseFromString(init_net_str);
  predict_net.ParseFromString(predict_net_str);

  Caffe2IOSPredictor* predictor = NULL;
  try {
    predictor = Caffe2IOSPredictor::NewCaffe2IOSPredictor(
        init_net, predict_net, disableMultithreadProcessing, allowMetalOperators);
  } catch (const std::exception& e) {
    std::string error = e.what();
    errorMessage.swap(error);
    return NULL;
  }
  return predictor;
}

void GenerateStylizedImage(std::vector<float>& originalImage,
                           const std::string& init_net_str,
                           const std::string& predict_net_str,
                           int height,
                           int width,
                           std::vector<float>& dataOut) {
  caffe2::NetDef init_net, predict_net;
  init_net.ParseFromString(init_net_str);
  predict_net.ParseFromString(predict_net_str);
  caffe2::Predictor p(init_net, predict_net);

  std::vector<int> dims({1, 3, height, width});
  caffe2::Tensor input(caffe2::CPU);
  input.Resize(dims);
  input.ShareExternalPointer(originalImage.data());
  caffe2::Predictor::TensorList input_vec;
  input_vec.emplace_back(std::move(input));
  caffe2::Predictor::TensorList output_vec;
  p(input_vec, &output_vec);
  assert(output_vec.size() == 1);
  caffe2::TensorCPU* output = &output_vec.front();
  // output is our styled image
  float* outputArray = output->mutable_data<float>();
  dataOut.assign(outputArray, outputArray + output->size());
}
