#include "caffe2/mobile/contrib/ios/ios_caffe_predictor.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/tensor.h"

#if defined(CAFFE2_USE_MPSCNN) && defined(C10_MOBILE)
#include "caffe2/mobile/contrib/ios/mpscnn/mpscnn.h"
#endif

C10_DECLARE_bool(caffe2_force_shared_col_buffer);

Caffe2IOSPredictor* Caffe2IOSPredictor::NewCaffe2IOSPredictor(const caffe2::NetDef& init_net,
                                                              const caffe2::NetDef& predict_net,
                                                              bool disableMultithreadProcessing,
                                                              bool allowMetalOperators) {
  caffe2::NetDef metal_predict_net;
  bool usingMetalOperators = false;
#if defined(CAFFE2_USE_MPSCNN) && defined(C10_MOBILE)
  if (allowMetalOperators) {
    caffe2::dumpDef(predict_net);
    if (caffe2::tryConvertToMPSCNN(init_net, predict_net, &metal_predict_net)) {
      LOG(INFO) << "Successfully converted to MPSCNN";
      caffe2::dumpDef(metal_predict_net);
      usingMetalOperators = true;
    } else {
      LOG(ERROR) << "Failed converting model to MPSCNN";
    }
  }
#endif

  return new Caffe2IOSPredictor(init_net,
                                usingMetalOperators ? metal_predict_net : predict_net,
                                disableMultithreadProcessing,
                                usingMetalOperators);
}

Caffe2IOSPredictor::Caffe2IOSPredictor(const caffe2::NetDef& init_net,
                                       const caffe2::NetDef& predict_net,
                                       bool disableMultithreadProcessing,
                                       bool usingMetalOperators)
    : usingMetalOperators(usingMetalOperators), predictor_(init_net, predict_net) {
#ifdef C10_MOBILE
  if (disableMultithreadProcessing) {
    caffe2::ThreadPool* threadpool = predictor_.ws()->GetThreadPool();
    if (threadpool != nullptr) {
      threadpool->setMinWorkSize(std::numeric_limits<size_t>::max());
    }
  }
#endif
}

void Caffe2IOSPredictor::run(const Tensor& inData, Tensor& outData, std::string& errorMessage) {
  FLAGS_caffe2_force_shared_col_buffer = true;
  caffe2::Tensor input = caffe2::empty(inData.dims, at::dtype<uint8_t>().device(caffe2::CPU));
  input.ShareExternalPointer(inData.data);
  caffe2::Predictor::TensorList input_vec;
  input_vec.emplace_back(std::move(input));
  caffe2::Predictor::TensorList output_vec;
  try {
    predictor_(input_vec, &output_vec);
  } catch (const std::exception& e) {
    std::string error = e.what();
    errorMessage.swap(error);
    return;
  }
  caffe2::Tensor* output = &output_vec.front();
  outData.data = output->mutable_data<uint8_t>();
  outData.dims = output->sizes().vec();
}
