#include "NNPACK.h"

#include "TH/TH.h"

namespace torch {
namespace nnpack {

// Stolen from Caffe2
static pthreadpool_t nnpack_threadpool_ = nullptr;

pthreadpool_t nnpack_threadpool() {
  if (nnpack_threadpool_ == nullptr) {
    enum nnp_status nnpack_status = nnp_initialize();
    if (nnpack_status != nnp_status_success) throw std::runtime_error("could not initialize NNPack");
    nnpack_threadpool_ = pthreadpool_create(4);
  }
  return nnpack_threadpool_;
}

void convolutionOutput(
    at::Tensor& input,
    at::Tensor& weight,
    at::Tensor& bias,
    const std::vector<int>& padding,
    at::Tensor& output) {
  // Setup parameters for the NNPack convolution output function call

  // For now, we use the default algorithm
  auto algorithm = nnp_convolution_algorithm_auto;

  // Our input Tensor must be in the form N,C,H,W
  if (input.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D input Tensor N,C,H,W");
  }

  // Our weight Tensor must be in the form oC,iC,kH,kW
  if (weight.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D weight Tensor oC,iC,kH,kW");
  }

  const size_t batchSize = input.sizes()[0];
  const size_t inputChannels = input.sizes()[1];
  const size_t outputChannels = weight.sizes()[0];
  const struct nnp_size input_size = {
    .width = input.sizes()[3],
    .height = input.sizes()[2]
  };
  const struct nnp_padding input_padding = {
    .top = padding[0],
    .right = padding[1],
    .bottom = padding[0],
    .left = padding[1]
  };
  const struct nnp_size kernel_size = {
    .width = weight.sizes()[3],
    .height = weight.sizes()[2]
  };

  // If we don't have a defined bias Tensor, we need to create one filled with zeroes
  auto bdef = bias.defined();
  auto bias_ = bdef ? bias : input.type().zeros({outputChannels});

  // Note: we assume that the output is shaped correctly, probably should add an assert

  auto status = nnp_convolution_output(
      algorithm,
      batchSize,
      inputChannels,
      outputChannels,
      input_size,
      input_padding,
      kernel_size,
      (THFloatTensor_data((THFloatTensor*)input.unsafeGetTH(false))),
      (THFloatTensor_data((THFloatTensor*)weight.unsafeGetTH(false))),
      (THFloatTensor_data((THFloatTensor*)bias_.unsafeGetTH(false))),
      (THFloatTensor_data((THFloatTensor*)output.unsafeGetTH(false))),
      nullptr, // workspace_buffer
      nullptr, // workspace_size
      nnp_activation_identity,
      nullptr, // activation parameters
      nnpack_threadpool(),
      nullptr  // profile
  );

  if (status != nnp_status_success) throw std::runtime_error("err");
}

} // torch::nnpack
} // torch
