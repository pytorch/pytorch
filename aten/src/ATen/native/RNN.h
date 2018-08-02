#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using lstm_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&, TensorList, TensorList, bool, int64_t, double, bool, bool, bool);
using rnn_fn = void(*)(Tensor&, Tensor&, const Tensor&, const Tensor&, TensorList, bool, int64_t, double, bool, bool, bool);
using lstm_packed_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&, const Tensor&, TensorList, TensorList, bool, int64_t, double, bool, bool);
using rnn_packed_fn = void(*)(Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&, TensorList, bool, int64_t, double, bool, bool);

DECLARE_DISPATCH(lstm_fn, lstm_cudnn_stub);
DECLARE_DISPATCH(rnn_fn, gru_cudnn_stub);
DECLARE_DISPATCH(rnn_fn, rnn_tanh_cudnn_stub);
DECLARE_DISPATCH(rnn_fn, rnn_relu_cudnn_stub);
DECLARE_DISPATCH(lstm_packed_fn, lstm_packed_cudnn_stub);
DECLARE_DISPATCH(rnn_packed_fn, gru_packed_cudnn_stub);
DECLARE_DISPATCH(rnn_packed_fn, rnn_tanh_packed_cudnn_stub);
DECLARE_DISPATCH(rnn_packed_fn, rnn_relu_packed_cudnn_stub);

void check_device(const Tensor& input, const TensorList& params, const TensorList& hiddens) {
  auto input_device = input.device();
  bool input_device_is_cuda = input_device.is_cuda();

  auto check_tensors = [&](const std::string& name, const Tensor& t) {
    if (!t.defined()) return;
    auto t_device = t.device();
    bool t_device_is_cuda = t_device.is_cuda();
    AT_CHECK(input_device_is_cuda == t_device_is_cuda,
             "Input and ", name, " tensors are not at the same device, found input tensor at ",
             input_device.type(), " and hidden tensor at ", t_device.type());
    // implicitly check for CUDA device mismatch among tensors
    if (input_device_is_cuda && t_device_is_cuda) {
      AT_CHECK(input_device.index() == t_device.index(),
               "Input and ", name, " CUDA tensors are not at the same GPU, found input tensor ",
               "at CUDA:", input_device.index(), " and hidden tensor at CUDA:",
               t_device.index());
    }
  };

  for (auto h : hiddens) {
    check_tensors("hidden", h);
  }

  for (auto p : params) {
    // if (!p.defined()) continue;
    check_tensors("parameter", p);
  }
}

}} // namespace at::native
