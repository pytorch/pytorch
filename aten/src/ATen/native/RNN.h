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

}} // namespace at::native

