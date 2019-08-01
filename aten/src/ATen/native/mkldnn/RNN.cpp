#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor _mkldnn_rnn_flatten_weight(TensorList weight, bool has_bias,
    int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional) {
  AT_ERROR("_mkldnn_rnn_flatten_weight: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> _mkldnn_rnn(const Tensor& input,
    const Tensor& flatten_weight, bool has_bias, const Tensor& hx, const Tensor& cx,
    int64_t mode, int64_t input_size, int64_t hidden_size, int64_t num_layers,
    bool batch_first, bool bidirectional) {
  AT_ERROR("_mkldnn_rnn: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_ENABLED

#include <ATen/MatrixRef.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at { namespace native {

namespace {

auto dtype = ideep::tensor::data_type::f32;

constexpr int64_t ldigo_shuffle_dim = 1;
constexpr int64_t ldgoi_shuffle_dim = 0;
constexpr int64_t ldgo_shuffle_dim = 0;

// MKLDNN expects weight in ldigo format while PyTorch stores weight in ldgoi format
//   ldigo: {num_layers, num_directions, input_size, num_gates, hidden_size}
//   ldgoi: {num_layers, num_directions, num_gates, hidden_size, input_size}
//
// MKLDNN GRU gate order is different from PyTorch's which requires gates shuffle
//   (reset, input, new): MKLDNN
//   (input, reset, new): PyTorch
//
Tensor _shuffle_weight(const Tensor& weight, int64_t mode) {
  auto weight_t = weight.t();
  if (mode == ideep::rnn_kind::GRU) {
    std::vector<Tensor> gates = weight_t.chunk(3, ldigo_shuffle_dim);
    return at::cat({gates[1], gates[0], gates[2]}, ldigo_shuffle_dim);
  }
  return weight_t.contiguous();
};

// MKLDNN GRU bias has 4 gates instead of 3
// (let rt,zt,nt be reset,input,new)
//
//  (PyTorch GRU bias)     (MKLDNN GRU bias)
//
//  bias_ih    bias_hh          bias
//  +-----+    +-----+       +---------+
//  | rt1 |    | rt2 |       | zt1+zt2 |
//  |-----|    |-----|       |---------|
//  | zt1 |    | zt2 |       | rt1+rt2 |
//  |-----|    |-----|       |---------|
//  | nt1 |    | nt2 |       |   nt1   |
//  +-----+    +-----+       |---------|
//                           |   nt2   |
//                           +---------+
//
Tensor _shuffle_bias(const Tensor& bias_ih, const Tensor& bias_hh, int64_t mode) {
  if (mode == ideep::rnn_kind::GRU) {
    std::vector<Tensor> b1 = bias_ih.chunk(3, ldgo_shuffle_dim);
    std::vector<Tensor> b2 = bias_hh.chunk(3, ldgo_shuffle_dim);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, ldgo_shuffle_dim);
  }
  return bias_ih + bias_hh;
};

inline int64_t get_num_gates(int64_t mode) {
  return (mode == ideep::rnn_kind::LSTM) ? 4
      : (mode == ideep::rnn_kind::GRU) ? 3 : 1;
}

inline int64_t get_num_biases(int64_t mode) {
  return (mode == ideep::rnn_kind::LSTM || mode == ideep::rnn_kind::GRU) ? 4 : 1;
}

inline int64_t get_num_states(int64_t mode) {
  return (mode == ideep::rnn_kind::LSTM ? 2 : 1);
}

// MKLDNN RNN weight format
//   weight_ih (ldigo): {num_layers, num_directions, input_size, num_gates, hidden_size}
//   weight_hh (ldigo): {num_layers, num_directions, hidden_size, num_gates, hidden_size}
//   bias (ldgo): {num_layers, num_directions, num_biases, hidden_size}
//
// return vector holds weight ideep::tensor per layer per direction (num_layers = 1,
// num_directions = 1):
// {
//   weight_ih_00, weight_hh_00, bias_00, /* layer = 0, direction = 0 */
//   weight_ih_01, weight_hh_01, bias_01, /* layer = 0, direction = 1 */
//   ..., ..., ...,
//   weight_ih_ld, weight_hh_ld, bias_ld /* layer = ld, direction = d */
// }
//
std::vector<ideep::tensor> get_weight_itensors(const Tensor& flatten_weight,
    int64_t weight_stride0, int64_t mode, int64_t input_size, int64_t hidden_size,
    int64_t num_layers, int64_t num_directions) {
  std::vector<ideep::tensor> weights;
  weights.reserve(num_layers * num_directions * weight_stride0);

  auto num_gates = get_num_gates(mode);
  auto num_biases = get_num_biases(mode);

  auto base = flatten_weight.data<float>();
  int64_t offset = 0;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;
      auto index = layer * num_directions + direction;

      std::vector<int64_t> weight_ih_size{1, 1, layer_input_size, num_gates, hidden_size};
      std::vector<int64_t> weight_hh_size{1, 1, hidden_size, num_gates, hidden_size};
      std::vector<int64_t> bias_size{1, 1, num_biases, hidden_size};
      for (std::vector<int64_t> sz : {weight_ih_size, weight_hh_size, bias_size}) {
        weights.emplace_back(ideep::tensor{{{sz.cbegin(), sz.cend()}, dtype}, base + offset});
        offset += std::accumulate(sz.cbegin(), sz.cend(), 1, std::multiplies<int64_t>());
      }
    }
  }

  return weights;
};

} // anonymous namespace 

Tensor _mkldnn_rnn_flatten_weight(TensorList weight, bool has_bias,
    int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional) {
  int64_t num_directions = bidirectional ? 2 : 1;

  int64_t weight_stride0 = has_bias ? 4 : 2;
  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  int64_t flatten_weight_stride0 = has_bias ? 3 : 2;
  std::vector<Tensor> flatten_weight_arr;
  flatten_weight_arr.reserve(num_layers * num_directions * flatten_weight_stride0);

  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];

      flatten_weight_arr.emplace_back(_shuffle_weight(layer_weights[0], mode).view(-1));
      flatten_weight_arr.emplace_back(_shuffle_weight(layer_weights[1], mode).view(-1));
      if (has_bias) {
        flatten_weight_arr.emplace_back(_shuffle_bias(layer_weights[2], layer_weights[3], mode));
      } else {
        auto bias = at::zeros({get_num_biases(mode) * hidden_size});
      }
    }
  }

  return at::cat(flatten_weight_arr, 0);
}

std::tuple<Tensor, Tensor, Tensor> _mkldnn_rnn(const Tensor& input,
    const Tensor& flatten_weight, bool has_bias, const Tensor& hx, const Tensor& cx,
    int64_t mode, int64_t input_size, int64_t hidden_size, int64_t num_layers,
    bool batch_first, bool bidirectional) {
  TORCH_CHECK(!batch_first, "_mkldnn_rnn: don't support batch first input");

  int64_t seq_length = input.size(0);
  int64_t mini_batch = input.size(1);
  int64_t num_directions = bidirectional ? 2 : 1;

  auto x = itensor_from_mkldnn(input);
  auto hx_ = itensor_from_mkldnn(hx);

  ideep::tensor hidden_x;
  if (mode == ideep::rnn_kind::LSTM) {
    auto cx_ = itensor_from_mkldnn(cx);
    std::vector<ideep::tensor> hidden_arr{hx_, cx_};
    ideep::concat::compute<AllocForMKLDNN>(hidden_arr, 1, hidden_x);
  } else {
    hidden_x = hx_;
  }

  auto is_single_layer = num_layers * num_directions == 1;
  std::vector<ideep::tensor> hidden_x_arr{hidden_x};
  if (!is_single_layer) {
    std::vector<int32_t> split_sizes(num_layers * num_directions, 1);
    hidden_x_arr = ideep::spliter::compute<AllocForMKLDNN>(hidden_x, split_sizes, 0, false);
  }

  // MKLDNN hidden state format
  //   hidden_x (ldsnc): {num_layers, num_directions, num_states, mini_batch, hidden_size}
  //   hidden_y (ldsnc): {num_layers, num_directions, num_states, mini_batch, hidden_size}
  std::vector<int64_t> _hidden_size{1, 1, get_num_states(mode), mini_batch, hidden_size};
  for (auto& hidden : hidden_x_arr) {
    hidden.reinit({{_hidden_size.cbegin(), _hidden_size.cend()}, dtype});
  }
  std::vector<ideep::tensor> hidden_y_arr(hidden_x_arr.size());

  int64_t weight_stride0 = has_bias ? 3 : 2;
  auto weight_arr = get_weight_itensors(flatten_weight, weight_stride0,
      mode, input_size, hidden_size, num_layers, num_directions);

  auto layer_x = x;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    std::vector<ideep::tensor> layer_y(num_directions);
    std::vector<int64_t> _output_size{seq_length, mini_batch, hidden_size};
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto reverse = (direction > 0);
      ideep::rnn_forward::compute<AllocForMKLDNN>(
          /* input     */ layer_x,
          /* {hx, cx}  */ hidden_x_arr[index],
          /* weight_ih */ weight_arr[index * 3],
          /* weight_hh */ weight_arr[index * 3 + 1],
          /* bias      */ weight_arr[index * 3 + 2],
          /* output    */ {_output_size.cbegin(), _output_size.cend()}, layer_y[direction],
          /* {hy, cy}  */ {_hidden_size.cbegin(), _hidden_size.cend()}, hidden_y_arr[index],
          /* rnn_kind  */ static_cast<ideep::rnn_kind>(mode),
          /* direction */ reverse);
    }
    if (num_directions == 1) {
      layer_x = layer_y[0];
    } else {
      ideep::concat::compute<AllocForMKLDNN>(layer_y, 2, layer_x);
    }
  }
  auto y = layer_x;

  for (auto& hidden : hidden_y_arr) {
    hidden.reinit({{1, get_num_states(mode) * mini_batch, hidden_size}, dtype});
  }

  ideep::tensor hidden_y, hy_, cy_;
  if (is_single_layer) {
    hidden_y = hidden_y_arr[0];
  } else {
    ideep::concat::compute<AllocForMKLDNN>(hidden_y_arr, 0, hidden_y);
  }

  // split mkldnn hidden into {hy, cy}
  if (mode == ideep::rnn_kind::LSTM) {
    std::vector<int32_t> split_sizes{mini_batch, mini_batch};
    auto hidden_outputs = ideep::spliter::compute<AllocForMKLDNN>(hidden_y, split_sizes, 1, false);
    hy_ = hidden_outputs[0];
    cy_ = hidden_outputs[1];
  } else {
    hy_ = hidden_y;
  }

  auto output = new_with_itensor_mkldnn(std::move(y), input.options());
  auto hy = new_with_itensor_mkldnn(std::move(hy_), input.options());
  auto cy = new_with_itensor_mkldnn(std::move(cy_), input.options());

  return std::make_tuple(output, hy, cy);
}

}} // at::native

#endif
