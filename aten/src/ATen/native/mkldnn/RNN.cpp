#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {


std::tuple<Tensor, Tensor, Tensor> lstm_mkldnn_stub(
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  AT_ERROR("lstm_mkldnn_stub: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn_layer(
const Tensor& input,
    const Tensor& w0,
    const Tensor& w1,
    const Tensor& w2,
    const Tensor& w3,
    const Tensor& hx_,
    const Tensor& cx_,
    bool reverse,
    IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
  AT_ERROR("mkldnn_rnn_layer: ATen not compiled with MKLDNN support");

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

struct RNNParams {
  ideep::rnn_kind mode;
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t hidden_size;
  int64_t num_directions;
  int64_t num_layers;
  bool batch_first;
  bool train;
  at::IntArrayRef batch_sizes;
  int64_t num_gates;
  int64_t num_bias_gates;

  RNNParams(
      const at::Tensor& input,
      at::IntArrayRef batch_sizes_,
      int64_t mode_,
      int64_t hidden_size_,
      int64_t num_layers_,
      bool bidirectional,
      bool batch_first_,
      bool train_) {
    mode = static_cast<ideep::rnn_kind>(mode_);
    batch_first = batch_first_;
    seq_length = input.size(0);
    mini_batch = input.size(1);
    input_size = input.size(2);
    hidden_size = hidden_size_;
    num_directions = bidirectional ? 2 : 1;
    num_layers = num_layers_;
    train = train_;
    batch_sizes = batch_sizes_;
    if (mode == ideep::rnn_kind::LSTM) {
      num_gates = 4;
      num_bias_gates = 4;
    } else if (mode == ideep::rnn_kind::GRU) {
      num_gates = 3;
      num_bias_gates = 4;
    } else {
      // RNN_RELU; RNN_TANH
      num_gates = 1;
      num_bias_gates = 1;
    }
  }

  bool is_input_packed() const {
    return batch_sizes.size() != 0;
  }

  // mkldnn memory descriptors
  using format = ideep::format_tag;
  using desc = ideep::tensor::desc;
  using dtype = ideep::tensor::data_type;
  desc src_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{seq_length, mini_batch, _input_size}, dtype, format::tnc};
  }
  desc src_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  desc src_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  // logical size described as ldigo
  desc weights_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }
  desc weights_layer_ldigo_desc(int64_t _input_size, dtype dtype) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype, format::ldigo};
  }
  desc weights_iter_desc(dtype dtype) const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }
  desc weights_iter_ldigo_desc(dtype dtype) const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype, format::ldigo};
  }
  desc bias_desc(dtype dtype) const {
    return {{1, 1, num_bias_gates, hidden_size}, dtype, format::ldgo};
  }
  desc dst_layer_desc(dtype dtype) const {
    return {{seq_length, mini_batch, hidden_size}, dtype, format::tnc};
  }
  desc dst_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  desc dst_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
};

std::vector<int64_t> _hidden_size(const RNNParams& rnn) {
  return {rnn.num_layers * rnn.num_directions, rnn.mini_batch, rnn.hidden_size};
}

template<bool is_single_direction>
std::vector<int64_t> _output_size(const RNNParams& rnn) {
  auto output_channels = is_single_direction ? rnn.hidden_size
                                             : rnn.hidden_size * rnn.num_directions;
  return {rnn.seq_length, rnn.mini_batch, output_channels};
}

// MKLDNN GRU gate order is different from PyTorch's which requires gates shuffle
// (let rt,zt,nt be reset, update, new gates respectively)
//
//   MKLDNN GRU weight_ih/weight_hh gates order: (zt, rt, nt)
//   PyTorch GRU weight_ih/weight_hh gates order: (rt, zt, nt)
//
// MKLDNN GRU bias has 4 gates instead of 3
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
Tensor _shuffle_weight(const Tensor& weight, int64_t fn_mode) {
  auto weight_t = weight.contiguous();
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> gates = weight_t.chunk(3, /*gates*/0);
    return at::cat({gates[1], gates[0], gates[2]}, /*gates*/0);
  }
  return weight_t;
};

Tensor _shuffle_bias(const Tensor& bias_ih, const Tensor& bias_hh, int64_t fn_mode) {
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> b1 = bias_ih.chunk(3, /*output_channels*/0);
    std::vector<Tensor> b2 = bias_hh.chunk(3, /*output_channels*/0);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, /*output_channels*/0);
  }
  return bias_ih + bias_hh;
};

// Create mkldnn memory view from ATen tensor
static inline ideep::tensor get_mkldnn_tensor(
    const Tensor& tensor, const ideep::tensor::desc& desc) {
  return {desc, tensor.data_ptr<float>()};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn_layer(const Tensor& input,
    const Tensor& w0,
    const Tensor& w1,
    const Tensor& w2,
    const Tensor& w3,
    const Tensor& hx_,
    const Tensor& cx_,
    bool reverse,
    IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
        std::cout<<"in lstm_mkldnn_rnn_layer\n";
  RNNParams rnn(
      input,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      bidirectional,
      batch_first,
      train);

    std::cout<<"1\n";
  auto output_size = _output_size</*is_single_direction*/ true>(rnn);
  auto output = at::empty(output_size, input.options());

  auto hy_ = at::empty(hx_.sizes(), hx_.options());
  auto cy_ = at::empty(cx_.sizes(), cx_.options());

  auto weight_ih = _shuffle_weight(w0, rnn.mode);
  auto weight_hh = _shuffle_weight(w1, rnn.mode);

  auto bias = has_biases
      ? _shuffle_bias(w2, w3, rnn.mode)
      : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());
std::cout<<"2\n";
  // per layer input size
  int64_t input_size = input.size(2);
  auto x = get_mkldnn_tensor(
      input,
      rnn.src_layer_desc(input_size, get_mkldnn_dtype(input.scalar_type())));
  auto hx = get_mkldnn_tensor(
      hx_, rnn.src_iter_desc(get_mkldnn_dtype(hx_.scalar_type())));
  auto cx = get_mkldnn_tensor(
      cx_, rnn.src_iter_c_desc(get_mkldnn_dtype(cx_.scalar_type())));
  auto b = get_mkldnn_tensor(
      bias, rnn.bias_desc(get_mkldnn_dtype(bias.scalar_type())));
  auto y = get_mkldnn_tensor(
      output, rnn.dst_layer_desc(get_mkldnn_dtype(output.scalar_type())));
  auto hy = get_mkldnn_tensor(
      hy_, rnn.dst_iter_desc(get_mkldnn_dtype(hy_.scalar_type())));
  auto cy = get_mkldnn_tensor(
      cy_, rnn.dst_iter_c_desc(get_mkldnn_dtype(cy_.scalar_type())));
std::cout<<"3\n";
//   ideep::tensor w1_, w2_;
  auto w1_ = get_mkldnn_tensor(weight_ih, rnn.weights_layer_desc(input_size, get_mkldnn_dtype(weight_ih.scalar_type())));
  auto w2_ = get_mkldnn_tensor(weight_hh, rnn.weights_iter_desc(get_mkldnn_dtype(weight_hh.scalar_type())));
std::cout<<"4\n";
//   ideep::lstm_forward_inference::compute(x, hx, cx, w1, w2, b, y, hy, cy, reverse);

//   std::vector<at::Tensor> result;
std::cout<<train<<"\n";
train=false;
  if (train) {
    // at::Tensor workspace = at::Tensor();
    // auto pd = ideep::lstm_forward_training::prepare(
    //     x, hx, cx, w1_, w2_, b, y, hy, cy, reverse);
    // workspace = torch_ipex::cpu::empty_aten_tensor_from_desc(
    //     pd.workspace_desc(), input.options().dtype(at::kByte));
    // ideep::tensor mkldnn_workspace;
    // mkldnn_workspace.init(
    //     pd.workspace_desc(), workspace.template data_ptr<uint8_t>());
    // ideep::lstm_forward_training::compute(
    //     pd, x, hx, cx, w1_, w2_, b, mkldnn_workspace, y, hy, cy, reverse);
    // result.reserve(4);
    // result.push_back(output);
    // result.push_back(hy_);
    // result.push_back(cy_);
    // result.push_back(workspace);
    return std::make_tuple(Tensor(), Tensor(), Tensor(), Tensor());
  } else {
    std::cout<<"before compute\n";
    ideep::lstm_forward_inference::compute(
        x, hx, cx, w1_, w2_, b, y, hy, cy, reverse);
    return std::make_tuple(output, hy_, cy_, Tensor());
    // result.reserve(4);
    // result.push_back(output);
    // result.push_back(hy_);
    // result.push_back(cy_);
    // result.push_back(Tensor());
  }
//   return result;
}

// MKLDNN RNN integration notes:
// I. Memory Formats
//   a. mkldnn will use plain formats for input, hx/cx, output, hy/cy
//      and possibly use blocked formats for weights depending shape info.
//   b. All mkldnn memorys are created (in plain format) as views on ATen tensor,
//      the weight reorder(if any) is handed automatically inside ideep (mkldnn bridge)
//
// II. MKLDNN Primitive Mapping
//   a. mkldnn rnn primitive doesn't support training with dropout or padded input sequence.
//   b. here break a single RNN module into { num_layers * num_directions } mkldnn rnn primitives
//      for future need to cover these feature gaps.
//
//TODO: a. training with dropout
//   b. padded sequence input support
//

std::tuple<Tensor, Tensor, Tensor> mkldnn_rnn(
    const Tensor& input_, TensorList weight, int64_t weight_stride0,
    const Tensor& hx_, const Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool has_biases, bool batch_first, double dropout_p,
    bool train, bool bidirectional, IntArrayRef batch_sizes) {
        std::cout<<"in lstm_mkldnn_rnn\n";
//   return std::make_tuple(Tensor(), Tensor(), Tensor());
//   TORCH_CHECK(!train || dropout_p == 0.0, "mkldnn_rnn doesn't support dropout");
  TORCH_CHECK(batch_sizes.size() == 0, "mkldnn_rnn doesn't support packed input");
  if (static_cast<ideep::rnn_kind>(mode) != ideep::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(), "mkldnn_rnn: illegal defined cx for non-LSTM RNN");
  }

//   RNNParams fn(input_, batch_sizes, mode, hidden_size, num_layers, bidirectional, batch_first, train);

  auto input = input_;
  bool is_input_packed = batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }
  input = input.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.contiguous();
//   auto cx = cx_.defined() ? cx_.contiguous() : Tensor();

//   auto hy = at::empty(_hidden_size(fn), hx.options());
//   // NB: Not allowed to return undefined tensors
//   auto cy = cx.defined() ? at::empty(_hidden_size(fn), cx.options())
//                          : at::empty({0}, hx.options());

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_directions = bidirectional ? 2 : 1;
  auto layer_input = input;
  std::vector<at::Tensor> layer_output(num_directions);
  std::vector<at::Tensor> layer_hy(num_layers * num_directions);
  std::vector<at::Tensor> layer_cy(num_layers * num_directions);
  for (int64_t layer = 0; layer < num_layers; layer++) {
    // std::vector<Tensor> layer_output(num_directions);
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      TORCH_CHECK(layer_weights.size() == 2 || layer_weights.size() == 4);
      auto layer_hx = hx[index];
      auto layer_cx = cx[index];
    //   auto layer_hy = hy[index];
    //   auto layer_cx = cx.defined() ? cx[index] : Tensor();
    //   auto layer_cy = cx.defined() ? cy[index] : at::empty({0}, input.options());
      auto reverse = (direction > 0);
    //   auto bias_dtype = get_bias_dtype(layer_input, layer_weights[0]);
      auto outputs = at::mkldnn_rnn_layer(layer_input, layer_weights[0], layer_weights[1],
                                        has_biases ? layer_weights[2] : at::zeros(layer_weights[0].sizes(), layer_weights[0].options()),
          has_biases ? layer_weights[3] : at::zeros(layer_weights[1].sizes(), layer_weights[1].options()), layer_hx,
          layer_cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train);
    //   layer_output[direction] = mkldnn_rnn_layer(layer_hy, layer_cy, layer_input, layer_weights, layer_hx, layer_cx, reverse, fn);
      layer_output[direction] = std::get<0>(outputs);
      layer_hy[index] = std::get<1>(outputs);
      layer_cy[index] = std::get<2>(outputs);
    }
    layer_input = num_directions == 1 ? layer_output[0]
                                      : at::cat(layer_output, /*output_channels*/-1);
    if (dropout_p != 0 && train && layer < num_layers - 1) {
      layer_input = at::dropout(layer_input, dropout_p, /*train=*/true);
    }
  }
  auto output = layer_input;
  auto hy = at::stack(layer_hy, 0);
  auto cy = at::stack(layer_cy, 0);
  if (batch_first && !is_input_packed) {
    output = output.transpose(0, 1);
  }
  return std::make_tuple(output, hy, cy);
}

////////////////////////////////////////////////////////////////////////////////
//// MKLDNN dispatch for the generic RNN ops (at::lstm, at::gru, ...)
////////////////////////////////////////////////////////////////////////////////

namespace {

// Helpers for working with different hidden types.
std::tuple<Tensor, Tensor> unpack_hidden(const Tensor& hidden) {
  return std::make_tuple(hidden, at::Tensor{});
}

std::tuple<Tensor, Tensor> unpack_hidden(const std::tuple<Tensor, Tensor>& hidden) {
  return hidden;
}

template<typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  static_assert(std::is_same<hidden_type, void>::value, "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

template<>
Tensor pack_hidden<Tensor>(const Tensor& hx, const Tensor& cx) {
  AT_ASSERT(cx.numel() == 0);
  return hx;
}

template<>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(const Tensor& hx, const Tensor& cx) {
  return std::make_tuple(hx, cx);
}

template<typename hidden_type>
std::pair<Tensor, hidden_type> mkldnn_impl(
    const Tensor& input, const hidden_type& hidden,
    TensorList params, bool has_biases, ideep::rnn_kind mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
std::cout<<"in mkldnn_impl\n";
  Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  // mkldnn_output = std::tuple<output, hy, cy, workspace>
  auto mkldnn_output = mkldnn_rnn(
      input, params, has_biases ? 4 : 2,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, has_biases, batch_first, dropout_p,
      train, bidirectional, /*batch_sizes*/{});

  return {std::get<0>(mkldnn_output),
          pack_hidden<hidden_type>(std::get<1>(mkldnn_output), std::get<2>(mkldnn_output))};
    // return {Tensor(), pack_hidden<hidden_type>(Tensor(), Tensor())};
}

} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor> lstm_mkldnn_stub(
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  std::cout<<"in lstm_mkldnn_stub\n";

  auto result = mkldnn_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
      ideep::rnn_kind::LSTM, num_layers, dropout_p, train, bidirectional, batch_first);
  auto output = result.first;
  auto hy = std::get<0>(result.second);
  auto cy = std::get<1>(result.second);

  return std::make_tuple(output, hy, cy);
}

}} // namespace at::native

#endif // AT_MKLDNN_EBABLED
