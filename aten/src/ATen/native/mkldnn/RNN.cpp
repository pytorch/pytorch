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

#define ONE_HIDDEN_RNN(NAME, MODE)                                             \
std::tuple<Tensor, Tensor> NAME##_mkldnn_stub(                                 \
    const Tensor& input, const Tensor& hx, TensorList params, bool has_biases, \
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
  AT_ERROR("NAME##_mkldnn_stub: ATen not compiled with MKLDNN support");     \
}

ONE_HIDDEN_RNN(gru, MKLDNN_GRU)
ONE_HIDDEN_RNN(rnn_tanh, MKLDNN_RNN_TANH)
ONE_HIDDEN_RNN(rnn_relu, MKLDNN_RNN_RELU)

std::tuple<Tensor, Tensor, Tensor> lstm_mkldnn_stub(
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  AT_ERROR("lstm_mkldnn_stub: ATen not compiled with MKLDNN support");
}

std::vector<Tensor> mkldnn_rnn_transpose_weight(
    TensorList weight, int64_t weight_stride0,
    int64_t fn_mode, int64_t fn_num_layers,
    bool fn_bidirectional) {
  AT_ERROR("mkldnn_rnn_transpose_weight: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn(
    const Tensor& input_r, TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state) {
  AT_ERROR("mkldnn_rnn: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn_backward(
    const Tensor& input_r, TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx, const Tensor& output_r,
    const Tensor& grad_output_r, const Tensor& grad_hy_r, const Tensor& grad_cy_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state, TensorList reserve,
    std::array<bool, 4> output_mask) {
  AT_ERROR("mkldnn_rnn_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_ENABLED()

#include <ATen/mkldnn/Runtime.h>
#include <ATen/mkldnn/TensorUtils.h>

namespace at { namespace native {

namespace {

typedef enum {
  MKLDNN_RNN_RELU = 0,
  MKLDNN_RNN_TANH = 1,
  MKLDNN_LSTM = 2,
  MKLDNN_GRU = 3
} mkldnnRNNMode_t;

typedef enum {
  MKLDNN_INDEX_INPUT = 0,
  MKLDNN_INDEX_OUTPUT = 1,
  MKLDNN_INDEX_HIDDEN_X = 2,
  MKLDNN_INDEX_HIDDEN_Y = 3,
  MKLDNN_INDEX_WEIGHT_IH = 4,
  MKLDNN_INDEX_WEIGHT_HH = 5,
  MKLDNN_INDEX_BIAS = 6,
  MKLDNN_INDEX_WORKSPACE = 7,
  MKLDNN_INDEX_NUM = 8
} mkldnnRNNIndex_t;

constexpr int64_t ldigo_shuffle_dim = 1;
constexpr int64_t ldgoi_shuffle_dim = 0;
constexpr int64_t ldgo_shuffle_dim = 0;
constexpr int64_t reserve_stride0 = static_cast<int64_t>(MKLDNN_INDEX_NUM);

struct DropoutDescriptorParams {
  bool train;
  double dropout;
  Tensor dropout_state;

  void set(bool train_, double dropout_, Tensor dropout_state_) {
    train = train_;
    dropout = dropout_;
    dropout_state = dropout_state_;
  }
};

struct RNNDescriptorParams {
  mkldnnRNNMode_t mode;
  int64_t hidden_size;
  int64_t num_layers;
  int64_t num_directions;
  int64_t num_gates;
  int64_t num_states;
  int64_t num_biases;
  algorithm rnn_algorithm; // mkldnn rnn algorithm kind
  algorithm rnn_activation; // mkldnn rnn activation kind

  void set_mode(int64_t fn_mode) {
    mode = static_cast<mkldnnRNNMode_t>(fn_mode);
    switch (fn_mode) {
      case MKLDNN_RNN_RELU:
        num_gates = 1;
        num_biases = 1;
        num_states = 1;
        rnn_algorithm = algorithm::vanilla_rnn;
        rnn_activation = algorithm::eltwise_relu;
        break;
      case MKLDNN_RNN_TANH:
        num_gates = 1;
        num_biases = 1;
        num_states = 1;
        rnn_algorithm = algorithm::vanilla_rnn;
        rnn_activation = algorithm::eltwise_tanh;
        break;
      case MKLDNN_LSTM:
        num_gates = 4;
        num_biases = 4;
        num_states = 2;
        rnn_algorithm = algorithm::vanilla_lstm;
        rnn_activation = algorithm::eltwise_tanh;
        break;
      case MKLDNN_GRU:
        num_gates = 3;
        num_biases = 4;
        num_states = 1;
        rnn_algorithm = algorithm::gru_linear_before_reset;
        rnn_activation = algorithm::eltwise_tanh;
        break;
      default:
        AT_ERROR("unrecognized MKLDNN RNN mode %d", fn_mode);  
    }
  }

  void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional) {
  this->set_mode(mode);
  this->hidden_size = hidden_size;
  this->num_layers = num_layers;
  this->num_directions = bidirectional ? 2 : 1;
  }
};

struct TensorDescriptorListParams {
  IntArrayRef batch_sizes;
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t batch_sizes_sum; // == sum(batch_sizes)

  bool is_input_packed() const {
    return batch_sizes.size() != 0;
  }

  void set(IntArrayRef input_sizes, IntArrayRef batch_sizes_, bool batch_first) {
    batch_sizes = batch_sizes_;
    if (is_input_packed()) {
      seq_length = batch_sizes.size();
      mini_batch = batch_sizes[0];
      batch_sizes_sum = input_sizes[0];
      input_size = input_sizes[1];
    } else {
      if (batch_first) {
        seq_length = input_sizes[1];
        mini_batch = input_sizes[0];
      } else {
        seq_length = input_sizes[0];
        mini_batch = input_sizes[1];
      }
      input_size = input_sizes[2];
      batch_sizes_sum = -1;
    }
  }
};

struct RNNParams {
  DropoutDescriptorParams dropout;
  RNNDescriptorParams rnn;
  TensorDescriptorListParams tensors;
};

// Layerwise RNN parameters used to compute key of hashes, POD struct
struct RNNLayerParams {
  mkldnnRNNMode_t mode;
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t hidden_size;
  int64_t train;
  int64_t reverse;
};

void setRNNLayerParams(RNNLayerParams* params, const RNNParams& fn, int64_t input_size, bool reverse_) {
   memset(params, 0, sizeof(RNNLayerParams));
   params->mode = fn.rnn.mode;
   params->seq_length = fn.tensors.seq_length;
   params->mini_batch = fn.tensors.mini_batch;
   params->input_size = input_size;
   params->hidden_size = fn.rnn.hidden_size;
   params->train = static_cast<int64_t>(fn.dropout.train);
   params->reverse = static_cast<int64_t>(reverse_);
}

// RNNDescriptor is the descriptor for a single layer uni-directional RNN call
//
// Since MKLDNN RNN API lack dropout support, multi-layer and bidirectional
// has to be separated into multiple single-layer and uni-directional RNN calls
// which means num_layers = 1 and num_directions = 1
//
// Also MKLDNN can't handle multi-layer RNN calls with input_size != hidden_size 
// with a single rnn primitve
//
struct RNNDescriptor {
  // memory dims
  memory::dims input_tz;
  memory::dims output_tz;
  memory::dims weight_ih_tz;
  memory::dims weight_hh_tz;
  memory::dims bias_tz;
  memory::dims hidden_tz;
  // memory generic desc
  memory::desc input_md;
  memory::desc output_md;
  memory::desc weight_ih_md;
  memory::desc weight_hh_md;
  memory::desc bias_md;
  memory::desc hidden_md;
  // rnn cell desc
  rnn_cell::desc rnn_cell_;
  // rnn direction
  rnn_direction rnn_dir;

  bool train;
  bool reverse;
  RNNLayerParams key;

  RNNDescriptor(const RNNParams& fn, int64_t input_size, bool reverse_)
      : input_tz{fn.tensors.seq_length, fn.tensors.mini_batch, input_size}
      , output_tz{fn.tensors.seq_length, fn.tensors.mini_batch, fn.rnn.hidden_size}
      , weight_ih_tz{1/*layer*/, 1/*direction*/, input_size, fn.rnn.num_gates, fn.rnn.hidden_size}
      , weight_hh_tz{1/*layer*/, 1/*direction*/, fn.rnn.hidden_size, fn.rnn.num_gates, fn.rnn.hidden_size}
      , bias_tz{1/*layer*/, 1/*direction*/, fn.rnn.num_biases, fn.rnn.hidden_size}
      , hidden_tz{1/*layer*/, 1/*direction*/, fn.rnn.num_states, fn.tensors.mini_batch, fn.rnn.hidden_size}
      , input_md(zero_md()), output_md(zero_md()), weight_ih_md(zero_md())
      , weight_hh_md(zero_md()), bias_md(zero_md()), hidden_md(zero_md())
      , rnn_cell_(fn.rnn.rnn_algorithm, fn.rnn.rnn_activation)
      , rnn_dir(rnn_direction::unidirectional_left2right)
      , train(fn.dropout.train), reverse(reverse_) {
    // set up memory descriptor
    if (reverse) { rnn_dir = rnn_direction::unidirectional_right2left; }
    input_md = _format_md(input_tz, memory::format::tnc);
    output_md = _format_md(output_tz, memory::format::tnc);
    hidden_md = _generic_md(hidden_tz);
    weight_ih_md = _generic_md(weight_ih_tz);
    weight_hh_md = _generic_md(weight_hh_tz);
    bias_md = _generic_md(bias_tz);

    setRNNLayerParams(&key, fn, input_size, reverse_);
  }

  rnn_forward::primitive_desc rnn_forward_pd() const {
    using pd_handle_t = std::shared_ptr<rnn_forward::primitive_desc>;
    static thread_local PrimitiveCache<RNNLayerParams, rnn_forward::primitive_desc> cache;
    return cache.get(key, [&](pd_handle_t& _pd) {
        auto _engine = MKLDNNEngine::Instance().get_engine();
        auto rnn_prop = train ? prop_kind::forward_training : prop_kind::forward_inference;
        auto rnn_desc = rnn_forward::desc(rnn_prop, rnn_cell_, rnn_dir,
            input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md);
        _pd.reset(new rnn_forward::primitive_desc(rnn_desc, _engine));
    });
  }

  rnn_backward::primitive_desc rnn_backward_pd() const {
    using pd_handle_t = std::shared_ptr<rnn_backward::primitive_desc>;
    static thread_local PrimitiveCache<RNNLayerParams, rnn_backward::primitive_desc> cache;
    return cache.get(key, [&](pd_handle_t& _pd) {
        auto _engine = MKLDNNEngine::Instance().get_engine();
        auto rnn_prop = prop_kind::backward;
        auto rnn_desc = rnn_backward::desc(rnn_prop, rnn_cell_, rnn_dir,
            input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md,
            input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md);
        _pd.reset(new rnn_backward::primitive_desc(rnn_desc, _engine, rnn_forward_pd()));
    });
  }

  using pdesc_t = memory::primitive_desc;
  pdesc_t input_pd() const { return _primitive_md(input_tz, memory::format::tnc); }
  pdesc_t output_pd() const { return _primitive_md(output_tz, memory::format::tnc); }
  pdesc_t weight_ih_pd() const { return _primitive_md(weight_ih_tz, memory::format::ldigo); }
  pdesc_t weight_hh_pd() const { return _primitive_md(weight_hh_tz, memory::format::ldigo); }
  pdesc_t bias_pd() const { return _primitive_md(bias_tz, memory::format::ldgo); }
  pdesc_t hidden_pd() const { return _primitive_md(hidden_tz, memory::format::ldsnc); }
};

struct RNNForwardPrimitive : MKLDNNPrimitive<rnn_forward> {
  memory src_layer, src_iter, weights_layer, weights_iter, bias, dst_layer, dst_iter, workspace;

  RNNForwardPrimitive(const rnn_forward::primitive_desc& pd, bool train) : MKLDNNPrimitive<rnn_forward>()
      , MREG(src_layer), MREG(src_iter), MREG(weights_layer), MREG(weights_iter), MREG(bias)
      , MREG(dst_layer), MREG(dst_iter), ZREG(workspace) {
    if (train) { workspace = _new_mkldnn_memory(pd.workspace_primitive_desc()); }
    prim_.reset(new rnn_forward(pd, src_layer, src_iter, weights_layer, weights_iter, bias, dst_layer, dst_iter, workspace));
  }

  template <typename T>
  void set(const T& input, const T& hidden_x, const T& weight_ih, const T& weight_hh,
      const T& bias, const T& output, const T& hidden_y, const T& workspace) {
    _set_memory_handle(src_layer, input, src_iter, hidden_x, weights_layer, weight_ih, weights_iter, weight_hh,
        this->bias, bias, dst_layer, output, dst_iter, hidden_y, this->workspace, workspace);
  }
};

struct RNNBackwardPrimitive : MKLDNNPrimitive<rnn_backward> {
  memory src_layer, src_iter, weights_layer, weights_iter, bias, dst_layer, dst_iter, diff_src_layer,
      diff_src_iter, diff_weights_layer, diff_weights_iter, diff_bias, diff_dst_layer, diff_dst_iter, workspace;

  RNNBackwardPrimitive(const rnn_backward::primitive_desc& pd) : MKLDNNPrimitive<rnn_backward>()
      , MREG(src_layer), MREG(src_iter), MREG(weights_layer), MREG(weights_iter), MREG(bias)
      , MREG(dst_layer), MREG(dst_iter), MREG(diff_src_layer), MREG(diff_src_iter)
      , MREG(diff_weights_layer), MREG(diff_weights_iter), MREG(diff_bias)
      , MREG(diff_dst_layer), MREG(diff_dst_iter), MREG(workspace) {
    prim_.reset(new rnn_backward(pd, src_layer, src_iter, weights_layer, weights_iter, bias, dst_layer, dst_iter, diff_src_layer,
        diff_src_iter, diff_weights_layer, diff_weights_iter, diff_bias, diff_dst_layer, diff_dst_iter, workspace));
  }

  template <typename T>
  void set(const T& input, const T& hidden_x, const T& weight_ih, const T& weight_hh,
      const T& bias, const T& output, const T& hidden_y, const T& grad_input, const T& grad_hidden_x,
      const T& grad_weight_ih, const T& grad_weight_hh, const T& grad_bias, const T& grad_output,
      const T& grad_hidden_y, const T& workspace) {
    _set_memory_handle(src_layer, input, src_iter, hidden_x, weights_layer, weight_ih, weights_iter, weight_hh,
        this->bias, bias, dst_layer, output, dst_iter, hidden_y, diff_src_layer, grad_input, diff_src_iter, grad_hidden_x,
        diff_weights_layer, grad_weight_ih, diff_weights_iter, grad_weight_hh, diff_bias, grad_bias,
        diff_dst_layer, grad_output, diff_dst_iter, grad_hidden_y, this->workspace, workspace);
  }
};

std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, tensors.input_size};
  } else {
    return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
  }
}

std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
  return {rnn.num_layers * rnn.num_directions, tensors.mini_batch, rnn.hidden_size};
}

std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, rnn.hidden_size * rnn.num_directions};
  } else {
    return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions};
  }
}

// MKLDNN expects weight in ldigo format while PyTorch stores weight in ldgoi format
//   ldigo: {num_layers, num_directions, input_size, num_gates, hidden_size}
//   ldgoi: {num_layers, num_directions, num_gates, hidden_size, input_size}
//
// MKLDNN GRU gate order is different from PyTorch's which requires gates shuffle
//   (reset, input, new): MKLDNN
//   (input, reset, new): PyTorch
//
Tensor _shuffle_weight(const Tensor& weight, int64_t fn_mode) {
  auto weight_t = weight.t();
  if (static_cast<mkldnnRNNMode_t>(fn_mode) == MKLDNN_GRU) {
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
Tensor _shuffle_bias(const Tensor& bias_ih, const Tensor& bias_hh, int64_t fn_mode) {
  if (static_cast<mkldnnRNNMode_t>(fn_mode) == MKLDNN_GRU) {
    std::vector<Tensor> b1 = bias_ih.chunk(3, ldgo_shuffle_dim);
    std::vector<Tensor> b2 = bias_hh.chunk(3, ldgo_shuffle_dim);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, ldgo_shuffle_dim);
  }
  return bias_ih + bias_hh;
};

std::tuple<Tensor, std::vector<Tensor>> mkldnn_rnn_layer(
    const RNNParams& fn, const Tensor& input, TensorList weights,
    const Tensor& hx, const Tensor& cx, Tensor& hy, Tensor& cy, bool reverse) {

  auto output = at::empty({input.size(0), hx.size(0), hx.size(1)}, input.options());
  std::vector<Tensor> reserve(reserve_stride0);
  for (auto& re : reserve) {
    re = at::empty({0}, hx.options());
  }

  // MKLDNN requires {hx, cx} to be concat for LSTM
  auto hidden_x = cx.defined() ? at::cat({hx, cx}, 0) : hx;
  auto hidden_y = at::empty_like(hidden_x);

  auto train = fn.dropout.train;
  auto mode = fn.rnn.mode;
  auto weight_ih = train ? _shuffle_weight(weights[0], mode) : weights[0];
  auto weight_hh = train ? _shuffle_weight(weights[1], mode) : weights[1];
  auto bias = weights.size() == 2 ? at::zeros({fn.rnn.num_biases * fn.rnn.hidden_size}) :
      train ? _shuffle_bias(weights[2], weights[3], mode) : weights[2];

  RNNDescriptor rnn_desc(fn, input.size(2), reverse);
  auto rnn_pd = rnn_desc.rnn_forward_pd();

  Tensor workspace;
  auto workspace_prv = null_memory();
  if (fn.dropout.train) {
    auto pd = rnn_pd.workspace_primitive_desc();
    int64_t size = pd.get_size() / sizeof(float);
    workspace = at::empty({size}, hx.options());
    workspace_prv = _new_mkldnn_memory(pd, workspace);
  }

  using prim_handle_t = std::shared_ptr<RNNForwardPrimitive>;
  static thread_local PrimitiveCache<RNNLayerParams, RNNForwardPrimitive> cache;
  auto rnn = cache.get(rnn_desc.key,
    [&](prim_handle_t& _prim) { _prim.reset(new RNNForwardPrimitive(rnn_pd, fn.dropout.train)); },
    [&](prim_handle_t& _prim) { _prim->set(input, hidden_x, weight_ih,
        weight_hh, bias, output, hidden_y, workspace); });

  MKLDNN_EXEC(rnn.get_primitive());

  std::vector<Tensor> hidden_arr = hidden_y.chunk(fn.rnn.num_states, 0);
  hy.copy_(hidden_arr[0]);
  if (cx.defined()) {
    cy.copy_(hidden_arr[1]);
  }

  // save for backward
  if (fn.dropout.train) {
    reserve[MKLDNN_INDEX_INPUT] = input;
    reserve[MKLDNN_INDEX_OUTPUT] = output;
    reserve[MKLDNN_INDEX_HIDDEN_X] = hidden_x;
    reserve[MKLDNN_INDEX_HIDDEN_Y] = hidden_y;
    reserve[MKLDNN_INDEX_WEIGHT_IH] = weight_ih;
    reserve[MKLDNN_INDEX_WEIGHT_HH] = weight_hh;
    reserve[MKLDNN_INDEX_BIAS] = bias;
    reserve[MKLDNN_INDEX_WORKSPACE] = workspace;
  }

  return std::make_tuple(output, reserve);
}

Tensor mkldnn_rnn_layer_backward(
    const RNNParams& fn, TensorList weights, const Tensor& grad_output,
    const Tensor& grad_hy, const Tensor& grad_cy, Tensor& grad_hx, Tensor& grad_cx,
    TensorList reserve, TensorList grad_weights,
    bool reverse, std::array<bool, 4> output_mask) {

  auto input = reserve[MKLDNN_INDEX_INPUT];
  auto output = reserve[MKLDNN_INDEX_OUTPUT];
  auto hidden_x = reserve[MKLDNN_INDEX_HIDDEN_X];
  auto hidden_y = reserve[MKLDNN_INDEX_HIDDEN_Y];
  auto weight_ih = reserve[MKLDNN_INDEX_WEIGHT_IH];
  auto weight_hh = reserve[MKLDNN_INDEX_WEIGHT_HH];
  auto bias = reserve[MKLDNN_INDEX_BIAS];
  auto workspace = reserve[MKLDNN_INDEX_WORKSPACE];
  auto grad_input = at::zeros_like(input);

  // MKLDNN requires {grad_hx, grad_cx} to be concat for LSTM
  auto grad_hidden_y = grad_cy.defined() ? at::cat({grad_hy, grad_cy}, 0) : grad_hy;
  auto grad_hidden_x = at::empty_like(grad_hidden_y);
  // MKLDNN layout gradients
  auto grad_weight_ih = at::zeros_like(weight_ih);
  auto grad_weight_hh = at::zeros_like(weight_hh);
  auto grad_bias = at::zeros_like(bias);

  RNNDescriptor rnn_desc(fn, input.size(2), reverse);
  auto rnn_pd = rnn_desc.rnn_backward_pd();

  // MKLDNN expects:
  // 1) weight to be in ldigo format in forward
  // 2) weight to be in ldgoi format in backward
  // 3) grad_weight to be in ldigo format in backward
  using pdesc_t = memory::primitive_desc;
  auto reorder_weight = [&](const Tensor& weight, const pdesc_t& usr_pd, const pdesc_t& prv_pd) {
    auto weight_usr_memory = _new_mkldnn_memory(usr_pd, weight);
    int64_t size = prv_pd.get_size() / sizeof(float);
    auto weight_prv = at::empty({size}, weight.options());
    auto weight_prv_memory = _new_mkldnn_memory(prv_pd, weight_prv);
    _reorder(weight_usr_memory, weight_prv_memory);
    return weight_prv;
  };
  auto weight_ih_ = reorder_weight(weight_ih, rnn_desc.weight_ih_pd(), rnn_pd.weights_layer_primitive_desc());
  auto weight_hh_ = reorder_weight(weight_hh, rnn_desc.weight_hh_pd(), rnn_pd.weights_iter_primitive_desc());

  using prim_handle_t = std::shared_ptr<RNNBackwardPrimitive>;
  static thread_local PrimitiveCache<RNNLayerParams, RNNBackwardPrimitive> cache;
  auto rnn = cache.get(rnn_desc.key,
    [&](prim_handle_t& _prim) { _prim.reset(new RNNBackwardPrimitive(rnn_pd)); },
    [&](prim_handle_t& _prim) { _prim->set(input, hidden_x, weight_ih_, weight_hh_,
        bias, output, hidden_y, grad_input, grad_hidden_x, grad_weight_ih,
        grad_weight_hh, grad_bias, grad_output, grad_hidden_y, workspace); });

  MKLDNN_EXEC(rnn.get_primitive());

  auto grad_w1 = grad_weights[0];
  auto grad_w2 = grad_weights[1];
  auto grad_b1 = grad_weights[2];
  auto grad_b2 = grad_weights[3];

  // MKLDNN GRU has special gate order, see NOTES from forward path
  auto shuffle_weight = [&](const Tensor& weight) {
    if (fn.rnn.mode == MKLDNN_GRU) {
      std::vector<Tensor> gates = weight.chunk(3, ldigo_shuffle_dim);
      return at::cat({gates[1], gates[0], gates[2]}, ldigo_shuffle_dim).t();
    }
    return weight.t();
  };
  grad_w1.copy_(shuffle_weight(grad_weight_ih));
  grad_w2.copy_(shuffle_weight(grad_weight_hh));

  // MKLDNN GRU has special bias gates, see NOTES from forward path
  auto shuffle_bias = [&](const Tensor& bias) {
    if (fn.rnn.mode == MKLDNN_GRU) {
      std::vector<Tensor> gates = bias.chunk(4, ldgo_shuffle_dim);
      auto bias_ih = at::cat({gates[1], gates[0], gates[2]}, ldgo_shuffle_dim);
      auto bias_hh = at::cat({gates[1], gates[0], gates[3]}, ldgo_shuffle_dim);
      return std::make_tuple(bias_ih, bias_hh);
    }
    return std::make_tuple(bias, bias);
  };
  auto biases = shuffle_bias(grad_bias);
  grad_b1.copy_(std::get<0>(biases));
  grad_b2.copy_(std::get<1>(biases));

  std::vector<Tensor> grad_hidden_arr = grad_hidden_x.chunk(fn.rnn.num_states, 0);
  grad_hx.copy_(grad_hidden_arr[0]);
  if (grad_cy.defined()) {
    grad_cx.copy_(grad_hidden_arr[1]);
  }

  return grad_input;
}

} // anonymous namespace

std::vector<Tensor> mkldnn_rnn_transpose_weight(
    TensorList weight, int64_t weight_stride0,
    int64_t fn_mode, int64_t fn_num_layers,
    bool fn_bidirectional) {

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  std::vector<Tensor> transposed_weight_arr;
  transposed_weight_arr.reserve(weights.numel());

  auto has_bias = weight_stride0 == 4;
  auto num_layers = fn_num_layers;
  auto num_directions = fn_bidirectional ? 2 : 1;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];

      transposed_weight_arr.emplace_back(_shuffle_weight(layer_weights[0], fn_mode));
      transposed_weight_arr.emplace_back(_shuffle_weight(layer_weights[1], fn_mode));
      if (has_bias) {
        transposed_weight_arr.emplace_back(_shuffle_bias(layer_weights[2], layer_weights[3], fn_mode));
        transposed_weight_arr.emplace_back(_shuffle_bias(layer_weights[2], layer_weights[3], fn_mode));
      }
    }
  }

  return transposed_weight_arr;
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn(
    const Tensor& input_r, TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state) {

  auto input = input_r;

  RNNParams fn;
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  if (fn.rnn.mode != MKLDNN_LSTM) {
    AT_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  auto is_input_packed = fn.tensors.is_input_packed();
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  AT_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  AT_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  auto x = input.contiguous();
  auto output = at::empty(output_size, input.options());
  auto hy = at::empty(hidden_size, hx.options());
  auto cy = (cx.defined()) ? at::empty(hidden_size, hx.options()) : at::empty({0}, hx.options());
  auto y = output;

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_layers = fn.rnn.num_layers;
  auto num_directions = fn.rnn.num_directions;

  std::vector<Tensor> reserve;
  reserve.reserve(num_layers * num_directions * reserve_stride0);

  auto layer_x = x;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    std::vector<Tensor> layer_y(num_directions);
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      auto layer_hx = hx[index];
      auto layer_hy = hy[index];
      auto layer_cx = cx.defined() ? cx[index] : Tensor();
      auto layer_cy = cx.defined() ? cy[index] : at::empty({0}, input.options());
      auto reverse = (direction > 0);
      auto result = mkldnn_rnn_layer(fn, layer_x, layer_weights, layer_hx, layer_cx, layer_hy, layer_cy, reverse);
      layer_y[direction] = std::get<0>(result);
      auto layer_reserve = std::get<1>(result);
      for (const auto& re : layer_reserve) {
        reserve.emplace_back(re);
      }
    }
    layer_x = at::cat(layer_y, input.dim() - 1);
  }

  y = layer_x;
  if (batch_first && !is_input_packed) {
    y = y.transpose_(0, 1);
  }

  return std::make_tuple(y, hy, cy, reserve);
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn_backward(
    const Tensor& input_r, TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx, const Tensor& output_r,
    const Tensor& grad_output_r, const Tensor& grad_hy_r, const Tensor& grad_cy_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state, TensorList reserve,
    std::array<bool, 4> output_mask) {

  AT_CHECK(!fn_dropout_state.defined() && fn_dropout == 0, "rnn: mkldnn doesn't support dropout");

  auto input = input_r;
  auto output = output_r;
  auto grad_output = grad_output_r.defined() ? grad_output_r : at::zeros_like(output);
  auto grad_hy = grad_hy_r.defined() ? grad_hy_r : at::zeros_like(hx);
  auto grad_cy = cx.defined() ? (grad_cy_r.defined() ? grad_cy_r : at::zeros_like(cx)) : grad_cy_r;

  RNNParams fn;
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  if (fn.rnn.mode != MKLDNN_LSTM) {
    AT_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  auto is_input_packed = fn.tensors.is_input_packed();
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    grad_output = grad_output.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  AT_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  AT_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");
  AT_ASSERTM(cx.defined() || !output_mask[2], "illegally required grad of cx for non-LSTM RNN");
  AT_CHECK(fn_train, "mkldnn RNN backward can only be called in training mode");

  auto x = input.contiguous();
  auto grad_y = grad_output.contiguous();
  grad_hy = grad_hy.contiguous().view(hidden_size);
  grad_cy = grad_cy.defined() ? grad_cy.contiguous().view(hidden_size) : Tensor();
  auto grad_x = at::empty(input_size, input.options());
  auto grad_hx = at::empty(hidden_size, hx.options());
  auto grad_cx = cx.defined() ? at::empty(hidden_size, cx.options()) : Tensor();

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};
  MatrixRef<Tensor> reserves{reserve, static_cast<size_t>(reserve_stride0)};

  std::vector<Tensor> grad_weight_arr;
  grad_weight_arr.reserve(weights.numel());
  for (const auto& w : weight) {
    grad_weight_arr.emplace_back(at::empty(w.sizes(), w.options()));
  }
  MatrixRef<Tensor> grad_weights{grad_weight_arr, static_cast<size_t>(weight_stride0)};

  auto num_layers = fn.rnn.num_layers;
  auto num_directions = fn.rnn.num_directions;

  auto layer_grad_output = grad_y;
  for (int64_t layer = num_layers-1; layer >= 0; layer--) {
    auto layer_grad_x = at::empty({0}, input.options());
    std::vector<Tensor> layer_grad_ys = layer_grad_output.chunk(num_directions, layer_grad_output.dim() - 1);
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;

      auto layer_weights = weights[index];
      auto layer_grad_y = layer_grad_ys[direction].contiguous();
      auto layer_grad_hy = grad_hy[index];
      auto layer_grad_cy = grad_cy.defined() ? grad_cy[index] : Tensor();
      auto layer_grad_hx = grad_hx[index];
      auto layer_grad_cx = cx.defined() ? grad_cx[index] : at::empty({0}, hx.options());
      auto layer_reserve = reserves[index];
      auto layer_grad_weights = grad_weights[index];

      auto reverse = (direction > 0);
      auto layer_grad_xd = mkldnn_rnn_layer_backward(fn, layer_weights, layer_grad_y, layer_grad_hy,
          layer_grad_cy, layer_grad_hx, layer_grad_cx, layer_reserve, layer_grad_weights, reverse, output_mask);

      if (reverse) {
        layer_grad_x.add_(layer_grad_xd);
      } else {
        layer_grad_x.resize_as_(layer_grad_xd).copy_(layer_grad_xd);
      }
    }
    layer_grad_output = layer_grad_x;
  }

  grad_x = layer_grad_output;
  if (batch_first && !is_input_packed) {
    grad_x = grad_x.transpose_(0, 1);
  }

  return std::make_tuple(grad_x, grad_hx, grad_cx, grad_weight_arr);
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
    TensorList params, bool has_biases, mkldnnRNNMode_t mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {

  Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  // mkldnn_output = std::tuple<output, hy, cy, reserve>
  auto mkldnn_output = at::mkldnn_rnn(
      input, params, has_biases ? 4 : 2,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, batch_first, dropout_p,
      train, bidirectional, /*batch_sizes*/{}, Tensor());

  return {std::get<0>(mkldnn_output),
          pack_hidden<hidden_type>(std::get<1>(mkldnn_output), std::get<2>(mkldnn_output))};
}

} // anonymous namespace

#define ONE_HIDDEN_RNN(NAME, MODE)                                             \
std::tuple<Tensor, Tensor> NAME##_mkldnn_stub(                                 \
    const Tensor& input, const Tensor& hx, TensorList params, bool has_biases, \
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
                                                                               \
  auto result = mkldnn_impl(input, hx, params, has_biases,                     \
      MODE, num_layers, dropout_p, train, bidirectional, batch_first);         \
  auto output = result.first;                                                  \
  auto hy = result.second;                                                     \
                                                                               \
  return std::make_tuple(output, hy);                                          \
}

ONE_HIDDEN_RNN(gru, MKLDNN_GRU)
ONE_HIDDEN_RNN(rnn_tanh, MKLDNN_RNN_TANH)
ONE_HIDDEN_RNN(rnn_relu, MKLDNN_RNN_RELU)

std::tuple<Tensor, Tensor, Tensor> lstm_mkldnn_stub(
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {

  auto result = mkldnn_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
      MKLDNN_LSTM, num_layers, dropout_p, train, bidirectional, batch_first);
  auto output = result.first;
  auto hy = std::get<0>(result.second);
  auto cy = std::get<1>(result.second);

  return std::make_tuple(output, hy, cy);
}

}} // namespace at::native

#endif // AT_MKLDNN_ENABLED()
