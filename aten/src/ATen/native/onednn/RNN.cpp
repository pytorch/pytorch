#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>

#include <ATen/TensorUtils.h>
#include <ATen/Dispatch.h>
#include <c10/core/GradMode.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/mkldnn_convolution_native.h>
#include <ATen/ops/onednn_rnn_layer_backward_native.h>
#include <ATen/ops/onednn_rnn_layer_native.h>
#endif

#if !AT_ONEDNN_ENABLED()

namespace at::native {


std::tuple<Tensor, Tensor, Tensor, Tensor> onednn_rnn_layer(
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
      TORCH_CHECK(false, "onednn_rnn_layer: ATen not compiled with ONEDNN support");
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
    TORCH_CHECK(false, "mkldnn_rnn_layer: ATen not compiled with ONEDNN support");
  }

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> onednn_rnn_layer_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace) {
      TORCH_CHECK(false, "onednn_rnn_layer_backward: ATen not compiled with ONEDNN support");
    }

    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
    mkldnn_rnn_layer_backward(
        const Tensor& input,
        const Tensor& weight0,
        const Tensor& weight1,
        const Tensor& weight2,
        const Tensor& weight3,
        const Tensor& hx_,
        const Tensor& cx_tmp,
        const Tensor& output,
        const Tensor& hy_,
        const Tensor& cy_,
        const std::optional<Tensor>& grad_output_r_opt,
        const std::optional<Tensor>& grad_hy_r_opt,
        const std::optional<Tensor>& grad_cy_r_opt,
        bool reverse,
        int64_t mode,
        int64_t hidden_size,
        int64_t num_layers,
        bool has_biases,
        bool train,
        bool bidirectional,
        at::IntArrayRef batch_sizes,
        bool batch_first,
        const at::Tensor& workspace) {
      TORCH_CHECK(false, "mkldnn_rnn_layer_backward: ATen not compiled with ONEDNN support");
    }

REGISTER_NO_CPU_DISPATCH(lstm_mkldnn_stub)

} // namespace at::native

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>
#include <ATen/native/onednn/Utils.h>

namespace at::native {

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

  // onednn memory descriptors
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

template<bool is_single_direction>
std::vector<int64_t> _output_size(const RNNParams& rnn) {
  auto output_channels = is_single_direction ? rnn.hidden_size
                                             : rnn.hidden_size * rnn.num_directions;
  return {rnn.seq_length, rnn.mini_batch, output_channels};
}

// ONEDNN GRU gate order is different from PyTorch's which requires gates shuffle
// (let rt,zt,nt be reset, update, new gates respectively)
//
//   ONEDNN GRU weight_ih/weight_hh gates order: (zt, rt, nt)
//   PyTorch GRU weight_ih/weight_hh gates order: (rt, zt, nt)
//
// ONEDNN GRU bias has 4 gates instead of 3
//  (PyTorch GRU bias)     (ONEDNN GRU bias)
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
static Tensor _shuffle_weight(const Tensor& weight, int64_t fn_mode) {
  auto weight_t = weight.contiguous();
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> gates = weight_t.chunk(3, /*gates*/0);
    return at::cat({gates[1], gates[0], gates[2]}, /*gates*/0);
  }
  return weight_t;
}

static Tensor _shuffle_bias(const Tensor& bias_ih, const Tensor& bias_hh, int64_t fn_mode) {
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> b1 = bias_ih.chunk(3, /*output_channels*/0);
    std::vector<Tensor> b2 = bias_hh.chunk(3, /*output_channels*/0);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, /*output_channels*/0);
  }
  return bias_ih + bias_hh;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> onednn_rnn_layer(const Tensor& input,
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
  RNNParams rnn(
      input,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      bidirectional,
      batch_first,
      train);

  auto output_size = _output_size</*is_single_direction*/ true>(rnn);
  auto output = at::empty(output_size, input.options());

  auto hy_ = at::empty(hx_.sizes(), hx_.options());
  auto cy_ = at::empty(cx_.sizes(), cx_.options());

  auto weight_ih = _shuffle_weight(w0, rnn.mode);
  auto weight_hh = _shuffle_weight(w1, rnn.mode);

  // Packed weight will be onednn layout while bias won't be packed
  auto bias = has_biases
      ? _shuffle_bias(w2, w3, rnn.mode)
      : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options().layout(at::Layout::Strided));

  // per layer input size
  int64_t input_size = input.size(2);
  ideep::tensor w1_, w2_;
  auto x = itensor_view_from_dense(
      input,
      rnn.src_layer_desc(input_size, get_onednn_dtype(input)));
  auto hx = itensor_view_from_dense(
      hx_, rnn.src_iter_desc(get_onednn_dtype(hx_)));
  auto cx = itensor_view_from_dense(
      cx_, rnn.src_iter_c_desc(get_onednn_dtype(cx_)));
  auto b = itensor_view_from_dense(
      bias, rnn.bias_desc(get_onednn_dtype(bias)));
  auto y = itensor_view_from_dense(
      output, rnn.dst_layer_desc(get_onednn_dtype(output)));
  auto hy = itensor_view_from_dense(
      hy_, rnn.dst_iter_desc(get_onednn_dtype(hy_)));
  auto cy = itensor_view_from_dense(
      cy_, rnn.dst_iter_c_desc(get_onednn_dtype(cy_)));
  w1_ = weight_ih.is_onednn() ? itensor_from_tensor(weight_ih) : itensor_view_from_dense(weight_ih, rnn.weights_layer_desc(input_size, get_onednn_dtype(weight_ih)));
  w2_ = weight_hh.is_onednn() ? itensor_from_tensor(weight_hh) : itensor_view_from_dense(weight_hh, rnn.weights_iter_desc(get_onednn_dtype(weight_hh)));
  if (at::GradMode::is_enabled()) {
    Tensor workspace = Tensor();
    auto pd = ideep::lstm_forward_training::prepare(
        x, hx, cx, w1_, w2_, b, y, hy, cy, reverse);
    workspace = at::empty(pd.workspace_desc().get_size() / sizeof(uint8_t), input.options().dtype(at::kByte));
    ideep::tensor onednn_workspace;
    onednn_workspace.init(
        pd.workspace_desc(), workspace.template data_ptr<uint8_t>());
    ideep::lstm_forward_training::compute(
        pd, x, hx, cx, w1_, w2_, b, onednn_workspace, y, hy, cy, reverse, ideep::prop_kind::forward_training);
    return std::make_tuple(output, hy_, cy_, workspace);
  } else {
    ideep::lstm_forward_inference::compute(
        x, hx, cx, w1_, w2_, b, y, hy, cy, reverse, ideep::prop_kind::forward_inference);
    return std::make_tuple(output, hy_, cy_, Tensor());
  }
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
  return at::native::onednn_rnn_layer(
      input,
      w0,
      w1,
      w2,
      w3,
      hx_,
      cx_,
      reverse,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      bidirectional,
      batch_first,
      train);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> onednn_rnn_layer_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace) {
  const Tensor& grad_output_r = grad_output_r_opt.value_or(Tensor());
  const Tensor& grad_hy_r = grad_hy_r_opt.value_or(Tensor());
  const Tensor& grad_cy_r = grad_cy_r_opt.value_or(Tensor());
  if (!grad_output_r.defined() && !grad_hy_r.defined() && !grad_cy_r.defined()) {
      return std::make_tuple(Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
  }
  auto grad_output = grad_output_r.defined() ? grad_output_r.contiguous() : at::zeros_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_hy = grad_hy_r.defined() ? grad_hy_r.contiguous() : at::zeros_like(hx_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_cy = cx_tmp.defined() ? (grad_cy_r.defined() ? grad_cy_r.contiguous() : at::zeros_like(cx_tmp, LEGACY_CONTIGUOUS_MEMORY_FORMAT)) : grad_cy_r.contiguous();
  RNNParams rnn(
      input,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      bidirectional,
      batch_first,
      train);
  auto output_size = _output_size</*is_single_direction*/ true>(rnn);

  auto weight_ih = _shuffle_weight(weight0, rnn.mode);
  auto weight_hh = _shuffle_weight(weight1, rnn.mode);
  auto bias = has_biases
      ? _shuffle_bias(weight2, weight3, rnn.mode)
      : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  auto cx_  =  hx_.storage().unsafeGetStorageImpl() == cx_tmp.storage().unsafeGetStorageImpl() ? at::clone(cx_tmp) : cx_tmp;

  // per layer input size
  int64_t input_size = input.size(2);
  auto x = itensor_view_from_dense(
      input,
      rnn.src_layer_desc(input_size, get_onednn_dtype(input.scalar_type())));
  auto hx = itensor_view_from_dense(
      hx_, rnn.src_iter_desc(get_onednn_dtype(hx_.scalar_type())));
  auto cx = itensor_view_from_dense(
      cx_, rnn.src_iter_c_desc(get_onednn_dtype(cx_.scalar_type())));
  auto w1 = itensor_view_from_dense(
      weight_ih,
      rnn.weights_layer_desc(
          input_size, get_onednn_dtype(weight_ih.scalar_type())));
  auto w2 = itensor_view_from_dense(
      weight_hh,
      rnn.weights_iter_desc(get_onednn_dtype(weight_hh.scalar_type())));
  auto b = itensor_view_from_dense(
      bias, rnn.bias_desc(get_onednn_dtype(bias.scalar_type())));
  auto y = itensor_view_from_dense(
      output, rnn.dst_layer_desc(get_onednn_dtype(output.scalar_type())));
  auto hy = itensor_view_from_dense(
      hy_, rnn.dst_iter_desc(get_onednn_dtype(hy_.scalar_type())));
  auto cy = itensor_view_from_dense(
      cy_, rnn.dst_iter_c_desc(get_onednn_dtype(cy_.scalar_type())));

  // Create diff_* ATen tensor and corresponding ideep tensor as fp32
  auto diff_x_ =
      at::empty(input.sizes(), input.options().dtype(at::ScalarType::Float));
  auto diff_hx_ =
      at::empty(hx_.sizes(), hx_.options().dtype(at::ScalarType::Float));
  auto diff_cx_ =
      at::empty(cx_.sizes(), cx_.options().dtype(at::ScalarType::Float));
  auto diff_w1_ = at::empty(
      weight_ih.sizes(), weight_ih.options().dtype(at::ScalarType::Float));
  auto diff_w2_ = at::empty(
      weight_hh.sizes(), weight_hh.options().dtype(at::ScalarType::Float));
  auto diff_b_ =
      at::empty(bias.sizes(), bias.options().dtype(at::ScalarType::Float));

  auto diff_x = itensor_view_from_dense(
      diff_x_, rnn.src_layer_desc(input_size, ideep::tensor::data_type::f32));
  auto diff_hx = itensor_view_from_dense(
      diff_hx_, rnn.src_iter_desc(ideep::tensor::data_type::f32));
  auto diff_cx = itensor_view_from_dense(
      diff_cx_, rnn.src_iter_c_desc(ideep::tensor::data_type::f32));
  auto diff_w1 = itensor_view_from_dense(
      diff_w1_,
      rnn.weights_layer_desc(input_size, ideep::tensor::data_type::f32));
  auto diff_w2 = itensor_view_from_dense(
      diff_w2_, rnn.weights_iter_desc(ideep::tensor::data_type::f32));
  auto diff_b = itensor_view_from_dense(
      diff_b_, rnn.bias_desc(ideep::tensor::data_type::f32));

  // Convert grad_y, grad_hy, grad_cy to fp32 in non-fp32 backward
  ideep::tensor diff_y, diff_hy, diff_cy;
  at::Tensor grad_y_, grad_hy_, grad_cy_;
  if (input.scalar_type() != at::ScalarType::Float) {
    grad_y_ = at::empty(
        grad_output.sizes(),
        grad_output.options().dtype(at::ScalarType::Float));
    grad_y_.copy_(grad_output);
    grad_hy_ = at::empty(
        grad_hy.sizes(), grad_hy.options().dtype(at::ScalarType::Float));
    grad_hy_.copy_(grad_hy);
    grad_cy_ = at::empty(
        grad_cy.sizes(), grad_cy.options().dtype(at::ScalarType::Float));
    grad_cy_.copy_(grad_cy);

    diff_y = itensor_view_from_dense(
        grad_y_, rnn.dst_layer_desc(get_onednn_dtype(grad_y_.scalar_type())));
    diff_hy = itensor_view_from_dense(
        grad_hy_, rnn.dst_iter_desc(get_onednn_dtype(grad_hy_.scalar_type())));
    diff_cy = itensor_view_from_dense(
        grad_cy_, rnn.dst_iter_desc(get_onednn_dtype(grad_cy_.scalar_type())));
  } else {
    diff_y = itensor_view_from_dense(
        grad_output, rnn.dst_layer_desc(ideep::tensor::data_type::f32));
    diff_hy = itensor_view_from_dense(
        grad_hy, rnn.dst_iter_desc(ideep::tensor::data_type::f32));
    diff_cy = itensor_view_from_dense(
        grad_cy, rnn.dst_iter_desc(ideep::tensor::data_type::f32));
  }

  auto forward_hint = ideep::lstm_forward_training::prepare(x, hx, cx, w1, w2, b, y, hy, cy, reverse);
  ideep::tensor onednn_workspace;
  onednn_workspace.init(
      forward_hint.workspace_desc(), workspace.template data_ptr<uint8_t>());
  ideep::lstm_backward::compute(forward_hint, x, hx, cx, w1, w2, b, y, hy, cy, diff_y, diff_hy, diff_cy, onednn_workspace, diff_x, diff_hx, diff_cx, diff_w1, diff_w2, diff_b, reverse);
  auto diff_b2_ = at::clone(diff_b_);
  return std::make_tuple(diff_x_, diff_w1_, diff_w2_, diff_b_, diff_b2_, diff_hx_, diff_cx_);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
mkldnn_rnn_layer_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace) {
  return at::native::onednn_rnn_layer_backward(
      input,
      weight0,
      weight1,
      weight2,
      weight3,
      hx_,
      cx_tmp,
      output,
      hy_,
      cy_,
      grad_output_r_opt,
      grad_hy_r_opt,
      grad_cy_r_opt,
      reverse,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      train,
      bidirectional,
      batch_sizes,
      batch_first,
      workspace);
}

// ONEDNN RNN integration notes:
// I. Memory Formats
//   a. onednn will use plain formats for input, hx/cx, output, hy/cy
//      and possibly use blocked formats for weights depending shape info.
//   b. All onednn memorys are created (in plain format) as views on ATen tensor,
//      the weight reorder(if any) is handed automatically inside ideep (onednn bridge)
//
// II. ONEDNN Primitive Mapping
//   a. onednn rnn primitive doesn't support training with dropout or padded input sequence.
//   b. here break a single RNN module into { num_layers * num_directions } onednn rnn primitives
//      for future need to cover these feature gaps.
//
//TODO: a. training with dropout
//   b. padded sequence input support
//

static std::tuple<Tensor, Tensor, Tensor> onednn_rnn(
    const Tensor& input_, TensorList weight, int64_t weight_stride0,
    const Tensor& hx_, const Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool has_biases, bool batch_first, double dropout_p,
    bool train, bool bidirectional, IntArrayRef batch_sizes) {
  TORCH_CHECK(batch_sizes.size() == 0, "onednn_rnn doesn't support packed input");
  if (static_cast<ideep::rnn_kind>(mode) != ideep::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(), "onednn_rnn: illegal defined cx for non-LSTM RNN");
  }

  auto input = input_;
  if (batch_first) {
    input = input.transpose(0, 1);
  }
  input = input.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.contiguous();

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_directions = bidirectional ? 2 : 1;
  auto layer_input = input;
  std::vector<at::Tensor> layer_output(num_directions);
  std::vector<at::Tensor> layer_hy(num_layers * num_directions);
  std::vector<at::Tensor> layer_cy(num_layers * num_directions);
  for (const auto layer: c10::irange(num_layers)) {
    for (const auto direction: c10::irange(num_directions)) {
      const auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      TORCH_CHECK(layer_weights.size() == 2 || layer_weights.size() == 4);
      auto layer_hx = hx[index];
      auto layer_cx = cx[index];
      auto reverse = (direction > 0);
      // bias won't be packed
      std::tie(
          layer_output[direction],
          layer_hy[index],
          layer_cy[index],
          std::ignore) =
          at::onednn_rnn_layer(
              layer_input,
              layer_weights[0],
              layer_weights[1],
              has_biases
                  ? layer_weights[2]
                  : at::zeros(
                        layer_weights[0].sizes(),
                        layer_weights[0].options().layout(at::Layout::Strided)),
              has_biases
                  ? layer_weights[3]
                  : at::zeros(
                        layer_weights[1].sizes(),
                        layer_weights[1].options().layout(at::Layout::Strided)),
              layer_hx,
              layer_cx,
              reverse,
              batch_sizes,
              mode,
              hidden_size,
              num_layers,
              has_biases,
              bidirectional,
              batch_first,
              train);
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
  if (batch_first) {
    output = output.transpose(0, 1);
  }
  return std::make_tuple(output, hy, cy);
}

////////////////////////////////////////////////////////////////////////////////
//// ONEDNN dispatch for the generic RNN ops (at::lstm, at::gru, ...)
////////////////////////////////////////////////////////////////////////////////

namespace {

// Helpers for working with different hidden types.
std::tuple<Tensor, Tensor> unpack_hidden(const std::tuple<Tensor, Tensor>& hidden) {
  return hidden;
}

template<typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  static_assert(false && sizeof(hidden_type), "pack_hidden not implemented for this type");
}

template<>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(const Tensor& hx, const Tensor& cx) {
  return std::make_tuple(hx, cx);
}

template<typename hidden_type>
std::pair<Tensor, hidden_type> onednn_impl(
    const Tensor& input, const hidden_type& hidden,
    TensorList params, bool has_biases, ideep::rnn_kind mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  auto [hx, cx] = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  auto onednn_output = onednn_rnn(
      input, params, has_biases ? 4 : 2,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, has_biases, batch_first, dropout_p,
      train, bidirectional, /*batch_sizes*/{});

  return {std::get<0>(onednn_output),
          pack_hidden<hidden_type>(std::get<1>(onednn_output), std::get<2>(onednn_output))};
}

void lstm_mkldnn(Tensor& output, Tensor& hy, Tensor& cy,
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  auto result = onednn_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
      ideep::rnn_kind::LSTM, num_layers, dropout_p, train, bidirectional, batch_first);
  output = std::move(result.first);
  std::tie(hy, cy) = std::move(result.second);
}
} // anonymous namespace

REGISTER_ALL_CPU_DISPATCH(lstm_onednn_stub, &lstm_mkldnn)

} // namespace at::native

#endif // AT_ONEDNN_ENABLED
