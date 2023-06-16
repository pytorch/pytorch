#include <ATen/TensorOperators.h>
#include <ATen/native/vulkan/ops/Lstm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/addmm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/tanh.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
//
// input_vk: input tensor of shape (L, N, H_in) when batch_first=False or
// (N, L, H_in) when batch_first=True containing the features of the input
// sequence
//
// hx_vk: tensor of shape (D * num_layers, N, H_out) containing the initial
// hidden state for each element in the input sequence.
//
// cx_vk: tensor of shape (D * num_layers, N, H_cell) containing the initial
// cell state for each element in the input sequence.
//
// output: tensor of shape (L, N, D * H_out) when batch_first=False or
// (N, L, D * H_out) when batch_first=True, containing the output features
// (h_t) from the last layer of the LSTM, for each t
//
// h_n: tensor of shape (D * num_layers, N, H_out) containing the final hidden
// state for each element in the sequence.
//
// c_n: tensor of shape (D * num_layers, N, H_cell) containing the final cell
// state for each element in the sequence.
//
//  where
//    L = sequence length
//    N = batch size
//    D = 2 if bidirectional=True otherwise 1
//    H_in = input_size (# of expected features in the input x)
//    H_cell = hidden_size (# of features in the hidden state h)
//    H_out = hidden_size
//
std::tuple<Tensor, Tensor, Tensor> lstm_input(
    const Tensor& input_vk, // input sequence (vulkan)
    TensorList
        hx, // initial hidden state (vulkan) & initial cell state (vulkan)
    TensorList params_cpu, // weights/biases (cpu)
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  TORCH_CHECK(
      hx[0].size(2) == hx[1].size(2),
      "Vulkan LSTM with projections is not supported");
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()),
      "Vulkan LSTM expects 'params_cpu' size to be 4 * 'num_layers'.");
  TORCH_INTERNAL_ASSERT(
      input_vk.sizes().size() == 3, "Vulkan LSTM expects input dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      hx[0].sizes().size() == 3,
      "Vulkan LSTM expects hidden state dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      hx[1].sizes().size() == 3,
      "Vulkan LSTM expects cell state dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      has_biases, "Vulkan LSTM expects 'has_biases' to be true.");
  TORCH_INTERNAL_ASSERT(!train, "Vulkan LSTM expects 'train' to be false.");
  TORCH_INTERNAL_ASSERT(
      !bidirectional, "Vulkan LSTM expects 'bidirectional' to be false.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan LSTM expects 'dropout' to be 0.0.");

  const auto batch_size = input_vk.size(0);
  const auto seq_length = input_vk.size(1);

  TORCH_INTERNAL_ASSERT(
      (batch_size == 1 && seq_length == 1) || batch_first,
      "Vulkan gru expects batch-first input");

  const Tensor& hx_vk = hx[0];
  const Tensor& cx_vk = hx[1];

  const auto hidden_size = hx_vk.size(2);
  std::vector<at::Tensor> h_n_list; // hidden state output
  std::vector<at::Tensor> c_n_list; // cell state output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x = input_vk.reshape({batch_size * seq_length, input_vk.size(2)});

  h_n_list.reserve(num_layers);
  c_n_list.reserve(num_layers);

  for (int64_t l = 0; l < num_layers; ++l) {
    // extract each hidden state and squeeze into 2D dim
    auto h = at::slice(hx_vk, 0, l, l + 1, 1);
    h = h.reshape({h.size(0) * h.size(1), h.size(2)});

    auto c = at::slice(cx_vk, 0, l, l + 1, 1);
    c = c.reshape({c.size(0) * c.size(1), c.size(2)});

    const auto& w_ih = params_cpu[l * 4];
    const auto& w_hh = params_cpu[l * 4 + 1];
    const auto& b_ih = params_cpu[l * 4 + 2];
    const auto& b_hh = params_cpu[l * 4 + 3];

    const auto& w_i_ifgo = w_ih.split(hidden_size);
    const auto& w_h_ifgo = w_hh.split(hidden_size);
    const auto& b_i_ifgo = b_ih.split(hidden_size);
    const auto& b_h_ifgo = b_hh.split(hidden_size);

    const auto& w_ii = w_i_ifgo[0];
    const auto& w_if = w_i_ifgo[1];
    const auto& w_ig = w_i_ifgo[2];
    const auto& w_io = w_i_ifgo[3];
    const auto& w_hi = w_h_ifgo[0];
    const auto& w_hf = w_h_ifgo[1];
    const auto& w_hg = w_h_ifgo[2];
    const auto& w_ho = w_h_ifgo[3];
    const auto& b_ii = b_i_ifgo[0];
    const auto& b_if = b_i_ifgo[1];
    const auto& b_ig = b_i_ifgo[2];
    const auto& b_io = b_i_ifgo[3];
    const auto& b_hi = b_h_ifgo[0];
    const auto& b_hf = b_h_ifgo[1];
    const auto& b_hg = b_h_ifgo[2];
    const auto& b_ho = b_h_ifgo[3];

    const auto& i = at::sigmoid(
        at::addmm(b_ii, x, w_ii.t()) + at::addmm(b_hi, h, w_hi.t()));
    const auto& f = at::sigmoid(
        at::addmm(b_if, x, w_if.t()) + at::addmm(b_hf, h, w_hf.t()));
    const auto& g =
        at::tanh(at::addmm(b_ig, x, w_ig.t()) + at::addmm(b_hg, h, w_hg.t()));
    const auto& o = at::sigmoid(
        at::addmm(b_io, x, w_io.t()) + at::addmm(b_ho, h, w_ho.t()));
    c = f * c + i * g;
    h = o * at::tanh(c);
    x = h; // next input
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 2D to 4D for cat op
    c_n_list.emplace_back(
        c.reshape({1, 1, c.size(0), c.size(1)})); // 2D to 4D for cat op
  }

  auto h_n = at::cat(h_n_list, 1);
  auto c_n = at::cat(c_n_list, 1);
  x = x.reshape({batch_size, seq_length, x.size(1)});
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  c_n = c_n.reshape({c_n.size(0) * c_n.size(1), c_n.size(2), c_n.size(3)});
  return std::tuple<Tensor, Tensor, Tensor>(x, h_n, c_n);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::lstm.input"), TORCH_FN(lstm_input));
}

#endif /* USE_VULKAN_API */

} // namespace

std::
    vector<c10::intrusive_ptr<LinearPackedContext>> static pack_lstm_linear_op_contexts(
        const std::vector<Tensor>& params_cpu,
        int64_t num_layers) {
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()) == 4 * num_layers,
      "Vulkan LSTM expects 'params_cpu' size to be 4 * 'num_layers'."
      " But 'params_cpu' has size: ",
      params_cpu.size(),
      " and 'num_layers' is: ",
      num_layers);
  std::vector<c10::intrusive_ptr<LinearPackedContext>> linear_op_contexts;
  linear_op_contexts.reserve(num_layers * 8);

  for (int64_t l = 0; l < num_layers; ++l) {
    const auto& w_ih = params_cpu[l * 4];
    const auto& w_hh = params_cpu[l * 4 + 1];
    const auto& b_ih = params_cpu[l * 4 + 2];
    const auto& b_hh = params_cpu[l * 4 + 3];
    const auto& hidden_size = w_ih.size(0) / 4;

    const auto& w_i_ifgo = w_ih.split(hidden_size);
    const auto& w_h_ifgo = w_hh.split(hidden_size);
    const auto& b_i_ifgo = b_ih.split(hidden_size);
    const auto& b_h_ifgo = b_hh.split(hidden_size);

    const auto& w_ii = w_i_ifgo[0];
    const auto& w_if = w_i_ifgo[1];
    const auto& w_ig = w_i_ifgo[2];
    const auto& w_io = w_i_ifgo[3];
    const auto& w_hi = w_h_ifgo[0];
    const auto& w_hf = w_h_ifgo[1];
    const auto& w_hg = w_h_ifgo[2];
    const auto& w_ho = w_h_ifgo[3];
    const auto& b_ii = b_i_ifgo[0];
    const auto& b_if = b_i_ifgo[1];
    const auto& b_ig = b_i_ifgo[2];
    const auto& b_io = b_i_ifgo[3];
    const auto& b_hi = b_h_ifgo[0];
    const auto& b_hf = b_h_ifgo[1];
    const auto& b_hg = b_h_ifgo[2];
    const auto& b_ho = b_h_ifgo[3];

    linear_op_contexts.emplace_back(create_linear_context(w_ii.t(), b_ii));
    linear_op_contexts.emplace_back(create_linear_context(w_hi.t(), b_hi));
    linear_op_contexts.emplace_back(create_linear_context(w_if.t(), b_if));
    linear_op_contexts.emplace_back(create_linear_context(w_hf.t(), b_hf));
    linear_op_contexts.emplace_back(create_linear_context(w_ig.t(), b_ig));
    linear_op_contexts.emplace_back(create_linear_context(w_hg.t(), b_hg));
    linear_op_contexts.emplace_back(create_linear_context(w_io.t(), b_io));
    linear_op_contexts.emplace_back(create_linear_context(w_ho.t(), b_ho));
  }
  return linear_op_contexts;
}

LstmPackedContext::LstmPackedContext(
    const std::vector<Tensor>& params_cpu, // weights/biases (cpu)
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  TORCH_INTERNAL_ASSERT(
      has_biases, "Vulkan LSTM expects 'has_biases' to be true.");
  TORCH_INTERNAL_ASSERT(!train, "Vulkan LSTM expects 'train' to be false.");
  TORCH_INTERNAL_ASSERT(
      !bidirectional, "Vulkan LSTM expects 'bidirectional' to be false.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan LSTM expects 'dropout' to be 0.0.");

  packed_.reserve(Packed::NumArgs);
  packed_.emplace_back(pack_lstm_linear_op_contexts(params_cpu, num_layers));
  packed_.emplace_back(has_biases);
  packed_.emplace_back(num_layers);
  packed_.emplace_back(dropout);
  packed_.emplace_back(train);
  packed_.emplace_back(bidirectional);
  packed_.emplace_back(batch_first);
}

LstmPackedContext LstmPackedContext::pack(c10::impl::GenericList unpacked) {
  return LstmPackedContext(
      unpacked.get(Unpacked::Params).toTensorVector(),
      unpacked.get(Unpacked::hasBiases).toBool(),
      unpacked.get(Unpacked::NumLayers).toInt(),
      unpacked.get(Unpacked::Dropout).toDouble(),
      unpacked.get(Unpacked::Train).toBool(),
      unpacked.get(Unpacked::Bidirectional).toBool(),
      unpacked.get(Unpacked::BatchFirst).toBool());
}

const c10::impl::GenericList LstmPackedContext::unpack() const {
  c10::impl::GenericList unpacked_lstm_context{c10::AnyType::get()};
  unpacked_lstm_context.reserve(Unpacked::NumArgs);

  const c10::List<c10::IValue> packed_linear_contexts =
      get_val(Packed::LinearContexts).toList();

  const int64_t num_layers = get_val(Packed::NumLayers).toInt();
  const int64_t linear_contexts_per_layer = 8;

  std::vector<Tensor> params_cpu;
  params_cpu.reserve(num_layers * linear_contexts_per_layer);

  for (c10::IValue packed_linear_context : packed_linear_contexts) {
    const c10::impl::GenericList unpacked_linear_context =
        packed_linear_context.toCustomClass<LinearPackedContext>()->unpack();

    TORCH_CHECK(
        unpacked_linear_context.size() > 0u,
        "unpacked_linear_context does not have any elements!");

    params_cpu.emplace_back(
        unpacked_linear_context.get(LinearPackedContext::Unpacked::Weight)
            .toTensor()
            .t());
    params_cpu.emplace_back(
        unpacked_linear_context.get(LinearPackedContext::Unpacked::Bias)
            .toTensor());
  }
  unpacked_lstm_context.emplace_back(params_cpu);
  for (int64_t i = 1; i < 7; ++i) {
    unpacked_lstm_context.emplace_back(get_val(i));
  }

  return unpacked_lstm_context;
}

c10::intrusive_ptr<LstmPackedContext> create_lstm_context(
    std::vector<Tensor>&& params_cpu,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return c10::make_intrusive<LstmPackedContext>(LstmPackedContext(
      params_cpu,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first));
}

std::tuple<Tensor, Tensor, Tensor> run_lstm_context(
    const Tensor& input_vk, // input sequence (vulkan)
    const Tensor& hx_vk, // initial hidden state (vulkan)
    const Tensor& cx_vk, // initial cell state (vulkan)
    const c10::intrusive_ptr<LstmPackedContext>& lstm_context) {
  TORCH_INTERNAL_ASSERT(
      input_vk.sizes().size() == 3, "Vulkan LSTM expects input dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      hx_vk.sizes().size() == 3,
      "Vulkan LSTM expects hidden state dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      cx_vk.sizes().size() == 3,
      "Vulkan LSTM expects cell state dims to be 3.");

  const int64_t num_layers =
      lstm_context->get_val(LstmPackedContext::Packed::NumLayers).toInt();
  const bool batch_first =
      lstm_context->get_val(LstmPackedContext::Packed::BatchFirst).toBool();
  const auto batch_size = input_vk.size(0);
  const auto seq_length = input_vk.size(1);

  TORCH_INTERNAL_ASSERT(
      (batch_size == 1 && seq_length == 1) || batch_first,
      "Vulkan gru expects batch-first input");

  const c10::List<c10::IValue> packed_linear_op_contexts =
      lstm_context->get_val(LstmPackedContext::Packed::LinearContexts).toList();

  const int64_t linear_op_contexts_per_layer = 8;
  // (b_ii, w_ii), (b_hi, w_hi), (b_if, w_if), (b_hf, w_hf),
  // (b_ig, w_ig), (b_hg, w_hg), (b_io, w_io), (b_ho, w_ho)

  std::vector<at::Tensor> h_n_list; // hidden state output
  std::vector<at::Tensor> c_n_list; // cell state output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x = input_vk.reshape({batch_size * seq_length, input_vk.size(2)});

  h_n_list.reserve(num_layers);
  c_n_list.reserve(num_layers);

  for (int64_t l = 0; l < num_layers; ++l) {
    // extract each hidden state and squeeze into 2D dim
    auto h = at::slice(hx_vk, 0, l, l + 1, 1);
    h = h.reshape({h.size(0) * h.size(1), h.size(2)});

    auto c = at::slice(cx_vk, 0, l, l + 1, 1);
    c = c.reshape({c.size(0) * c.size(1), c.size(2)});

    const auto& cxt_ii =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 0]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_hi =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 1]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_if =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 2]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_hf =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 3]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_ig =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 4]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_hg =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 5]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_io =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 6]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_ho =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 7]
            .toCustomClass<LinearPackedContext>();

    const auto& i = at::sigmoid(
        run_linear_context(x, cxt_ii) + run_linear_context(h, cxt_hi));
    // cxt_ii->run(x, 1.0f, 1.0f) + cxt_hi->run(h, 1.0f, 1.0f));
    const auto& f = at::sigmoid(
        run_linear_context(x, cxt_if) + run_linear_context(h, cxt_hf));
    // cxt_if->run(x, 1.0f, 1.0f) + cxt_hf->run(h, 1.0f, 1.0f));
    const auto& g =
        at::tanh(run_linear_context(x, cxt_ig) + run_linear_context(h, cxt_hg));
    // cxt_ig->run(x, 1.0f, 1.0f) + cxt_hg->run(h, 1.0f, 1.0f));
    const auto& o = at::sigmoid(
        run_linear_context(x, cxt_io) + run_linear_context(h, cxt_ho));
    // cxt_io->run(x, 1.0f, 1.0f) + cxt_ho->run(h, 1.0f, 1.0f));
    c = f * c + i * g;
    h = o * at::tanh(c);
    x = h; // next input
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 2D to 4D for cat op
    c_n_list.emplace_back(
        c.reshape({1, 1, c.size(0), c.size(1)})); // 2D to 4D for cat op
  }

  auto h_n = at::cat(h_n_list, 1);
  auto c_n = at::cat(c_n_list, 1);
  x = x.reshape({batch_size, seq_length, x.size(1)});
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  c_n = c_n.reshape({c_n.size(0) * c_n.size(1), c_n.size(2), c_n.size(3)});
  return std::tuple<Tensor, Tensor, Tensor>(x, h_n, c_n);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
