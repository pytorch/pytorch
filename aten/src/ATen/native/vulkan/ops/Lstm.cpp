#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
//
// input_vk: input tensor of shape (L, N, H_in) when batch_first=False or (N, L,
// H_in) when batch_first=True
//           containing the features of the input sequence
// hx_vk: tensor of shape (D * num_layers, N, H_out) containing the initial
// hidden state for each element in the input sequence. cx_vk: tensor of shape
// (D * num_layers, N, H_cell) containing the initial cell state for each
// element in the input sequence. output: tensor of shape (L, N, D * H_out) when
// batch_first=False or (N, L, D * H_out) when batch_first=True
//         containing the output features (h_t) from the last layer of the LSTM,
//         for each t
// h_n: tensor of shape (D * num_layers, N, H_out) containing the final hidden
// state for each element in the sequence. c_n: tensor of shape (D * num_layers,
// N, H_cell) containing the final cell state for each element in the sequence.
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
      batch_first, "Vulkan LSTM expects 'batch_first' to be true.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan LSTM expects 'dropout' to be 0.0.");

  const Tensor& hx_vk = hx[0];
  const Tensor& cx_vk = hx[1];

  const auto hidden_size = hx_vk.size(2);
  std::vector<at::Tensor> h_n_list; // hidden state output
  std::vector<at::Tensor> c_n_list; // cell state output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x =
      input_vk.reshape({input_vk.size(0) * input_vk.size(1), input_vk.size(2)});

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

std::vector<c10::intrusive_ptr<VulkanOpContext>> pack_lstm_linear_op_contexts(
    const std::vector<Tensor>& params_cpu,
    int64_t num_layers) {
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()) == 4 * num_layers,
      "Vulkan LSTM expects 'params_cpu' size to be 4 * 'num_layers'.");
  std::vector<c10::intrusive_ptr<VulkanOpContext>> linear_op_contexts;
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

VulkanOpContext lstm_context_create(
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
      batch_first, "Vulkan LSTM expects 'batch_first' to be true.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan LSTM expects 'dropout' to be 0.0.");

  c10::impl::GenericList packed_context{c10::AnyType::get()};
  packed_context.reserve(7);
  packed_context.emplace_back(
      pack_lstm_linear_op_contexts(params_cpu, num_layers));
  packed_context.emplace_back(has_biases);
  packed_context.emplace_back(num_layers);
  packed_context.emplace_back(dropout);
  packed_context.emplace_back(train);
  packed_context.emplace_back(bidirectional);
  packed_context.emplace_back(batch_first);

  c10::impl::GenericList unpacked_context{c10::AnyType::get()};
  unpacked_context.reserve(7);
  unpacked_context.emplace_back(params_cpu);
  unpacked_context.emplace_back(has_biases);
  unpacked_context.emplace_back(num_layers);
  unpacked_context.emplace_back(dropout);
  unpacked_context.emplace_back(train);
  unpacked_context.emplace_back(bidirectional);
  unpacked_context.emplace_back(batch_first);

  return VulkanOpContext::create(packed_context, unpacked_context);
}

std::tuple<Tensor, Tensor, Tensor> lstm_context_run(
    const Tensor& input_vk, // input sequence (vulkan)
    const Tensor& hx_vk, // initial hidden state (vulkan)
    const Tensor& cx_vk, // initial cell state (vulkan)
    const c10::impl::GenericList& packed_context,
    const c10::impl::GenericList& unpacked_context) {
  TORCH_INTERNAL_ASSERT(
      input_vk.sizes().size() == 3, "Vulkan LSTM expects input dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      hx_vk.sizes().size() == 3,
      "Vulkan LSTM expects hidden state dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      cx_vk.sizes().size() == 3,
      "Vulkan LSTM expects cell state dims to be 3.");

  const c10::List<c10::IValue> packed_linear_op_contexts =
      packed_context.get(0).toList();
  const int64_t packed_num_layers = packed_context.get(2).toInt();

  const int64_t linear_op_contexts_per_layer =
      8; // (b_ii, w_ii), (b_hi, w_hi), (b_if, w_if), (b_hf, w_hf), (b_ig,
         // w_ig), (b_hg, w_hg), (b_io, w_io), (b_ho, w_ho)
  std::vector<at::Tensor> h_n_list; // hidden state output
  std::vector<at::Tensor> c_n_list; // cell state output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x =
      input_vk.reshape({input_vk.size(0) * input_vk.size(1), input_vk.size(2)});

  h_n_list.reserve(packed_num_layers);
  c_n_list.reserve(packed_num_layers);

  for (int64_t l = 0; l < packed_num_layers; ++l) {
    // extract each hidden state and squeeze into 2D dim
    auto h = at::slice(hx_vk, 0, l, l + 1, 1);
    h = h.reshape({h.size(0) * h.size(1), h.size(2)});

    auto c = at::slice(cx_vk, 0, l, l + 1, 1);
    c = c.reshape({c.size(0) * c.size(1), c.size(2)});

    const auto& cxt_ii =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 0]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_hi =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 1]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_if =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 2]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_hf =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 3]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_ig =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 4]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_hg =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 5]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_io =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 6]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_ho =
        packed_linear_op_contexts[l * linear_op_contexts_per_layer + 7]
            .toCustomClass<VulkanOpContext>();

    const auto& i = at::sigmoid(
        linear_context_run(
            x, cxt_ii->get_packed(), cxt_ii->get_unpacked(), 1.0f, 1.0f) +
        linear_context_run(
            h, cxt_hi->get_packed(), cxt_hi->get_unpacked(), 1.0f, 1.0f));
    const auto& f = at::sigmoid(
        linear_context_run(
            x, cxt_if->get_packed(), cxt_if->get_unpacked(), 1.0f, 1.0f) +
        linear_context_run(
            h, cxt_hf->get_packed(), cxt_hf->get_unpacked(), 1.0f, 1.0f));
    const auto& g = at::tanh(
        linear_context_run(
            x, cxt_ig->get_packed(), cxt_ig->get_unpacked(), 1.0f, 1.0f) +
        linear_context_run(
            h, cxt_hg->get_packed(), cxt_hg->get_unpacked(), 1.0f, 1.0f));
    const auto& o = at::sigmoid(
        linear_context_run(
            x, cxt_io->get_packed(), cxt_io->get_unpacked(), 1.0f, 1.0f) +
        linear_context_run(
            h, cxt_ho->get_packed(), cxt_ho->get_unpacked(), 1.0f, 1.0f));
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
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  c_n = c_n.reshape({c_n.size(0) * c_n.size(1), c_n.size(2), c_n.size(3)});
  return std::tuple<Tensor, Tensor, Tensor>(x, h_n, c_n);
}

c10::intrusive_ptr<VulkanOpContext> create_lstm_context(
    std::vector<Tensor>&& params_cpu,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return c10::make_intrusive<VulkanOpContext>(lstm_context_create(
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
    const c10::intrusive_ptr<VulkanOpContext>& vulkan_context) {
  return lstm_context_run(
      input_vk,
      hx_vk,
      cx_vk,
      vulkan_context->get_packed(),
      vulkan_context->get_unpacked());
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
