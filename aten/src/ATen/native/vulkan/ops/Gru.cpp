#include <ATen/TensorOperators.h>
#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/addmm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/gru.h>
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
// input_vk: input tensor containing the features of the input sequence
//           tensor of shape (N, L, H_in) when batch_first=True
//                           (L, N, H_in) when batch_first=False
//
// hx_vk: initial hidden state for each element in the batch.
//        tensor of shape (D * num_layers, N, H_out)
//
// output: tensor of shape (N, L, D * H_out) when batch_first=True
//                         (L, N, D * H_out) when batch_first=False
//
// h_n: tensor of shape (D * num_layers, N, H_out)
//
// where
//    L = sequence length
//    N = batch size
//    D = 2 if bidirectional=True otherwise 1
//    H_in = input_size (# of expected features in the input x)
//    H_out = hidden_size (# of features in the hidden state h)
//
std::tuple<Tensor, Tensor> gru_input(
    const Tensor& input_vk, // input sequence (vulkan)
    const Tensor& hx_vk, // initial hidden state (vulkan)
    TensorList params_cpu, // weights/biases (cpu)
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()) == 4 * num_layers,
      "Vulkan gru expects 'params_cpu' size to be 4 * 'num_layers'.");
  TORCH_INTERNAL_ASSERT(
      input_vk.sizes().size() == 3,
      "Vulkan gru expects 'input_vk' dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      hx_vk.sizes().size() == 3, "Vulkan gru expects 'hx_vk' dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      has_biases, "Vulkan gru expects 'has_biases' to be true.");
  TORCH_INTERNAL_ASSERT(!train, "Vulkan gru expects 'train' to be false.");
  TORCH_INTERNAL_ASSERT(
      !bidirectional, "Vulkan gru expects 'bidirectional' to be false.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan gru expects 'dropout' to be 0.0.");

  const auto batch_size = input_vk.size(0);
  const auto seq_length = input_vk.size(1);

  TORCH_INTERNAL_ASSERT(
      (batch_size == 1 && seq_length == 1) || batch_first,
      "Vulkan gru expects batch-first input");

  const auto hidden_size = hx_vk.size(2);
  std::vector<at::Tensor> h_n_list; // hidden output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x = input_vk.reshape({batch_size * seq_length, input_vk.size(2)});

  for (int64_t i = 0; i < num_layers; ++i) {
    // extract each hidden state and squeeze into 2D dim
    auto h = at::slice(hx_vk, 0, i, i + 1, 1);
    h = h.reshape({h.size(0) * h.size(1), h.size(2)});

    const auto& w_ih = params_cpu[i * 4];
    const auto& w_hh = params_cpu[i * 4 + 1];
    const auto& b_ih = params_cpu[i * 4 + 2];
    const auto& b_hh = params_cpu[i * 4 + 3];

    const auto& w_i_rzn = w_ih.split(hidden_size);
    const auto& w_h_rzn = w_hh.split(hidden_size);
    const auto& b_i_rzn = b_ih.split(hidden_size);
    const auto& b_h_rzn = b_hh.split(hidden_size);

    const auto& w_ir = w_i_rzn[0];
    const auto& w_iz = w_i_rzn[1];
    const auto& w_in = w_i_rzn[2];
    const auto& w_hr = w_h_rzn[0];
    const auto& w_hz = w_h_rzn[1];
    const auto& w_hn = w_h_rzn[2];
    const auto& b_ir = b_i_rzn[0];
    const auto& b_iz = b_i_rzn[1];
    const auto& b_in = b_i_rzn[2];
    const auto& b_hr = b_h_rzn[0];
    const auto& b_hz = b_h_rzn[1];
    const auto& b_hn = b_h_rzn[2];

    const auto& r = at::sigmoid(
        at::addmm(b_ir, x, w_ir.t()) + at::addmm(b_hr, h, w_hr.t()));
    const auto& z = at::sigmoid(
        at::addmm(b_iz, x, w_iz.t()) + at::addmm(b_hz, h, w_hz.t()));
    const auto& n = at::tanh(
        at::addmm(b_in, x, w_in.t()) + r * (at::addmm(b_hn, h, w_hn.t())));
    h = (z * (-1) + 1) * n + z * h;
    x = h; // next input
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 2D to 4D for cat op
  }

  auto h_n = at::cat(h_n_list, 1);
  x = x.reshape({batch_size, seq_length, x.size(1)});
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  return std::tuple<Tensor, Tensor>(x, h_n);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::gru.input"), TORCH_FN(gru_input));
}

#endif /* USE_VULKAN_API */

} // namespace

static std::vector<c10::intrusive_ptr<LinearPackedContext>>
pack_linear_op_contexts(
    const std::vector<Tensor>& params_cpu,
    int64_t num_layers) {
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()) == 4 * num_layers,
      "Vulkan gru expects 'params_cpu' size to be 4 * 'num_layers'."
      " But 'params_cpu' has size: ",
      params_cpu.size(),
      " and 'num_layers' is: ",
      num_layers);
  std::vector<c10::intrusive_ptr<LinearPackedContext>> linear_op_contexts;
  linear_op_contexts.reserve(num_layers * 6);

  for (int64_t i = 0; i < num_layers; ++i) {
    const auto& w_ih = params_cpu.at(i * 4);
    const auto& w_hh = params_cpu.at(i * 4 + 1);
    const auto& b_ih = params_cpu.at(i * 4 + 2);
    const auto& b_hh = params_cpu.at(i * 4 + 3);
    const auto& hidden_size = w_ih.size(0) / 3;

    const auto& w_i_rzn = w_ih.split(hidden_size);
    const auto& w_h_rzn = w_hh.split(hidden_size);
    const auto& b_i_rzn = b_ih.split(hidden_size);
    const auto& b_h_rzn = b_hh.split(hidden_size);

    const auto& w_ir = w_i_rzn[0];
    const auto& w_iz = w_i_rzn[1];
    const auto& w_in = w_i_rzn[2];
    const auto& w_hr = w_h_rzn[0];
    const auto& w_hz = w_h_rzn[1];
    const auto& w_hn = w_h_rzn[2];
    const auto& b_ir = b_i_rzn[0];
    const auto& b_iz = b_i_rzn[1];
    const auto& b_in = b_i_rzn[2];
    const auto& b_hr = b_h_rzn[0];
    const auto& b_hz = b_h_rzn[1];
    const auto& b_hn = b_h_rzn[2];

    linear_op_contexts.emplace_back(create_linear_context(w_ir.t(), b_ir));
    linear_op_contexts.emplace_back(create_linear_context(w_hr.t(), b_hr));
    linear_op_contexts.emplace_back(create_linear_context(w_iz.t(), b_iz));
    linear_op_contexts.emplace_back(create_linear_context(w_hz.t(), b_hz));
    linear_op_contexts.emplace_back(create_linear_context(w_in.t(), b_in));
    linear_op_contexts.emplace_back(create_linear_context(w_hn.t(), b_hn));
  }
  return linear_op_contexts;
}

GruPackedContext::GruPackedContext(
    const std::vector<Tensor>& params_cpu, // weights/biases (cpu)
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  TORCH_INTERNAL_ASSERT(
      has_biases, "Vulkan gru expects 'has_biases' to be true.");
  TORCH_INTERNAL_ASSERT(!train, "Vulkan gru expects 'train' to be false.");
  TORCH_INTERNAL_ASSERT(
      !bidirectional, "Vulkan gru expects 'bidirectional' to be false.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan gru expects 'dropout' to be 0.0.");

  packed_.reserve(Packed::NumArgs);
  packed_.emplace_back(pack_linear_op_contexts(params_cpu, num_layers));
  packed_.emplace_back(has_biases);
  packed_.emplace_back(num_layers);
  packed_.emplace_back(dropout);
  packed_.emplace_back(train);
  packed_.emplace_back(bidirectional);
  packed_.emplace_back(batch_first);
}

GruPackedContext GruPackedContext::pack(c10::impl::GenericList unpacked) {
  return GruPackedContext(
      unpacked.get(Unpacked::Params).toTensorVector(),
      unpacked.get(Unpacked::hasBiases).toBool(),
      unpacked.get(Unpacked::NumLayers).toInt(),
      unpacked.get(Unpacked::Dropout).toDouble(),
      unpacked.get(Unpacked::Train).toBool(),
      unpacked.get(Unpacked::Bidirectional).toBool(),
      unpacked.get(Unpacked::BatchFirst).toBool());
}

const c10::impl::GenericList GruPackedContext::unpack() const {
  c10::impl::GenericList unpacked_gru_context{c10::AnyType::get()};
  unpacked_gru_context.reserve(Unpacked::NumArgs);

  const c10::List<c10::IValue> packed_linear_contexts =
      get_val(Packed::LinearContexts).toList();

  const int64_t num_layers = get_val(Packed::NumLayers).toInt();
  const int64_t linear_contexts_per_layer = 6;

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
  unpacked_gru_context.emplace_back(params_cpu);
  for (int64_t i = 1; i < Unpacked::NumArgs; ++i) {
    unpacked_gru_context.emplace_back(get_val(i));
  }

  return unpacked_gru_context;
}

c10::intrusive_ptr<GruPackedContext> create_gru_context(
    std::vector<Tensor>&& params_cpu,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return c10::make_intrusive<GruPackedContext>(GruPackedContext(
      params_cpu,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first));
}

std::tuple<Tensor, Tensor> run_gru_context(
    const Tensor& input_vk, // input sequence (vulkan)
    const Tensor& hx_vk, // initial hidden state (vulkan)
    const c10::intrusive_ptr<GruPackedContext>& gru_context) {
  TORCH_INTERNAL_ASSERT(
      input_vk.sizes().size() == 3,
      "Vulkan gru expects 'input_vk' dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      hx_vk.sizes().size() == 3, "Vulkan gru expects 'hx_vk' dims to be 3.");

  const int64_t num_layers =
      gru_context->get_val(GruPackedContext::Packed::NumLayers).toInt();
  const bool batch_first =
      gru_context->get_val(GruPackedContext::Packed::BatchFirst).toBool();
  const auto batch_size = input_vk.size(0);
  const auto seq_length = input_vk.size(1);

  TORCH_INTERNAL_ASSERT(
      (batch_size == 1 && seq_length == 1) || batch_first,
      "Vulkan gru expects batch-first input");

  const c10::List<c10::IValue> packed_linear_contexts =
      gru_context->get_val(GruPackedContext::Packed::LinearContexts).toList();

  const int64_t linear_contexts_per_layer = 6;
  // (b_ir, w_ir), (b_hr, w_hr), (b_iz, w_iz),
  // (b_hz, w_hz), (b_in,cw_in), (b_hn, w_hn)
  std::vector<at::Tensor> h_n_list; // hidden output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x = input_vk.reshape({batch_size * seq_length, input_vk.size(2)});

  for (int64_t i = 0; i < num_layers; ++i) {
    // extract each hidden state and squeeze into 2D dim
    auto h = at::slice(hx_vk, 0, i, i + 1, 1);
    h = h.reshape({h.size(0) * h.size(1), h.size(2)});

    const auto& cxt_ir =
        packed_linear_contexts[i * linear_contexts_per_layer + 0]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_hr =
        packed_linear_contexts[i * linear_contexts_per_layer + 1]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_iz =
        packed_linear_contexts[i * linear_contexts_per_layer + 2]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_hz =
        packed_linear_contexts[i * linear_contexts_per_layer + 3]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_in =
        packed_linear_contexts[i * linear_contexts_per_layer + 4]
            .toCustomClass<LinearPackedContext>();
    const auto& cxt_hn =
        packed_linear_contexts[i * linear_contexts_per_layer + 5]
            .toCustomClass<LinearPackedContext>();

    const auto& r = at::sigmoid(
        run_linear_context(x, cxt_ir) + run_linear_context(h, cxt_hr));
    // cxt_ir->run(x, 1.0f, 1.0f) + cxt_hr->run(h, 1.0f, 1.0f));
    const auto& z = at::sigmoid(
        run_linear_context(x, cxt_iz) + run_linear_context(h, cxt_hz));
    // cxt_iz->run(x, 1.0f, 1.0f) + cxt_hz->run(h, 1.0f, 1.0f));
    const auto& n = at::tanh(
        run_linear_context(x, cxt_in) + r * run_linear_context(h, cxt_hn));
    // cxt_in->run(x, 1.0f, 1.0f) + r * (cxt_hn->run(h, 1.0f, 1.0f)));
    h = (z * (-1) + 1) * n + z * h;
    x = h; // next input
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 2D to 4D for cat op
  }

  auto h_n = at::cat(h_n_list, 1);
  x = x.reshape({batch_size, seq_length, x.size(1)});
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  return std::tuple<Tensor, Tensor>(x, h_n);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
