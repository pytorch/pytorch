#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
//
// input_vk: input tensor of shape (L, N, H_in) when batch_first=False
//                                 (N, L, H_in) when batch_first=True containing
//                                 the features of the input sequence
// hx_vk: initial hidden state for each element in the batch. tensor of shape (D
// * num_layers, N, H_out) output: tensor of shape (N, L, D * H_out)) when
// batch_first=True h_n: tensor of shape (D * num_layers, N, H_out)
//
//  where
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
      batch_first, "Vulkan gru expects 'batch_first' to be true.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan gru expects 'dropout' to be 0.0.");

  const auto hidden_size = hx_vk.size(2);
  std::vector<at::Tensor> h_n_list; // hidden output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x =
      input_vk.reshape({input_vk.size(0) * input_vk.size(1), input_vk.size(2)});

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
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  return std::tuple<Tensor, Tensor>(x, h_n);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::gru.input"), TORCH_FN(gru_input));
}

#endif /* USE_VULKAN_API */

} // namespace

std::vector<c10::intrusive_ptr<VulkanOpContext>> pack_linear_op_contexts(
    const std::vector<Tensor>& params_cpu,
    int64_t num_layers) {
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()) == 4 * num_layers,
      "Vulkan gru expects 'params_cpu' size to be 4 * 'num_layers'.");
  std::vector<c10::intrusive_ptr<VulkanOpContext>> linear_op_contexts;
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

VulkanOpContext gru_context_create(
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
      batch_first, "Vulkan gru expects 'batch_first' to be true.");
  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan gru expects 'dropout' to be 0.0.");

  c10::impl::GenericList packed_context{c10::AnyType::get()};
  packed_context.reserve(7);
  packed_context.emplace_back(pack_linear_op_contexts(params_cpu, num_layers));
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

std::tuple<Tensor, Tensor> gru_context_run(
    const Tensor& input_vk, // input sequence (vulkan)
    const Tensor& hx_vk, // initial hidden state (vulkan)
    const c10::impl::GenericList& packed_context,
    const c10::impl::GenericList& unpacked_context) {
  TORCH_INTERNAL_ASSERT(
      input_vk.sizes().size() == 3,
      "Vulkan gru expects 'input_vk' dims to be 3.");
  TORCH_INTERNAL_ASSERT(
      hx_vk.sizes().size() == 3, "Vulkan gru expects 'hx_vk' dims to be 3.");

  const c10::List<c10::IValue> packed_linear_op_contexts =
      packed_context.get(0).toList();
  const int64_t packed_num_layers = packed_context.get(2).toInt();

  const int64_t linear_op_contexts_per_layer =
      6; // (b_ir, w_ir), (b_hr, w_hr), (b_iz, w_iz), (b_hz, w_hz), (b_in,
         // w_in), (b_hn, w_hn)
  std::vector<at::Tensor> h_n_list; // hidden output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x =
      input_vk.reshape({input_vk.size(0) * input_vk.size(1), input_vk.size(2)});

  for (int64_t i = 0; i < packed_num_layers; ++i) {
    // extract each hidden state and squeeze into 2D dim
    auto h = at::slice(hx_vk, 0, i, i + 1, 1);
    h = h.reshape({h.size(0) * h.size(1), h.size(2)});

    const auto& cxt_ir =
        packed_linear_op_contexts[i * linear_op_contexts_per_layer + 0]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_hr =
        packed_linear_op_contexts[i * linear_op_contexts_per_layer + 1]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_iz =
        packed_linear_op_contexts[i * linear_op_contexts_per_layer + 2]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_hz =
        packed_linear_op_contexts[i * linear_op_contexts_per_layer + 3]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_in =
        packed_linear_op_contexts[i * linear_op_contexts_per_layer + 4]
            .toCustomClass<VulkanOpContext>();
    const auto& cxt_hn =
        packed_linear_op_contexts[i * linear_op_contexts_per_layer + 5]
            .toCustomClass<VulkanOpContext>();

    const auto& r = at::sigmoid(
        linear_context_run(
            x, cxt_ir->get_packed(), cxt_ir->get_unpacked(), 1.0f, 1.0f) +
        linear_context_run(
            h, cxt_hr->get_packed(), cxt_hr->get_unpacked(), 1.0f, 1.0f));
    const auto& z = at::sigmoid(
        linear_context_run(
            x, cxt_iz->get_packed(), cxt_iz->get_unpacked(), 1.0f, 1.0f) +
        linear_context_run(
            h, cxt_hz->get_packed(), cxt_hz->get_unpacked(), 1.0f, 1.0f));
    const auto& n = at::tanh(
        linear_context_run(
            x, cxt_in->get_packed(), cxt_in->get_unpacked(), 1.0f, 1.0f) +
        r *
            (linear_context_run(
                h, cxt_hn->get_packed(), cxt_hn->get_unpacked(), 1.0f, 1.0f)));
    h = (z * (-1) + 1) * n + z * h;
    x = h; // next input
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 2D to 4D for cat op
  }

  auto h_n = at::cat(h_n_list, 1);
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  return std::tuple<Tensor, Tensor>(x, h_n);
}

c10::intrusive_ptr<VulkanOpContext> create_gru_context(
    std::vector<Tensor>&& params_cpu,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return c10::make_intrusive<VulkanOpContext>(gru_context_create(
      params_cpu,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first));
}

std::tuple<Tensor, Tensor> run_gru_context(
    const Tensor& input_vk,
    const Tensor& hx_vk,
    const c10::intrusive_ptr<VulkanOpContext>& vulkan_context) {
  return gru_context_run(
      input_vk,
      hx_vk,
      vulkan_context->get_packed(),
      vulkan_context->get_unpacked());
}

/* Backwards compatibility */
GruOpContext::GruOpContext(VulkanOpContext vulkan_context)
    : vulkan_context_{std::move(vulkan_context)} {}

GruOpContext GruOpContext::create(
    const std::vector<Tensor>& params_cpu, // weights/biases (cpu)
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return GruOpContext{gru_context_create(
      params_cpu,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first)};
}

std::tuple<Tensor, Tensor> GruOpContext::run(
    const Tensor& input_vk, // input sequence (vulkan)
    const Tensor& hx_vk) const { // initial hidden state (vulkan)
  return gru_context_run(
      input_vk,
      hx_vk,
      vulkan_context_.get_packed(),
      vulkan_context_.get_unpacked());
}

GruOpContext::State GruOpContext::unpack() const {
  const c10::impl::GenericList unpacked_ =
      std::get<1>(vulkan_context_.get_state());
  const std::vector<Tensor> unpacked_params_cpu =
      unpacked_.get(0).toTensorVector();
  const bool unpacked_has_biases = unpacked_.get(1).toBool();
  const int64_t unpacked_num_layers = unpacked_.get(2).toInt();
  const double unpacked_dropout = unpacked_.get(3).toDouble();
  const bool unpacked_train = unpacked_.get(4).toBool();
  const bool unpacked_bidirectional = unpacked_.get(5).toBool();
  const bool unpacked_batch_first = unpacked_.get(6).toBool();
  return GruOpContext::State{
      unpacked_params_cpu,
      unpacked_has_biases,
      unpacked_num_layers,
      unpacked_dropout,
      unpacked_train,
      unpacked_bidirectional,
      unpacked_batch_first,
  };
}

c10::intrusive_ptr<GruOpContext> gru_prepack(
    std::vector<Tensor>&& params_cpu,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return c10::make_intrusive<GruOpContext>(GruOpContext::create(
      params_cpu,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first));
}

std::tuple<Tensor, Tensor> gru_run(
    const Tensor& input_vk,
    const Tensor& hx_vk,
    const c10::intrusive_ptr<GruOpContext>& context) {
  return context->run(input_vk, hx_vk);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
