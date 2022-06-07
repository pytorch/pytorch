#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
//
// input_vk: input tensor of shape (L, N, H_in) when batch_first=False or (N, L, H_in) when batch_first=True
//           containing the features of the input sequence
// hx_vk: tensor of shape (D * num_layers, N, H_out) containing the initial hidden state for each element in the input sequence.
// cx_vk: tensor of shape (D * num_layers, N, H_cell) containing the initial cell state for each element in the input sequence.
// output: tensor of shape (L, N, D * H_out) when batch_first=False or (N, L, D * H_out) when batch_first=True
//         containing the output features (h_t) from the last layer of the LSTM, for each t
// h_n: tensor of shape (D * num_layers, N, H_out) containing the final hidden state for each element in the sequence.
// c_n: tensor of shape (D * num_layers, N, H_cell) containing the final cell state for each element in the sequence.
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
  const Tensor & input_vk,  // input sequence (vulkan)
  TensorList hx,    // initial hidden state (vulkan) & initial cell state (vulkan)
  TensorList params_cpu,    // weights/biases (cpu)
  bool has_biases,
  int64_t num_layers,
  double dropout,
  bool train,
  bool bidirectional,
  bool batch_first) {
  TORCH_CHECK(hx[0].size(2) == hx[1].size(2), "Vulkan LSTM with projections is not supported");
  TORCH_CHECK(static_cast<int64_t>(params_cpu.size()), "Vulkan LSTM expects 'params_cpu' size to be 4 * 'num_layers'.");
  TORCH_INTERNAL_ASSERT(input_vk.sizes().size() == 3, "Vulkan LSTM expects input dims to be 3.");
  TORCH_INTERNAL_ASSERT(hx[0].sizes().size() == 3, "Vulkan LSTM expects hidden state dims to be 3.");
  TORCH_INTERNAL_ASSERT(hx[1].sizes().size() == 3, "Vulkan LSTM expects cell state dims to be 3.");
  TORCH_INTERNAL_ASSERT(has_biases, "Vulkan LSTM expects 'has_biases' to be true.");
  TORCH_INTERNAL_ASSERT(!train, "Vulkan LSTM expects 'train' to be false.");
  TORCH_INTERNAL_ASSERT(!bidirectional, "Vulkan LSTM expects 'bidirectional' to be false.");
  TORCH_INTERNAL_ASSERT(batch_first, "Vulkan LSTM expects 'batch_first' to be true.");
  TORCH_INTERNAL_ASSERT(dropout < std::numeric_limits<double>::epsilon()*1000, "Vulkan LSTM expects 'dropout' to be 0.0.");

  const Tensor& hx_vk = hx[0];
  const Tensor& cx_vk = hx[1];

  const auto hidden_size = hx_vk.size(2);
  std::vector<at::Tensor> h_n_list;  // hidden state output
  std::vector<at::Tensor> c_n_list;  // cell state output

  // reshape to 2D due to Vulkan at::mm op accepts only 2D
  auto x = input_vk.reshape({input_vk.size(0) * input_vk.size(1), input_vk.size(2)});

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

    const auto& i = at::sigmoid(at::addmm(b_ii, x, w_ii.t()) + at::addmm(b_hi, h, w_hi.t()));
    const auto& f = at::sigmoid(at::addmm(b_if, x, w_if.t()) + at::addmm(b_hf, h, w_hf.t()));
    const auto& g = at::tanh(at::addmm(b_ig, x, w_ig.t()) + at::addmm(b_hg, h, w_hg.t()));
    const auto& o = at::sigmoid(at::addmm(b_io, x, w_io.t()) + at::addmm(b_ho, h, w_ho.t()));
    c = f * c + i * g;
    h = o * at::tanh(c);
    x = h;  // next input
    h_n_list.emplace_back(h.reshape({1, 1, h.size(0), h.size(1)}));  // 2D to 4D for cat op
    c_n_list.emplace_back(c.reshape({1, 1, c.size(0), c.size(1)}));  // 2D to 4D for cat op
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
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
