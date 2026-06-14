#include <ATen/ATen.h>
#include <ATen/native/RNN.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace at::native {

namespace {

void lstm_onednn_xpu(
    Tensor& output,
    Tensor& hy,
    Tensor& cy,
    const Tensor& input,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  using namespace at::native::onednn;
  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;

  TORCH_INTERNAL_ASSERT(
      !train, "oneDNN LSTM on XPU only supports inference mode");
  TORCH_INTERNAL_ASSERT(
      input.scalar_type() == kFloat || input.scalar_type() == kHalf ||
          input.scalar_type() == kBFloat16,
      "oneDNN LSTM on XPU only supports float, half, and bfloat16");

  auto& engine = GpuEngineManager::Instance().get_engine();
  auto& stream = GpuStreamManager::Instance().get_stream();

  auto input_ =
      batch_first ? input.transpose(0, 1).contiguous() : input.contiguous();
  auto hx_ = hx[0].contiguous(); // [num_layers * num_directions, batch, hidden]
  auto cx_ = hx[1].contiguous();

  int64_t seq_length = input_.size(0);
  int64_t mini_batch = input_.size(1);
  int64_t hidden_size = hx_.size(2);
  int64_t num_directions = bidirectional ? 2 : 1;
  int64_t weight_stride0 = has_biases ? 4 : 2;

  auto get_dt = [](const Tensor& t) -> dt {
    switch (t.scalar_type()) {
      case kFloat:
        return dt::f32;
      case kBFloat16:
        return dt::bf16;
      case kHalf:
        return dt::f16;
      default:
        TORCH_CHECK(false, "Unsupported dtype");
    }
  };

  auto data_type = get_dt(input_);
  auto rnn_direction = bidirectional
      ? dnnl::rnn_direction::bidirectional_concat
      : dnnl::rnn_direction::unidirectional_left2right;

  auto layer_input = input_;
  int64_t dst_hidden = hidden_size * num_directions;
  std::vector<Tensor> all_hy, all_cy;

  for (int64_t layer = 0; layer < num_layers; layer++) {
    int64_t layer_input_size = layer_input.size(2);

    // Gather weights for this layer: stack directions along dim 1
    // PyTorch params layout: [w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0,
    //                          w_ih_l0_reverse, w_hh_l0_reverse, ...]
    std::vector<Tensor> w_ih_list, w_hh_list, bias_list, hx_list, cx_list;
    for (int64_t dir = 0; dir < num_directions; dir++) {
      int64_t idx = (layer * num_directions + dir) * weight_stride0;
      w_ih_list.push_back(
          params[idx + 0].reshape({1, 1, 4, hidden_size, layer_input_size}));
      w_hh_list.push_back(
          params[idx + 1].reshape({1, 1, 4, hidden_size, hidden_size}));
      if (has_biases) {
        bias_list.push_back((params[idx + 2] + params[idx + 3])
                                .reshape({1, 1, 4, hidden_size}));
      } else {
        bias_list.push_back(
            at::zeros({1, 1, 4, hidden_size}, layer_input.options()));
      }
      int64_t hx_idx = layer * num_directions + dir;
      hx_list.push_back(hx_[hx_idx].unsqueeze(0).unsqueeze(0));
      cx_list.push_back(cx_[hx_idx].unsqueeze(0).unsqueeze(0));
    }

    // [1, num_directions, 4, hidden, input] for weights_layer
    auto w_ih = at::cat(w_ih_list, 1).contiguous();
    auto w_hh = at::cat(w_hh_list, 1).contiguous();
    auto bias = at::cat(bias_list, 1).contiguous();
    // [1, num_directions, batch, hidden]
    auto layer_hx = at::cat(hx_list, 1).contiguous();
    auto layer_cx = at::cat(cx_list, 1).contiguous();

    // Memory descriptors
    auto src_layer_md = dnnl::memory::desc(
        {seq_length, mini_batch, layer_input_size}, data_type, tag::tnc);
    auto src_iter_md = dnnl::memory::desc(
        {1, num_directions, mini_batch, hidden_size}, data_type, tag::ldnc);
    auto src_iter_c_md = dnnl::memory::desc(
        {1, num_directions, mini_batch, hidden_size}, data_type, tag::ldnc);
    auto weights_layer_md = dnnl::memory::desc(
        {1, num_directions, layer_input_size, 4, hidden_size},
        data_type,
        tag::ldgoi);
    auto weights_iter_md = dnnl::memory::desc(
        {1, num_directions, hidden_size, 4, hidden_size},
        data_type,
        tag::ldgoi);
    auto bias_md = dnnl::memory::desc(
        {1, num_directions, 4, hidden_size}, data_type, tag::ldgo);
    auto dst_layer_md = dnnl::memory::desc(
        {seq_length, mini_batch, dst_hidden}, data_type, tag::tnc);
    auto dst_iter_md = dnnl::memory::desc(
        {1, num_directions, mini_batch, hidden_size}, data_type, tag::ldnc);
    auto dst_iter_c_md = dnnl::memory::desc(
        {1, num_directions, mini_batch, hidden_size}, data_type, tag::ldnc);

    dnnl::primitive_attr pattr;
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = dnnl::lstm_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        rnn_direction,
        src_layer_md,
        src_iter_md,
        src_iter_c_md,
        weights_layer_md,
        weights_iter_md,
        bias_md,
        dst_layer_md,
        dst_iter_md,
        dst_iter_c_md,
        pattr);

    auto layer_out =
        at::empty({seq_length, mini_batch, dst_hidden}, layer_input.options());
    auto out_hy = at::empty(
        {1, num_directions, mini_batch, hidden_size}, layer_input.options());
    auto out_cy = at::empty(
        {1, num_directions, mini_batch, hidden_size}, layer_input.options());

    size_t scratchpad_size = pd.scratchpad_desc().get_size();
    auto scratchpad_tensor = at::empty(
        {static_cast<int64_t>(scratchpad_size)},
        layer_input.options().dtype(at::kByte));

    auto src_layer_mem =
        make_onednn_memory(src_layer_md, engine, layer_input.data_ptr());
    auto src_iter_mem =
        make_onednn_memory(src_iter_md, engine, layer_hx.data_ptr());
    auto src_iter_c_mem =
        make_onednn_memory(src_iter_c_md, engine, layer_cx.data_ptr());

    // Handle potential weight reorder
    auto expected_wl_md = pd.weights_layer_desc();
    auto expected_wi_md = pd.weights_iter_desc();

    dnnl::memory wl_mem, wi_mem;
    at::Tensor wl_reordered, wi_reordered;

    if (expected_wl_md != weights_layer_md) {
      wl_reordered = at::empty(
          {static_cast<int64_t>(expected_wl_md.get_size())},
          layer_input.options().dtype(at::kByte));
      auto wl_src =
          make_onednn_memory(weights_layer_md, engine, w_ih.data_ptr());
      wl_mem =
          make_onednn_memory(expected_wl_md, engine, wl_reordered.data_ptr());
      dnnl::reorder(wl_src, wl_mem).execute(stream, wl_src, wl_mem);
    } else {
      wl_mem = make_onednn_memory(weights_layer_md, engine, w_ih.data_ptr());
    }

    if (expected_wi_md != weights_iter_md) {
      wi_reordered = at::empty(
          {static_cast<int64_t>(expected_wi_md.get_size())},
          layer_input.options().dtype(at::kByte));
      auto wi_src =
          make_onednn_memory(weights_iter_md, engine, w_hh.data_ptr());
      wi_mem =
          make_onednn_memory(expected_wi_md, engine, wi_reordered.data_ptr());
      dnnl::reorder(wi_src, wi_mem).execute(stream, wi_src, wi_mem);
    } else {
      wi_mem = make_onednn_memory(weights_iter_md, engine, w_hh.data_ptr());
    }

    auto bias_mem = make_onednn_memory(bias_md, engine, bias.data_ptr());
    auto dst_layer_mem =
        make_onednn_memory(dst_layer_md, engine, layer_out.data_ptr());
    auto dst_iter_mem =
        make_onednn_memory(dst_iter_md, engine, out_hy.data_ptr());
    auto dst_iter_c_mem =
        make_onednn_memory(dst_iter_c_md, engine, out_cy.data_ptr());
    auto scratchpad_mem = make_onednn_memory(
        pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_LAYER, src_layer_mem},
        {DNNL_ARG_SRC_ITER, src_iter_mem},
        {DNNL_ARG_SRC_ITER_C, src_iter_c_mem},
        {DNNL_ARG_WEIGHTS_LAYER, wl_mem},
        {DNNL_ARG_WEIGHTS_ITER, wi_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST_LAYER, dst_layer_mem},
        {DNNL_ARG_DST_ITER, dst_iter_mem},
        {DNNL_ARG_DST_ITER_C, dst_iter_c_mem},
        {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
    };

    auto prim = dnnl::lstm_forward(pd);
    dnnl::sycl_interop::execute(prim, stream, args, {});

    layer_input = layer_out;

    // Collect hy/cy: out_hy is [1, num_directions, batch, hidden]
    for (int64_t dir = 0; dir < num_directions; dir++) {
      all_hy.push_back(out_hy.select(1, dir).squeeze(0));
      all_cy.push_back(out_cy.select(1, dir).squeeze(0));
    }
  }

  output = batch_first ? layer_input.transpose(0, 1).contiguous() : layer_input;
  hy = at::stack(all_hy, 0);
  cy = at::stack(all_cy, 0);
}

} // anonymous namespace

REGISTER_XPU_DISPATCH(lstm_mkldnn_stub, &lstm_onednn_xpu)

} // namespace at::native
