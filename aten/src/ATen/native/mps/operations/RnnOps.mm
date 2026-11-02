#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/RNN.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/Matmul.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_lstm_mps_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/lstm_mps_backward_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <c10/util/TypeCast.h>
#include <limits>

namespace at::native {
namespace {
using namespace mps;
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Rnn_metallib.h>
#endif

// A batched at::mm_out runs MPSGraph on every timestep and closes the cell's encoder, so batches use the Metal kernels.
// A single row keeps ATen's native GEMV, and reductions above this size measured slower on the Metal kernels.
constexpr int64_t kMetalMmMaxReduction = 512;

void lstm_mm(Tensor& out, const Tensor& a, const Tensor& b) {
  const bool batched = a.size(0) > 1;
  if (batched && a.size(1) <= kMetalMmMaxReduction) {
    metal_mm_out(a, b, out);
  } else {
    at::mm_out(out, a, b);
  }
}

template <typename... Args>
void lstm_cell(MTLComputePipelineState_t pso, const Tensor& cell, const Args&... args) {
  if (cell.numel() == 0) {
    return;
  }
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pso, "lstm_cell", {cell}, stream);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, args..., c10::checked_convert<uint32_t>(cell.size(-1), "hidden size"));
      mtl_dispatch1DJob(encoder, pso, cell.numel());
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

} // namespace

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> _lstm_mps(const Tensor& input,
                                                                     TensorList hx,
                                                                     TensorList params,
                                                                     bool has_biases,
                                                                     int64_t num_layers,
                                                                     double dropout_p,
                                                                     bool train,
                                                                     bool bidirectional,
                                                                     bool batch_first) {
  check_attributes(input, params, hx, true);
  TORCH_CHECK_TYPE(
      supportedFloatingType(input), "LSTM on MPS supports float32, float16 and bfloat16, got ", input.scalar_type());
  TORCH_CHECK(hx[0].size(2) == hx[1].size(2), "LSTM with projections is not currently supported with MPS.");
  auto layer_input = (batch_first ? input.transpose(0, 1) : input).contiguous();
  const auto seq_len = layer_input.size(0), batch = layer_input.size(1), hidden = hx[0].size(2);
  const int64_t directions = bidirectional ? 2 : 1, stride = has_biases ? 4 : 2;
  TORCH_CHECK(seq_len > 0, "Expected sequence length to be larger than 0 in RNN");
  // Cell elements are indexed by a 32-bit thread position.
  TORCH_CHECK_NOT_IMPLEMENTED(batch * hidden <= std::numeric_limits<uint32_t>::max(),
                              "LSTM on MPS supports at most 2^32 - 1 elements per step, got batch ",
                              batch,
                              " and hidden size ",
                              hidden);
  const bool dropout = train && dropout_p > 0 && num_layers > 1;
  auto gates = at::empty({num_layers * directions, seq_len, batch, 4 * hidden}, input.options());
  // Keep the initial states followed by each recurrent step for backward.
  auto states = at::empty({num_layers * directions, 2, seq_len + 1, batch, hidden}, input.options());
  auto layers = at::empty({(num_layers - 1) * (dropout ? 2 : 1), seq_len, batch, directions * hidden}, input.options());
  auto hy = at::empty(hx[0].sizes(), input.options()), cy = at::empty(hx[1].sizes(), input.options());
  auto pso = lib.getPipelineStateForFunc("lstm_cell_forward_" + scalarToMetalTypeString(input));
  for (const auto layer : c10::irange(num_layers)) {
    std::vector<Tensor> outputs;
    for (const auto direction : c10::irange(directions)) {
      const auto index = layer * directions + direction, param = index * stride;
      TORCH_CHECK_VALUE(params[param].size(0) == 4 * hidden && params[param + 1].size(0) == 4 * hidden,
                        "LSTM weights must have ",
                        4 * hidden,
                        " rows, got ",
                        params[param].sizes(),
                        " and ",
                        params[param + 1].sizes());
      auto x = direction ? layer_input.flip({0}) : layer_input;
      auto z = gates[index], h = states[index][0], c = states[index][1];
      auto z_flat = z.view({seq_len * batch, 4 * hidden});
      at::mm_out(z_flat, x.reshape({seq_len * batch, x.size(2)}), params[param].t());
      h[0].copy_(hx[0][index]);
      c[0].copy_(hx[1][index]);
      auto bias = has_biases ? params[param + 2].add(params[param + 3]).expand({4 * hidden}).contiguous()
                             : at::zeros({4 * hidden}, input.options());
      const auto recurrent_weight = params[param + 1].contiguous().t();
      auto recurrent_gates = at::empty({batch, 4 * hidden}, input.options());
      for (const auto step : c10::irange(seq_len)) {
        lstm_mm(recurrent_gates, h[step], recurrent_weight);
        lstm_cell(pso, c[step], z[step], recurrent_gates, c[step], c[step + 1], h[step + 1], bias);
      }
      hy[index].copy_(h[seq_len]);
      cy[index].copy_(c[seq_len]);
      auto output = h.narrow(0, 1, seq_len);
      outputs.push_back(direction ? output.flip({0}) : output);
    }
    layer_input = directions == 1 ? outputs[0] : at::cat(outputs, -1);
    if (layer + 1 < num_layers) {
      if (dropout) {
        auto mask = layers[num_layers - 1 + layer];
        mask.bernoulli_(1.0 - dropout_p).div_(dropout_p == 1.0 ? 1.0 : 1.0 - dropout_p);
        layer_input = layer_input.mul(mask);
      }
      layers[layer].copy_(layer_input);
    }
  }
  auto output = directions == 1 ? layer_input.clone() : std::move(layer_input);
  if (batch_first) {
    output = output.transpose(0, 1).contiguous();
  }
  return {std::move(output), std::move(hy), std::move(cy), std::move(gates), std::move(states), std::move(layers)};
}

std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> lstm_mps_backward(const std::optional<Tensor>& grad_y_opt,
                                                                               const std::optional<Tensor>& grad_hy_opt,
                                                                               const std::optional<Tensor>& grad_cy_opt,
                                                                               const Tensor& z_state,
                                                                               const Tensor& cell_state_fwd,
                                                                               const Tensor& input,
                                                                               const Tensor& layersOutputs,
                                                                               TensorList hx,
                                                                               TensorList params,
                                                                               bool has_biases,
                                                                               int64_t num_layers,
                                                                               double dropout_p,
                                                                               bool train,
                                                                               bool bidirectional,
                                                                               bool batch_first) {
  // z_state: activated input, forget, cell and output gates, [layers * directions, seq_len, batch, 4 * hidden].
  // cell_state_fwd: hidden (0) and cell (1) with initial states, [layers * directions, 2, seq_len + 1, batch, hidden].
  auto x = (batch_first ? input.transpose(0, 1) : input).contiguous();
  const auto seq_len = x.size(0), batch = x.size(1), hidden = hx[0].size(2);
  const int64_t directions = bidirectional ? 2 : 1, stride = has_biases ? 4 : 2;
  const bool dropout = train && dropout_p > 0 && num_layers > 1;
  auto grad_y = grad_y_opt.value_or(Tensor());
  grad_y = grad_y.defined() ? (batch_first ? grad_y.transpose(0, 1) : grad_y)
                            : at::zeros({seq_len, batch, directions * hidden}, input.options());
  auto grad_h = grad_hy_opt.value_or(Tensor()), grad_c = grad_cy_opt.value_or(Tensor());
  grad_h = grad_h.defined() ? grad_h.clone(MemoryFormat::Contiguous) : at::zeros(hx[0].sizes(), input.options());
  grad_c = grad_c.defined() ? grad_c.clone(MemoryFormat::Contiguous) : at::zeros(hx[1].sizes(), input.options());
  std::vector<Tensor> grad_params(params.size());
  auto pso = lib.getPipelineStateForFunc("lstm_cell_backward_" + scalarToMetalTypeString(input));
  for (int64_t layer = num_layers - 1; layer >= 0; --layer) {
    const auto layer_input = layer == 0 ? x : layersOutputs[layer - 1];
    auto grad_input = at::zeros_like(layer_input);
    for (const auto direction : c10::irange(directions)) {
      const auto index = layer * directions + direction, param = index * stride;
      auto dy = grad_y.narrow(-1, direction * hidden, hidden);
      dy = direction ? dy.flip({0}).contiguous() : dy.contiguous();
      const auto h = cell_state_fwd[index][0], c = cell_state_fwd[index][1], gates = z_state[index];
      auto dh = grad_h[index], dc = grad_c[index];
      auto dg = at::empty(gates.sizes(), input.options());
      const auto recurrent_weight = params[param + 1].contiguous();
      for (int64_t step = seq_len - 1; step >= 0; --step) {
        lstm_cell(pso, dc, gates[step], c[step], c[step + 1], dy[step], dh, dc, dg[step]);
        lstm_mm(dh, dg[step], recurrent_weight);
      }
      auto dg_flat = dg.view({seq_len * batch, 4 * hidden});
      grad_params[param + 1] = dg_flat.t().mm(h.narrow(0, 0, seq_len).reshape({seq_len * batch, hidden}));
      if (direction) {
        dg_flat = dg.flip({0}).view({seq_len * batch, 4 * hidden});
      }
      grad_params[param] = dg_flat.t().mm(layer_input.reshape({seq_len * batch, layer_input.size(2)}));
      auto grad_input_flat = grad_input.view({seq_len * batch, layer_input.size(2)});
      // grad_input starts at zero, so the first direction doesn't need to read it.
      const auto beta = direction ? 1 : 0;
      at::addmm_out(grad_input_flat, grad_input_flat, dg_flat, params[param], beta);
      if (has_biases) {
        grad_params[param + 2] = dg_flat.sum(0);
        grad_params[param + 3] = grad_params[param + 2];
      }
    }
    if (dropout && layer > 0) {
      grad_input.mul_(layersOutputs[num_layers - 1 + layer - 1]);
    }
    grad_y = std::move(grad_input);
  }
  return {batch_first ? grad_y.transpose(0, 1).contiguous() : std::move(grad_y),
          {std::move(grad_h), std::move(grad_c)},
          std::move(grad_params)};
}
} // namespace at::native
