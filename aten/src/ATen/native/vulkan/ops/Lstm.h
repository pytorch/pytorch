#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

//   packed
//     std::vector<c10::intrusive_ptr<VulkanOpContext>> linear_op_contexts;  //
//     {{ op context for b_ii, w_ii, op context for b_hi, w_hi,
//                                                                           //
//                                                                           op
//                                                                           context
//                                                                           for
//                                                                           b_if,
//                                                                           w_if,
//                                                                           op
//                                                                           context
//                                                                           for
//                                                                           b_hf,
//                                                                           w_hf,
//                                                                           //
//                                                                           op
//                                                                           context
//                                                                           for
//                                                                           b_ig,
//                                                                           w_ig,
//                                                                           op
//                                                                           context
//                                                                           for
//                                                                           b_hg,
//                                                                           w_hg,
//                                                                           //
//                                                                           op
//                                                                           context
//                                                                           for
//                                                                           b_io,
//                                                                           w_io,
//                                                                           op
//                                                                           context
//                                                                           for
//                                                                           b_ho,
//                                                                           w_ho,},
//                                                                           ...}
//     bool has_biases{};
//     int64_t num_layers{};
//     double dropout{};
//     bool train{};
//     bool bidirectional{};
//     bool batch_first{};

//   unpacked
//     std::vector<Tensor> params_cpu      // weights/biases (cpu)
//     bool has_biases
//     int64_t num_layers
//     double dropout
//     bool train
//     bool bidirectional
//     bool batch_first

c10::intrusive_ptr<VulkanOpContext> create_lstm_context(
    std::vector<Tensor>&& params_cpu, // weights/biases (cpu)
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first);

std::tuple<Tensor, Tensor, Tensor> run_lstm_context(
    const Tensor& input_vk, // input sequence (vulkan)
    const Tensor& hx_vk, // initial hidden state (vulkan)
    const Tensor& cx_vk, // initial cell state (vulkan)
    const c10::intrusive_ptr<VulkanOpContext>& vulkan_context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
