#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <tuple>

namespace at::native::mkldnn {

// Ported from ideep::lstm_forward_inference::compute.
void lstm_forward_inference(
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    ideep::tensor& dst_layer,
    ideep::tensor& dst_iter,
    ideep::tensor& dst_iter_c,
    bool reverse,
    dnnl::prop_kind prop_kind);

// Ported from ideep::lstm_forward_inference::expected_weights_desc.
std::tuple<ideep::tensor::desc, ideep::tensor::desc>
lstm_forward_inference_expected_weights_desc(
    const dnnl::memory::dims& output_sizes,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    bool reverse);

// Ported from ideep::lstm_forward_training::prepare.
dnnl::lstm_forward::primitive_desc lstm_forward_training_prepare(
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    const ideep::tensor& dst_layer,
    const ideep::tensor& dst_iter,
    const ideep::tensor& dst_iter_c,
    bool reverse);

// Ported from ideep::lstm_forward_training::compute.
void lstm_forward_training(
    const dnnl::lstm_forward::primitive_desc& pd,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    const ideep::tensor& workspace,
    ideep::tensor& dst_layer,
    ideep::tensor& dst_iter,
    ideep::tensor& dst_iter_c);

// Ported from ideep::lstm_backward::compute.
void lstm_backward(
    const dnnl::lstm_forward::primitive_desc& forward_hints,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    const ideep::tensor& dst_layer,
    const ideep::tensor& dst_iter,
    const ideep::tensor& dst_iter_c,
    const ideep::tensor& diff_dst_layer,
    const ideep::tensor& diff_dst_iter,
    const ideep::tensor& diff_dst_iter_c,
    const ideep::tensor& workspace,
    ideep::tensor& diff_src_layer,
    ideep::tensor& diff_src_iter,
    ideep::tensor& diff_src_iter_c,
    ideep::tensor& diff_weights_layer,
    ideep::tensor& diff_weights_iter,
    ideep::tensor& diff_bias,
    bool reverse);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
