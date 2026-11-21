#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <tuple>

namespace at::native::mkldnn {

// Ported from ideep::lstm_forward_inference::compute.
void lstm_forward_inference(
    const ITensor& src_layer,
    const ITensor& src_iter,
    const ITensor& src_iter_c,
    const ITensor& weights_layer,
    const ITensor& weights_iter,
    const ITensor& bias,
    ITensor& dst_layer,
    ITensor& dst_iter,
    ITensor& dst_iter_c,
    bool reverse,
    dnnl::prop_kind prop_kind);

// Ported from ideep::lstm_forward_inference::expected_weights_desc.
std::tuple<ITensor::desc, ITensor::desc>
lstm_forward_inference_expected_weights_desc(
    const dnnl::memory::dims& output_sizes,
    const ITensor& src_layer,
    const ITensor& src_iter,
    const ITensor& src_iter_c,
    const ITensor& weights_layer,
    const ITensor& weights_iter,
    const ITensor& bias,
    bool reverse);

// Ported from ideep::lstm_forward_training::prepare.
dnnl::lstm_forward::primitive_desc lstm_forward_training_prepare(
    const ITensor& src_layer,
    const ITensor& src_iter,
    const ITensor& src_iter_c,
    const ITensor& weights_layer,
    const ITensor& weights_iter,
    const ITensor& bias,
    const ITensor& dst_layer,
    const ITensor& dst_iter,
    const ITensor& dst_iter_c,
    bool reverse);

// Ported from ideep::lstm_forward_training::compute.
void lstm_forward_training(
    const dnnl::lstm_forward::primitive_desc& pd,
    const ITensor& src_layer,
    const ITensor& src_iter,
    const ITensor& src_iter_c,
    const ITensor& weights_layer,
    const ITensor& weights_iter,
    const ITensor& bias,
    const ITensor& workspace,
    ITensor& dst_layer,
    ITensor& dst_iter,
    ITensor& dst_iter_c);

// Ported from ideep::lstm_backward::compute.
void lstm_backward(
    const dnnl::lstm_forward::primitive_desc& forward_hints,
    const ITensor& src_layer,
    const ITensor& src_iter,
    const ITensor& src_iter_c,
    const ITensor& weights_layer,
    const ITensor& weights_iter,
    const ITensor& bias,
    const ITensor& dst_layer,
    const ITensor& dst_iter,
    const ITensor& dst_iter_c,
    const ITensor& diff_dst_layer,
    const ITensor& diff_dst_iter,
    const ITensor& diff_dst_iter_c,
    const ITensor& workspace,
    ITensor& diff_src_layer,
    ITensor& diff_src_iter,
    ITensor& diff_src_iter_c,
    ITensor& diff_weights_layer,
    ITensor& diff_weights_iter,
    ITensor& diff_bias,
    bool reverse);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
