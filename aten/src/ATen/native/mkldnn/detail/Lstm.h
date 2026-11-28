#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <tuple>

namespace at::native::mkldnn {

// Ported from ideep::lstm_forward_inference::compute.
void lstm_forward_inference(
    const MkldnnMemory& src_layer,
    const MkldnnMemory& src_iter,
    const MkldnnMemory& src_iter_c,
    const MkldnnMemory& weights_layer,
    const MkldnnMemory& weights_iter,
    const MkldnnMemory& bias,
    MkldnnMemory& dst_layer,
    MkldnnMemory& dst_iter,
    MkldnnMemory& dst_iter_c,
    bool reverse,
    dnnl::prop_kind prop_kind);

// Ported from ideep::lstm_forward_inference::expected_weights_desc.
std::tuple<MkldnnMemory::desc, MkldnnMemory::desc>
lstm_forward_inference_expected_weights_desc(
    const dnnl::memory::dims& output_sizes,
    const MkldnnMemory& src_layer,
    const MkldnnMemory& src_iter,
    const MkldnnMemory& src_iter_c,
    const MkldnnMemory& weights_layer,
    const MkldnnMemory& weights_iter,
    const MkldnnMemory& bias,
    bool reverse);

// Ported from ideep::lstm_forward_training::prepare.
dnnl::lstm_forward::primitive_desc lstm_forward_training_prepare(
    const MkldnnMemory& src_layer,
    const MkldnnMemory& src_iter,
    const MkldnnMemory& src_iter_c,
    const MkldnnMemory& weights_layer,
    const MkldnnMemory& weights_iter,
    const MkldnnMemory& bias,
    const MkldnnMemory& dst_layer,
    const MkldnnMemory& dst_iter,
    const MkldnnMemory& dst_iter_c,
    bool reverse);

// Ported from ideep::lstm_forward_training::compute.
void lstm_forward_training(
    const dnnl::lstm_forward::primitive_desc& pd,
    const MkldnnMemory& src_layer,
    const MkldnnMemory& src_iter,
    const MkldnnMemory& src_iter_c,
    const MkldnnMemory& weights_layer,
    const MkldnnMemory& weights_iter,
    const MkldnnMemory& bias,
    const MkldnnMemory& workspace,
    MkldnnMemory& dst_layer,
    MkldnnMemory& dst_iter,
    MkldnnMemory& dst_iter_c);

// Ported from ideep::lstm_backward::compute.
void lstm_backward(
    const dnnl::lstm_forward::primitive_desc& forward_hints,
    const MkldnnMemory& src_layer,
    const MkldnnMemory& src_iter,
    const MkldnnMemory& src_iter_c,
    const MkldnnMemory& weights_layer,
    const MkldnnMemory& weights_iter,
    const MkldnnMemory& bias,
    const MkldnnMemory& dst_layer,
    const MkldnnMemory& dst_iter,
    const MkldnnMemory& dst_iter_c,
    const MkldnnMemory& diff_dst_layer,
    const MkldnnMemory& diff_dst_iter,
    const MkldnnMemory& diff_dst_iter_c,
    const MkldnnMemory& workspace,
    MkldnnMemory& diff_src_layer,
    MkldnnMemory& diff_src_iter,
    MkldnnMemory& diff_src_iter_c,
    MkldnnMemory& diff_weights_layer,
    MkldnnMemory& diff_weights_iter,
    MkldnnMemory& diff_bias,
    bool reverse);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
