
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

at::Tensor qconv_prepack_onednn(
    at::Tensor weight,
    at::Tensor weight_scales,
    double input_scale,
    int64_t input_zero_point,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    std::optional<torch::List<int64_t>> input_shape) {

    int ndim = weight.ndimension() - 2;

    auto& engine = GpuEngineManager::Instance().get_engine();
    auto& stream = GpuStreamManager::Instance().get_stream();

    auto x_dims = input_shape.has_value() ? input_shape.value().vec() : dnnl::memory::dims();
    dnnl::memory::dims _stride = stride.vec();
    dnnl::memory::dims _padding_front_top_left = padding.vec();
    dnnl::memory::dims _padding_back_bottom_right = padding.vec();
    dnnl::memory::dims _dilation = compatible_dilation(dilation);    

    int src_mask, dst_mask = 0;
    int mask_weight = weight_scales.numel() > 1 ? 1 : 0;
    if (groups > 1 && weight_scales.numel() > 1)
      mask_weight = (2 ^ 0) | (2 ^ 1); // 2^0 (group) | 2^1 (output channel)

    dnnl::primitive_attr pattr;

    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    pattr.set_scales_mask(DNNL_ARG_SRC, src_mask);
    pattr.set_scales_mask(DNNL_ARG_DST, dst_mask); 
    pattr.set_scales_mask(DNNL_ARG_WEIGHTS, mask_weight);
   
    int ic = weight.size(1);
    int oc = weight.size(0);
    auto dummy_src_md = dnnl::memory::desc({1, ic, weight.size(2), weight.size(3)}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
    auto weight_md = dnnl::memory::desc(weight.sizes().vec(), get_onednn_dtype(weight), conv_weight_fmt(ndim, groups != 1, true /*is_channels_last*/));
    auto bias_md = dnnl::memory::desc();
    auto out_h = weight.size(2) + 2 * padding[0];
    auto out_w = weight.size(3) + 2 * padding[1];
    auto output_md = dnnl::memory::desc({1, oc, out_h, out_w}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::nhwc);
    
    auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward,
        dnnl::algorithm::convolution_direct,
        dummy_src_md,
        weight_md,
        bias_md,
        output_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right,
        pattr);
    
    auto weight_expected_md = conv_fwd_pd.weights_desc();
    Tensor weight_expected = at::empty(weight_expected_md.get_dims(), weight.options());;

    if(weight_md != weight_expected_md) {
        //reorder the weight
        auto weight_reorder_pd = dnnl::reorder::primitive_desc(
            engine, weight_md, engine, weight_expected_md);
        auto weight_reorder = dnnl::reorder(weight_reorder_pd);
        auto orig_src_mem = make_onednn_memory(weight_md, engine, weight.data_ptr());
        auto dst_mem = make_onednn_memory(weight_expected_md, engine, weight.data_ptr());
        weight_reorder.execute(stream, orig_src_mem, dst_mem);
        weight = weight_expected;
    }

    return weight;
 
       
}

} // namespace at::native::onednn
