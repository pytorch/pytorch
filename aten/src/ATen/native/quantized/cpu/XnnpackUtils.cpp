#ifdef USE_XNNPACK

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace xnnp_utils {

std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in) {
  const auto mem_format = in.suggest_memory_format();
  const auto& sizes = in.sizes();
  std::vector<size_t> ret(sizes.begin(), sizes.end());
  if (mem_format == c10::MemoryFormat::ChannelsLast) {
    // NCHW -> NHWC
    // 0123 -> 0231
    ret[1] = sizes[2]; /* H */
    ret[2] = sizes[3]; /* W */
    ret[3] = sizes[1]; /* C */
  } else if (mem_format == c10::MemoryFormat::ChannelsLast3d) {
    // NCDHW -> NDHWC
    // 01234 -> 02341
    ret[1] = sizes[2]; /* D */
    ret[2] = sizes[3]; /* H */
    ret[3] = sizes[4]; /* W */
    ret[4] = sizes[1]; /* C */
  }
  return ret;
}

template <typename PT>
void q8_copy_int8_weight_and_add_offset(const at::Tensor& in, at::Tensor& out) {
  using T = typename PT::underlying;
  static constexpr auto offset = std::is_same<T, uint8_t>::value ? 128 : 0;
  TORCH_CHECK(
      in.scalar_type() == c10::kQInt8,
      "q8_copy_int8_weight_and_add_offset: Expected input weight data type ",
      toString(c10::kQInt8),
      " but got ",
      toString(in.scalar_type()))
  const int8_t* in_ptr =
      reinterpret_cast<const int8_t*>(in.data_ptr<c10::qint8>());
  T* out_ptr = reinterpret_cast<T*>(out.data_ptr<PT>());

  for (const auto i : c10::irange(in.numel())) {
    out_ptr[i] = static_cast<T>(static_cast<int32_t>(in_ptr[i]) + offset);
  }
}

template void q8_copy_int8_weight_and_add_offset<c10::quint8>(
    const at::Tensor& in,
    at::Tensor& out);
template void q8_copy_int8_weight_and_add_offset<c10::qint8>(
    const at::Tensor& in,
    at::Tensor& out);

/*
 * Stolen from fbgemm_utils::ConvertConvWeightsToChannelLastTensor to avoid
 * dependence on USE_FBGEMM. Reorder weights to the format xnnpack expects.
 * TODO: add a 3d variant.
 */
template <>
Tensor convert_conv_weights_to_channel_last_tensor<2>(
    const at::Tensor& src,
    int groups,
    bool transpose) {
  return transpose ?
                   // 2D conv transpose weight transform
                   // IC OC/G KH KW -> G OC/G KH KW IC/G
      [&]() {
        auto ic_g_oc_g_hw_tensors = src.chunk(groups);
        for (auto& tensor : ic_g_oc_g_hw_tensors) {
          tensor = tensor.unsqueeze(0);
        }
        auto fused_tensor = at::cat(ic_g_oc_g_hw_tensors);
        set_quantizer_(fused_tensor, src.quantizer());
        return fused_tensor.permute({0, 2, 3, 4, 1})
            .contiguous(c10::MemoryFormat::Contiguous);
      }()
                   // 2d conv weight transform
                   : src.contiguous(c10::MemoryFormat::ChannelsLast);
}
} // namespace xnnp_utils
} // namespace native
} // namespace at

#endif // USE_XNNPACK
