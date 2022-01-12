#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>

#ifdef USE_XNNPACK
namespace at {
namespace native {
namespace xnnp_utils {

at::Tensor reorder_weights_for_transpose_qconv(const at::Tensor& weight_nhwc,
    int num_groups) {

  TORCH_CHECK(weight_nhwc.size(0) % num_groups == 0,
    "The number of groups cannot be satisfied by the provided weight tensor.");

  at::Tensor reordered = at::_empty_affine_quantized(
      weight_nhwc.sizes(),
      at::device(c10::kCPU).dtype(c10::kQUInt8).memory_format(c10::MemoryFormat::ChannelsLast),
      weight_nhwc.q_scale(),
      weight_nhwc.q_zero_point(),
      c10::nullopt);

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int input_channels_per_group = weight_nhwc.size(0) / num_groups;
  int output_channels_per_group = weight_nhwc.size(1);
  int kernel_width = weight_nhwc.size(3);
  int kernel_height = weight_nhwc.size(2);

  int o_offset = 1;
  int h_offset = (output_channels_per_group);
  int w_offset = (output_channels_per_group)*(kernel_height);
  int i_offset = (output_channels_per_group)*(kernel_height)*(kernel_width);
  int g_offset = (output_channels_per_group)*(kernel_height)*(kernel_width)*(input_channels_per_group);

  uint8_t* out_ptr = reinterpret_cast<uint8_t*>(reordered.data_ptr<c10::quint8>());
  int8_t* in_ptr = reinterpret_cast<int8_t*>(weight_nhwc.data_ptr<c10::qint8>());

  int out_index = 0;
  for (int g = 0; g < num_groups; g++) {
    for (int o = 0; o < output_channels_per_group; o++) {
      for (int w = 0; w < kernel_width; w++) {
        for (int h = 0; h < kernel_height; h++) {
          for (int i = 0; i < input_channels_per_group; i++) {
            int in_index = (g*g_offset) + (i*i_offset) + (h*h_offset) + (w*w_offset) + (o*o_offset);
            out_ptr[out_index] = static_cast<uint8_t>(static_cast<int32_t>(in_ptr[in_index]) + 128);
            out_index++;
          }
        }
      }
    }
  }
  return reordered;
}

}  // xnnp_utils
}  // native
}  // at

#endif  // USE_XNNPACK
