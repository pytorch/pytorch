#include <ATen/ATen.h>

#ifdef USE_XNNPACK
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
} // namespace xnnp_utils
} // namespace native
} // namespace at

#endif  // USE_XNNPACK
