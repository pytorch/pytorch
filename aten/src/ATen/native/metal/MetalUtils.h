#include <vector>

namespace at {
namespace native {
namespace metal {

std::vector<uint16_t> fp32_to_fp16(const std::vector<float>& src);
std::vector<float> fp16_to_fp32(const std::vector<uint16_t>& src);
std::vector<float> NCHW_to_NC4(
    const float* src,
    const std::vector<int64_t>& sizes);
std::vector<float> NC4_to_NCHW(
    const float* src,
    const std::vector<int64_t>& sizes);

// When copying the result back to a CPU tensor, the memory format becomes NCHW.
// Thus,we compute the strides based on contiguous memory format.
static inline std::vector<int64_t> compute_strides(const std::vector<int64_t>& sizes) {
  const auto dim = sizes.size();
  std::vector<int64_t> strides(dim, 0);
  if (dim > 0) {
    const auto last_idx = dim - 1;
    strides[last_idx] = 1;
    for (int i = last_idx - 1; i >= 0; --i) {
      strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }
  }
  return strides;
}

} // namespace metal
} // namespace native
} // namespace at
