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


} // namespace metal
} // namespace native
} // namespace at
