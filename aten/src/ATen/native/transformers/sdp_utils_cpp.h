#pragma once
namespace sdp {
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2
};
} // namespace sdp