#pragma once

namespace at {

struct PhiloxXpuState {
  PhiloxXpuState() = default;
  PhiloxXpuState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // for graph capture
  PhiloxXpuState(
      int64_t* seed,
      int64_t* offset_extragraph,
      uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

namespace xpu::philox {
inline std::tuple<uint64_t, uint64_t> unpack(at::PhiloxXpuState arg) {
  if (arg.captured_) {
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
  }
}

} // namespace xpu::philox
} // namespace at
