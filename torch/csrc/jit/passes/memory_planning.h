#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {
enum class Strategy {
  NAIVE = 0,
  GREEDY_BY_SIZE,
  GREEDY_BY_BREADTH,
  LINEAR_SCAN,
};

inline const char* toString(Strategy s) {
  switch (s) {
    case Strategy::NAIVE:
      return "NAIVE";
    case Strategy::GREEDY_BY_SIZE:
      return "GREEDY_BY_SIZE";
    case Strategy::GREEDY_BY_BREADTH:
      return "GREEDY_BY_BREADTH";
    case Strategy::LINEAR_SCAN:
      return "LINEAR_SCAN";
    default:
      return "UNKNOWN STRATEGY";
  }
}

inline std::ostream& operator<<(std::ostream& str, Strategy rhs) {
  return str << toString(rhs);
}

typedef struct Region {
  uint64_t offset;
  uint64_t size;
} Region;

inline std::ostream& operator<<(std::ostream& str, Region reg) {
  return str << "{offset: " << reg.offset << ", size: " << reg.size << "}";
}

inline bool operator==(const LiveRange& lhs, const LiveRange& rhs) {
  return lhs.begin == rhs.begin && lhs.end == rhs.end;
}
inline bool operator!=(const LiveRange& lhs, const LiveRange& rhs) {
  return !(lhs == rhs);
}

struct live_range_start_cmp {
  bool operator()(LiveRange const& range1, LiveRange const& range2) const {
    return range1.begin < range2.begin;
  }
};

struct live_range_end_cmp {
  bool operator()(LiveRange const& range1, LiveRange const& range2) const {
    return range1.end < range2.end;
  }
};

struct live_range_hash {
  size_t operator()(LiveRange const& range) const {
    return std::hash<size_t>()(range.begin) ^
        (std::hash<size_t>()(range.end) << 1);
  }
};

struct region_size_cmp {
  bool operator()(Region const& reg1, Region const& reg2) const {
    return reg1.size < reg2.size;
  }
};

struct region_offset_cmp {
  bool operator()(const Region& reg1, const Region& reg2) const {
    return reg1.offset < reg2.offset;
  }
};

c10::optional<uint64_t> computeStorageSize(const Value& value);

bool hasOutVariant(Node* node);

TORCH_API void planMemory(std::shared_ptr<Graph>&, Strategy);

#define PRINT_CURR_ALLOC(x, y) \
  std::cout << __LINE__ << " " << x << " " << y << "\n";

} // namespace jit
} // namespace torch
