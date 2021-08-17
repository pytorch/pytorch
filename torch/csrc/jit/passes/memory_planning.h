#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {
enum class Strategy {
  NAIVE,
  GREEDY_BY_SIZE,
  GREEDY_BY_BREADTH,
  LINEAR_SCAN,
};

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

struct live_range_start_comp {
  bool operator()(LiveRange const& range1, LiveRange const& range2)
      const {
    return range1.begin < range2.begin;
  }
};

struct live_range_end_comp {
  bool operator()(LiveRange const& range1, LiveRange const& range2)
      const {
    return range1.end < range2.end;
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
