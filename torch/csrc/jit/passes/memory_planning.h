#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {
enum class Strategy {
  NAIVE = 0,
  LINEAR_SCAN,
  GREEDY_BY_SIZE,
  GREEDY_BY_SIZE_WITH_FIRST_GAP,
  GREEDY_BY_LONGEST_AND_SIZE,
  GREEDY_BY_BREADTH,
};

inline const char* toString(Strategy s) {
  switch (s) {
    case Strategy::NAIVE:
      return "NAIVE";
    case Strategy::LINEAR_SCAN:
      return "LINEAR_SCAN";
    case Strategy::GREEDY_BY_SIZE:
      return "GREEDY_BY_SIZE";
    case Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP:
      return "GREEDY_BY_SIZE_WITH_FIRST_GAP";
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE:
      return "GREEDY_BY_LONGEST_AND_SIZE";
    case Strategy::GREEDY_BY_BREADTH:
      return "GREEDY_BY_BREADTH";
    default:
      return "UNKNOWN STRATEGY";
  }
}

inline std::ostream& operator<<(std::ostream& str, Strategy rhs) {
  return str << toString(rhs);
}

typedef struct MemRegion {
  int64_t offset;
  int64_t size;
} Region;

inline std::ostream& operator<<(std::ostream& str, MemRegion reg) {
  return str << "{offset: " << reg.offset << ", size: " << reg.size << "}";
}

inline bool operator==(const MemRegion& lhs, const MemRegion& rhs) {
  return lhs.offset == rhs.offset && lhs.size == rhs.size;
}

struct region_size_cmp {
  bool operator()(MemRegion const& reg1, MemRegion const& reg2) const {
    return reg1.size == reg2.size ? reg1.offset < reg2.offset
                                  : reg1.size < reg2.size;
  }
};

struct region_offset_cmp {
  bool operator()(const MemRegion& reg1, const MemRegion& reg2) const {
    return reg1.offset == reg2.offset ? reg1.size < reg2.size
                                      : reg1.offset < reg2.offset;
  }
};

bool intersectLiveRange(LiveRange lvr1, LiveRange lvr2);

bool intersectMemRegion(MemRegion reg1, MemRegion reg2);

int intersectArea(int64_t a, int64_t b, int64_t c, int64_t d);

struct TORCH_API MemAllocation {
  LiveRange lvr;
  MemRegion reg;
};

inline std::ostream& operator<<(std::ostream& str, MemAllocation rhs) {
  return str << rhs.lvr << ", " << rhs.reg;
}

inline bool operator==(MemAllocation lhs, MemAllocation rhs) {
  return lhs.lvr == rhs.lvr && lhs.reg == rhs.reg;
}

c10::optional<int64_t> computeStorageSize(const Value& value);

bool hasOutVariant(Node* node);

TORCH_API void planMemory(std::shared_ptr<Graph>&, Strategy);

} // namespace jit
} // namespace torch
