#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <cstddef>
#include <stack>

namespace torch {
namespace jit {
enum class Strategy {
  NAIVE = 0,
  LINEAR_SCAN,
  GREEDY_BY_SIZE_WITH_SMALLEST_GAP,
  GREEDY_BY_SIZE_WITH_FIRST_GAP,
  GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP,
  GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP,
  GREEDY_BY_BREADTH,
};

inline const char* toString(Strategy s) {
  switch (s) {
    case Strategy::NAIVE:
      return "NAIVE";
    case Strategy::LINEAR_SCAN:
      return "LINEAR_SCAN";
    case Strategy::GREEDY_BY_SIZE_WITH_SMALLEST_GAP:
      return "GREEDY_BY_SIZE_WITH_SMALLEST_GAP";
    case Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP:
      return "GREEDY_BY_SIZE_WITH_FIRST_GAP";
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP:
      return "GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP";
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP:
      return "GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP";
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
  size_t offset;
  size_t size;
} MemRegion;

inline std::ostream& operator<<(std::ostream& str, MemRegion reg) {
  return str << "{offset: " << reg.offset << ", size: " << reg.size << "}";
}

inline bool operator==(const MemRegion& lhs, const MemRegion& rhs) {
  return lhs.offset == rhs.offset && lhs.size == rhs.size;
}

struct regionSizeCmp {
  bool operator()(const MemRegion& reg1, const MemRegion& reg2) const {
    return reg1.size == reg2.size ? reg1.offset < reg2.offset
                                  : reg1.size < reg2.size;
  }
};

struct regionOffsetCmp {
  bool operator()(const MemRegion& reg1, const MemRegion& reg2) const {
    return reg1.offset == reg2.offset ? reg1.size < reg2.size
                                      : reg1.offset < reg2.offset;
  }
};

bool overlapMemRegion(const MemRegion& reg1, const MemRegion& reg2);

struct UniqueLiveRange {
  LiveRange lvr;
  std::string id;
};

bool overlapLiveRange(
    const UniqueLiveRange& ulvr1,
    const UniqueLiveRange& ulvr2);

inline std::ostream& operator<<(std::ostream& str, UniqueLiveRange rhs) {
  return str << "{id: " << rhs.id << ", lvr: " << rhs.lvr << "}";
}

inline bool operator==(const UniqueLiveRange lhs, const UniqueLiveRange rhs) {
  return lhs.lvr == rhs.lvr && lhs.id == rhs.id;
}

struct liveRangeStartCmp {
  bool operator()(const UniqueLiveRange& u1, const UniqueLiveRange& u2) const {
    return u1.lvr.begin == u2.lvr.begin
        ? (u1.lvr.end == u2.lvr.end ? u1.id < u2.id : u1.lvr.end < u2.lvr.end)
        : u1.lvr.begin < u2.lvr.begin;
  }
};

struct liveRangeEndCmp {
  bool operator()(const UniqueLiveRange& u1, const UniqueLiveRange& u2) const {
    return u1.lvr.end == u2.lvr.end
        ? (u1.lvr.begin == u2.lvr.begin ? u1.id < u2.id
                                        : u1.lvr.begin < u2.lvr.begin)
        : u1.lvr.end < u2.lvr.end;
  }
};

template <typename Value>
using SortedLiveRangeMap = std::map<UniqueLiveRange, Value, liveRangeStartCmp>;
struct TORCH_API MemAllocation {
  UniqueLiveRange ulvr;
  MemRegion reg;
};

inline std::ostream& operator<<(std::ostream& str, MemAllocation rhs) {
  return str << rhs.ulvr << ", " << rhs.reg;
}

inline bool operator==(const MemAllocation lhs, const MemAllocation rhs) {
  return lhs.ulvr == rhs.ulvr && lhs.reg == rhs.reg;
}

inline bool valid_add(size_t a, size_t b) {
#if defined(_MSC_VER)
  return a + b >= a;
#else
  size_t _carry = 0;
  return !__builtin_add_overflow(a, b, &_carry);
#endif
}

inline bool valid_sub(size_t a, size_t b) {
#if defined(_MSC_VER)
  return a >= b;
#else
  size_t _carry = 0;
  return !__builtin_sub_overflow(a, b, &_carry);
#endif
}

struct TORCH_API MemEvent {
  enum class EventType { Allocate = 0, Free };

  size_t time;
  std::string allocation_trace;
  std::string ptr_addr;
  size_t size;
  EventType type;
  c10::optional<FrameNodeId> frame_node_id;
  MemEvent(
      size_t t,
      std::string alloc_trace,
      std::string address,
      size_t s,
      EventType e,
      c10::optional<FrameNodeId> frame_nodeid = c10::nullopt)
      : time(t),
        allocation_trace(std::move(alloc_trace)),
        ptr_addr(std::move(address)),
        size(s),
        type(e),
        frame_node_id(std::move(frame_nodeid)) {}
};

inline const char* toString(MemEvent::EventType me) {
  switch (me) {
    case MemEvent::EventType::Free:
      return "Free";
    case MemEvent::EventType::Allocate:
      return "Allocate";
    default:
      return "unknown event type";
  }
}

inline std::ostream& operator<<(std::ostream& str, MemEvent::EventType rhs) {
  return str << toString(rhs);
}

inline std::ostream& operator<<(std::ostream& str, MemEvent rhs) {
  str << std::left << std::setfill(' ') << std::setw(15) << "type: " << rhs.type
      << "\n"
      << std::setw(15) << "t: " << rhs.time << "\n"
      << std::setw(15) << "size: " << rhs.size << "\n"
      << std::setw(15) << "ptr_addr: " << rhs.ptr_addr << "\n"
      << std::setw(15) << "alloc_trace: " << rhs.allocation_trace.substr(0, 40)
      << "...\n"
      << std::setw(15)
      << "frame_node_id has value: " << rhs.frame_node_id.has_value() << "\n";
  if (rhs.frame_node_id) {
    str << "pc: " << std::setw(15) << rhs.frame_node_id.value().pc << "\n"
        << "node_schema: " << std::setw(15)
        << rhs.frame_node_id.value().node_schema << "\n"
        << "node_header: " << std::setw(15)
        << rhs.frame_node_id.value().node_header << "\n"
        << "node addr: " << std::setw(15)
        << std::addressof(*rhs.frame_node_id.value().node) << "\n";
  }
  return str;
}

struct frame_node_id_hash {
  size_t operator()(FrameNodeId const& frame_node_id) const {
    return std::hash<size_t>()(frame_node_id.pc) ^
        (std::hash<std::string>()(frame_node_id.node_schema) << 1) ^
        (std::hash<std::string>()(frame_node_id.node_header) << 2);
  }
};

struct frameNodeIdCmp {
  size_t operator()(
      const std::pair<FrameNodeId, std::vector<UniqueLiveRange>>& f1,
      const std::pair<FrameNodeId, std::vector<UniqueLiveRange>>& f2) const {
    return f1.first.pc < f2.first.pc;
  }
};

inline bool operator==(const FrameNodeId& lhs, const FrameNodeId& rhs) {
  return lhs.pc == rhs.pc && lhs.node_schema == rhs.node_schema &&
      lhs.node_header == rhs.node_header;
}

c10::optional<size_t> computeStorageSize(const Value& value);

TORCH_API bool hasOutVariant(Node* node);

TORCH_API void planMemory(const std::shared_ptr<Graph>&, Strategy);
TORCH_API void planMemoryWithTracing(
    std::shared_ptr<Graph>& graph,
    Strategy strat,
    std::vector<MemEvent> mem_events,
    at::Device device_type);

} // namespace jit
} // namespace torch

namespace std {
template <>
struct hash<torch::jit::MemRegion> {
  size_t operator()(torch::jit::MemRegion const& reg) const {
    return std::hash<size_t>()(reg.offset) ^
        (std::hash<size_t>()(reg.size) << 1);
  }
};

template <>
struct hash<torch::jit::UniqueLiveRange> {
  size_t operator()(torch::jit::UniqueLiveRange const& ulvr) const {
    return std::hash<torch::jit::LiveRange>()(ulvr.lvr) ^
        (std::hash<string>()(ulvr.id));
  }
};

} // namespace std

namespace c10 {

struct C10_API MemoryPlanningAllocator final : at::Allocator {
  MemoryPlanningAllocator(at::DeviceType device_type);
  at::DataPtr allocate(size_t nbytes) const override;
  at::DeleterFnPtr raw_deleter() const override;
  void push_allocation(
      c10::Storage buffer,
      size_t size,
      size_t offset,
      at::DeviceType device);

 private:
  uint8_t allocator_priority_;
  at::DeviceType device_type_;
  c10::Allocator& orig_allocator_;
  mutable std::stack<std::pair<size_t, void*>> allocs_;
};

class C10_API WithProfileTracingAllocationsGuard;

struct C10_API MemoryTracingAllocator final : at::Allocator {
  MemoryTracingAllocator(at::DeviceType device_type)
      : allocator_priority_(c10::GetAllocatorPriority(device_type)),
        device_type_(device_type),
        orig_allocator_(*c10::GetAllocator(device_type)) {
    c10::SetAllocator(device_type, this, allocator_priority_);
  }

  at::DataPtr allocate(size_t nbytes) const override;

  uint8_t allocator_priority_;
  at::DeviceType device_type_;
  c10::Allocator& orig_allocator_;
  mutable std::vector<torch::jit::MemEvent> allocation_traces_;
  mutable std::map<void*, size_t> allocations_;
  friend WithProfileTracingAllocationsGuard;

 private:
};

class C10_API WithProfileTracingAllocationsGuard {
 public:
  WithProfileTracingAllocationsGuard(at::DeviceType device_type);
  std::vector<torch::jit::MemEvent> getAllocationTraces();
  ~WithProfileTracingAllocationsGuard() {
    c10::SetAllocator(
        device_type_, &tracer_.orig_allocator_, tracer_.allocator_priority_);
  }

 private:
  MemoryTracingAllocator tracer_;
  at::DeviceType device_type_;
};
} // namespace c10