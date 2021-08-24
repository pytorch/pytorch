#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

#include <utility>

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

struct TORCH_API MemEvent {
  enum class EventType { Allocate = 0, Free };

  uint64_t time;
  std::string allocation_trace;
  std::string ptr_addr;
  uint64_t size;
  EventType type;
  c10::optional<FrameNodeId> frame_node_id;
  MemEvent(
      uint64_t t,
      std::string alloc_trace,
      std::string address,
      uint64_t s,
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
      << "..." << std::setw(15)
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

struct frame_node_id_hash {
  size_t operator()(FrameNodeId const& frame_node_id) const {
    return std::hash<size_t>()(frame_node_id.pc) ^
        (std::hash<std::string>()(frame_node_id.node_schema) << 1) ^
        (std::hash<std::string>()(frame_node_id.node_header) << 2);
  }
};

struct frame_node_id_cmp {
  size_t operator()(
      const std::pair<FrameNodeId, std::vector<LiveRange>>& f1,
      const std::pair<FrameNodeId, std::vector<LiveRange>>& f2) const {
    return f1.first.pc < f2.first.pc;
  }
};

inline bool operator==(const FrameNodeId& lhs, const FrameNodeId& rhs) {
  return lhs.pc == rhs.pc && lhs.node_schema == rhs.node_schema &&
      lhs.node_header == rhs.node_header;
}

c10::optional<uint64_t> computeStorageSize(const Value& value);

bool hasOutVariant(Node* node);

TORCH_API void planMemory(std::shared_ptr<Graph>&, Strategy);
TORCH_API void planMemoryWithTracing(
    std::shared_ptr<Graph>& graph,
    Strategy strat,
    std::vector<MemEvent> mem_events,
    c10::optional<at::Device> device_type);

} // namespace jit
} // namespace torch

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