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

struct TORCH_API MemEvent {
  enum class EventType { Allocate = 0, Free };

  uint64_t time;
  std::string allocation_trace;
  std::string ptr_addr;
  std::string node_schema;
  std::string node_header;
  uint64_t size;
  EventType type;
  c10::optional<Node*> node;
  MemEvent(
      uint64_t t,
      std::string alloc_trace,
      std::string address,
      std::string node_schem,
      std::string node_head,
      uint64_t s,
      EventType e,
      c10::optional<Node*> nod = c10::nullopt)
      : time(t),
        allocation_trace(std::move(alloc_trace)),
        ptr_addr(std::move(address)),
        node_schema(std::move(node_schem)),
        node_header(std::move(node_head)),
        size(s),
        type(e),
        node(nod) {}
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
  return str << std::left << std::setfill(' ') << std::setw(15)
             << "type: " << rhs.type << "\n"
             << std::setw(15) << "t: " << rhs.time << "\n"
             << std::setw(15) << "size: " << rhs.size << "\n"
             << std::setw(15) << "ptr_addr: " << rhs.ptr_addr << "\n"
             << std::setw(15) << "node_schema: " << rhs.node_schema << "\n"
             << std::setw(15) << "node_header: " << rhs.node_header << "\n"
             << std::setw(15)
             << "alloc_trace: " << rhs.allocation_trace.substr(0, 20) << "...";
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

#define PRINT_CURR_ALLOC(x, y) \
  std::cout << __LINE__ << " " << x << " " << y << "\n";

} // namespace jit
} // namespace torch
