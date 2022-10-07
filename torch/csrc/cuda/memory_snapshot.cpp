#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/cuda/memory_snapshot.h>
#include <torch/csrc/jit/serialization/pickler.h>
namespace torch {
namespace cuda {

using c10::Dict;
using c10::IValue;
using torch::jit::Pickler;

using c10::cuda::CUDACachingAllocator::BlockInfo;
using c10::cuda::CUDACachingAllocator::History;
using c10::cuda::CUDACachingAllocator::SegmentInfo;

namespace {
std::string write_pickle(const IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    Pickler pickler(writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}
Dict<IValue, IValue> new_dict() {
  return Dict<IValue, IValue>(c10::AnyType::get(), c10::AnyType::get());
}
c10::List<IValue> new_list() {
  return List<IValue>(c10::AnyType::get());
}
} // namespace
void _record_memory_history(bool enabled, int64_t alloc_trace_max_entries) {
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled, nullptr, alloc_trace_max_entries, false);
}

std::string _memory_snapshot_pickled() {
  IValue device_s = "device";
  IValue address_s = "address";
  IValue total_size_s = "total_size";
  IValue allocated_size_s = "allocated_size";
  IValue active_size_s = "active_size";
  IValue stream_s = "stream";
  IValue segment_type_s = "segment_type";
  IValue large_s = "large";
  IValue small_s = "small";
  IValue size_s = "size";
  IValue state_s = "state";
  IValue active_allocated_s = "active_allocated";
  IValue active_pending_free_s = "active_pending_free";
  IValue inactive_s = "inactive";
  IValue addr_s = "addr";
  IValue real_size_s = "real_size";
  IValue filename_s = "filename";
  IValue name_s = "name";
  IValue line_s = "line";
  IValue frames_s = "frames";
  IValue history_s = "history";
  IValue blocks_s = "blocks";

  auto empty_frames = new_list();

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    auto segmentDict = new_dict();
    segmentDict.insert(device_s, segmentInfo.device);
    segmentDict.insert(address_s, segmentInfo.address);
    segmentDict.insert(total_size_s, segmentInfo.total_size);
    segmentDict.insert(allocated_size_s, segmentInfo.allocated_size);
    segmentDict.insert(active_size_s, segmentInfo.active_size);
    segmentDict.insert(stream_s, int64_t(segmentInfo.stream));
    segmentDict.insert(
        segment_type_s, (segmentInfo.is_large ? large_s : small_s));

    auto blocks = new_list();
    for (const auto& blockInfo : segmentInfo.blocks) {
      auto blockDict = new_dict();
      blockDict.insert(size_s, blockInfo.size);
      blockDict.insert(
          state_s,
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s)));
      if (blockInfo.history.size()) {
        auto history = new_list();
        for (const History& h : blockInfo.history) {
          auto history_entry = new_dict();
          history_entry.insert(addr_s, (int64_t)h.addr);
          history_entry.insert(real_size_s, (int64_t)h.real_size);
          if (h.context) {
            history_entry.insert(frames_s, empty_frames);
          }
          history.push_back(std::move(history_entry));
        }
        blockDict.insert(history_s, std::move(history));
      }
      blocks.push_back(blockDict);
    }
    segmentDict.insert(blocks_s, blocks);

    return segmentDict;
  };

  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();

  auto segments = new_list();
  for (const auto& segmentInfo : snapshot.segments) {
    segments.push_back(segmentInfoToDict(segmentInfo));
  }

  auto traces = new_list();
  IValue action_s = "action";
  IValue alloc_s = "alloc";
  IValue free_requested_s = "free_requested";
  IValue free_completed_s = "free_completed";
  IValue segment_alloc_s = "segment_alloc";
  IValue segment_free_s = "segment_free";
  IValue snapshot_s = "snapshot";
  IValue oom_s = "oom";
  IValue device_free_s = "device_free";

  using namespace c10::cuda::CUDACachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    auto trace = new_list();
    for (const auto& te : traceInfo) {
      auto trace_entry = new_dict();
      trace_entry.insert(action_s, action_to_str(te.action_));
      trace_entry.insert(
          TraceEntry::OOM == te.action_ ? device_free_s : addr_s, te.addr_);
      trace_entry.insert(size_s, (int64_t)te.size_);
      trace_entry.insert(stream_s, int64_t(te.stream_));
      trace.push_back(trace_entry);
    }
    traces.push_back(trace);
  }

  auto result = new_dict();
  result.insert("segments", segments);
  result.insert("device_traces", traces);
  return write_pickle(result);
}
} // namespace cuda
} // namespace torch
