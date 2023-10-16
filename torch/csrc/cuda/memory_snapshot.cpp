#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/cuda/memory_snapshot.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>

namespace torch::cuda {

using c10::Dict;
using c10::IValue;
using torch::jit::Pickler;

using c10::cuda::CUDACachingAllocator::BlockInfo;
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

std::vector<IValue> ivalue_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  // we dedup repeated to_symbolize objects to prevent
  // creating a bunch of duplicated frame objects
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  IValue line_s = "line";
  IValue name_s = "name";
  IValue filename_s = "filename";
  std::vector<IValue> all_frames;
  for (const auto& f : s.all_frames) {
    auto d = new_dict();
    d.insert(name_s, f.funcname);
    d.insert(filename_s, f.filename);
    d.insert(line_s, int64_t(f.lineno));
    all_frames.emplace_back(std::move(d));
  }

  std::vector<IValue> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    auto l = new_list();
    for (const auto& e : t) {
      l.push_back(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<IValue> result;
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

std::shared_ptr<c10::GatheredContext> gather() {
  return CapturedTraceback::gather(true, true, false);
}

std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
  return CapturedTraceback::gather(true, true, true);
}

CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  if (CapturedTraceback* sc = dynamic_cast<CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

} // namespace

void _record_memory_history(
    bool enabled,
    bool record_context,
    int64_t trace_alloc_max_entries,
    bool trace_alloc_record_context,
    bool record_cpp_context) {
  c10::cuda::CUDACachingAllocator::CreateContextFn recorder = gather;
  if (enabled && record_cpp_context) {
    recorder = gather_with_cpp;
    // warm up C++ stack unwinding
    unwind::unwind();
  }
  auto when = c10::cuda::CUDACachingAllocator::RecordContext::NEVER;
  if (trace_alloc_record_context) {
    when = c10::cuda::CUDACachingAllocator::RecordContext::ALLOC;
  } else if (record_context) {
    when = c10::cuda::CUDACachingAllocator::RecordContext::STATE;
  }
  at::globalContext().lazyInitCUDA();
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled, recorder, trace_alloc_max_entries, when);
}

static void checkOptionIn(
    const std::string& option,
    std::initializer_list<std::string> valid,
    const char* error) {
  TORCH_CHECK(
      valid.end() != std::find(valid.begin(), valid.end(), option), error);
}

void _record_memory_history(
    c10::optional<std::string> enabled,
    c10::optional<std::string> context,
    std::string stacks,
    size_t max_entries) {
  if (enabled) {
    checkOptionIn(
        *enabled,
        {"state", "all"},
        "expected state to be 'state', 'all', or None");
  }
  if (context) {
    checkOptionIn(
        *context,
        {"state", "alloc", "all"},
        "expected context to be 'state', 'alloc', 'all', or None");
  }
  checkOptionIn(
      stacks, {"python", "all"}, "expected stacks to be 'python', or 'all'");

  c10::cuda::CUDACachingAllocator::CreateContextFn recorder = gather;
  if (enabled && stacks == "all") {
    recorder = gather_with_cpp;
    // warm up C++ stack unwinding
    unwind::unwind();
  }
  max_entries = (enabled && *enabled == "all") ? max_entries : 1;
  auto when = c10::cuda::CUDACachingAllocator::RecordContext::NEVER;
  if (context) {
    if (context == "all") {
      when = c10::cuda::CUDACachingAllocator::RecordContext::ALL;
    } else if (context == "alloc") {
      when = c10::cuda::CUDACachingAllocator::RecordContext::ALLOC;
    } else if (context == "state") {
      when = c10::cuda::CUDACachingAllocator::RecordContext::STATE;
    }
  }
  at::globalContext().lazyInitCUDA();
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled.has_value(), recorder, max_entries, when);
}

std::string _memory_snapshot_pickled() {
  IValue device_s = "device";
  IValue address_s = "address";
  IValue total_size_s = "total_size";
  IValue allocated_size_s = "allocated_size";
  IValue active_size_s = "active_size";
  IValue requested_size_s = "requested_size";
  IValue stream_s = "stream";
  IValue segment_type_s = "segment_type";
  IValue segment_pool_id = "segment_pool_id";
  IValue large_s = "large";
  IValue small_s = "small";
  IValue size_s = "size";
  IValue state_s = "state";
  IValue active_allocated_s = "active_allocated";
  IValue active_pending_free_s = "active_pending_free";
  IValue inactive_s = "inactive";
  IValue addr_s = "addr";
  IValue filename_s = "filename";
  IValue name_s = "name";
  IValue line_s = "line";
  IValue frames_s = "frames";
  IValue blocks_s = "blocks";
  IValue is_expandable_s = "is_expandable";

  auto empty_frames = new_list();

  std::vector<CapturedTraceback*> frame_tracebacks;
  std::vector<Dict<IValue, IValue>> frame_dict;

  auto add_frame_key = [&](const c10::Dict<IValue, IValue>& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      frame_tracebacks.push_back(getFromContext(ctx));
      frame_dict.push_back(d);
    } else {
      d.insert(frames_s, empty_frames);
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    auto segmentDict = new_dict();
    segmentDict.insert(device_s, segmentInfo.device);
    segmentDict.insert(address_s, segmentInfo.address);
    segmentDict.insert(total_size_s, segmentInfo.total_size);
    segmentDict.insert(allocated_size_s, segmentInfo.allocated_size);
    segmentDict.insert(active_size_s, segmentInfo.active_size);
    segmentDict.insert(requested_size_s, segmentInfo.requested_size);
    segmentDict.insert(stream_s, int64_t(segmentInfo.stream));
    segmentDict.insert(
        segment_type_s, (segmentInfo.is_large ? large_s : small_s));
    segmentDict.insert(
        segment_pool_id,
        std::tuple<int64_t, int64_t>(segmentInfo.owner_private_pool_id));
    segmentDict.insert(is_expandable_s, segmentInfo.is_expandable);

    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    auto blocks = new_list();
    for (const auto& blockInfo : segmentInfo.blocks) {
      auto blockDict = new_dict();
      blockDict.insert(address_s, address);
      blockDict.insert(size_s, blockInfo.size);
      blockDict.insert(requested_size_s, blockInfo.requested_size);
      blockDict.insert(
          state_s,
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s)));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      address += blockInfo.size;
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
  IValue segment_map_s = "segment_map";
  IValue segment_unmap_s = "segment_unmap";
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
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
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
      if (te.context_) {
        auto sc = getFromContext(te.context_);
        frame_tracebacks.push_back(sc);
        frame_dict.push_back(trace_entry);
      }
      trace.push_back(trace_entry);
    }
    traces.push_back(trace);
  }

  auto result = new_dict();
  result.insert("segments", segments);
  result.insert("device_traces", traces);

  auto frames = ivalue_symbolize(frame_tracebacks);
  for (auto i : c10::irange(frames.size())) {
    frame_dict.at(i).insert(frames_s, frames.at(i));
  }

  return write_pickle(result);
}
} // namespace torch::cuda
