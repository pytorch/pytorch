#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/cuda/memory_snapshot.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/unwind/unwind.h>

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

struct StackContext : public c10::GatheredContext {
  std::vector<void*> cpp_frames;
  std::vector<jit::StackEntry> script_frames;

  static std::shared_ptr<StackContext> _gather(bool script, bool cpp) {
    auto r = std::make_shared<StackContext>();
    if (script) {
      r->script_frames = torch::jit::currentCallstack();
    }
    if (cpp) {
      r->cpp_frames = unwind::unwind();
    }
    return r;
  }
  static std::shared_ptr<c10::GatheredContext> gather() {
    return _gather(true, false);
  }
  static std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
    return _gather(true, true);
  }
};

StackContext* getFromContext(const std::shared_ptr<c10::GatheredContext>& x) {
  if (StackContext* sc = dynamic_cast<StackContext*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

void gatherFrames(
    const std::vector<std::pair<StackContext*, Dict<IValue, IValue>>>&
        to_gather) {
  IValue frames_s = "frames";
  IValue filename_s = "filename";
  IValue name_s = "name";
  IValue line_s = "line";

  std::unordered_map<void*, size_t> ip_to_frame_offset; // in all_cpp_frames
  std::vector<void*> all_cpp_ips;
  struct CPPFrame {
    enum Kind { JIT, REPORT } kind;
    Dict<IValue, IValue> frame;
  };
  std::vector<CPPFrame> all_cpp_frames;

  // dedup and collect any C++ frames that need symbols for
  for (const auto& e : to_gather) {
    for (void* f : e.first->cpp_frames) {
      if (!ip_to_frame_offset.count(f)) {
        ip_to_frame_offset[f] = all_cpp_ips.size();
        all_cpp_ips.push_back(f);
      }
    }
  }

  // gather symbol names for C++ frames
  if (all_cpp_ips.size() > 0) {
    auto all_frames = unwind::symbolize(all_cpp_ips);
    for (auto& f : all_frames) {
      auto frame = new_dict();
      frame.insert(filename_s, f.filename);
      frame.insert(name_s, f.funcname);
      frame.insert(line_s, IValue(int64_t(f.lineno)));
      CPPFrame::Kind kind = CPPFrame::REPORT;
      if (f.funcname.rfind("torch::jit::InterpreterStateImpl::run", 0) !=
          std::string::npos) {
        kind = CPPFrame::JIT;
      }
      all_cpp_frames.emplace_back(CPPFrame{kind, frame});
    }
  }

  std::unordered_map<StackContext*, c10::List<IValue>> cached_frames;
  for (const auto& e : to_gather) {
    auto sc = e.first;
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      auto frames = new_list();

      bool jit_appended = false;

      auto append_jit = [&]() {
        if (jit_appended) {
          return;
        }
        jit_appended = true;
        for (const auto& f : sc->script_frames) {
          auto frame = new_dict();
          frame.insert(name_s, f.filename);
          auto flc = f.range.file_line_col();
          if (flc) {
            std::string filename;
            size_t line;
            size_t col;
            std::tie(filename, line, col) = *flc;
            frame.insert(filename_s, filename);
            frame.insert(line_s, int64_t(line));
          } else {
            frame.insert(filename_s, "??");
            frame.insert(line_s, 0);
          }
          frames.push_back(std::move(frame));
        }
      };

      for (void* f : sc->cpp_frames) {
        const CPPFrame& wf = all_cpp_frames.at(ip_to_frame_offset.at(f));
        if (wf.kind == CPPFrame::JIT) {
          append_jit();
        }
        frames.push_back(wf.frame);
      }

      // add frames if we otherwise haven't seen the C++ frame indicating where
      // it should go
      append_jit();
      it = cached_frames.insert({sc, frames}).first;
    }
    e.second.insert(frames_s, it->second);
  }
}

} // namespace

void _record_memory_history(
    bool enabled,
    bool record_context,
    int64_t trace_alloc_max_entries,
    bool trace_alloc_record_context,
    bool record_cpp_context) {
  c10::cuda::CUDACachingAllocator::CreateContextFn recorder = nullptr;
  if (record_context) {
    if (record_cpp_context) {
      recorder = StackContext::gather_with_cpp;
    } else {
      recorder = StackContext::gather;
    }
  }
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled, recorder, trace_alloc_max_entries, trace_alloc_record_context);
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

  std::vector<std::pair<StackContext*, Dict<IValue, IValue>>> frames_to_gather;

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

    auto blocks = new_list();
    for (const auto& blockInfo : segmentInfo.blocks) {
      auto blockDict = new_dict();
      blockDict.insert(size_s, blockInfo.size);
      blockDict.insert(requested_size_s, blockInfo.requested_size);
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
            frames_to_gather.emplace_back(
                getFromContext(h.context), history_entry);
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
      if (te.context_) {
        auto sc = getFromContext(te.context_);
        frames_to_gather.emplace_back(sc, trace_entry);
      }
      trace.push_back(trace_entry);
    }
    traces.push_back(trace);
  }

  auto result = new_dict();
  result.insert("segments", segments);
  result.insert("device_traces", traces);

  gatherFrames(frames_to_gather);

  return write_pickle(result);
}
} // namespace cuda
} // namespace torch
