#include <ATen/Context.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Exception.h>
#include <torch/csrc/cuda/memory_snapshot.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>

namespace torch::cuda {

using c10::Dict;
using c10::IValue;
using torch::jit::Pickler;

using c10::cuda::CUDACachingAllocator::SegmentInfo;

namespace {

class CallbackManager {
 public:
  // Constructor
  CallbackManager() = default;
  // Destructor
  ~CallbackManager() = default;
  // Methods to get and set the callback handles
  at::CallbackHandle getAnnotationHandle() const {
    return annotationHandle_;
  }
  void setAnnotationHandle(at::CallbackHandle handle) {
    annotationHandle_ = handle;
  }
  at::CallbackHandle getCompileContextHandle() const {
    return compileContextHandle_;
  }
  void setCompileContextHandle(at::CallbackHandle handle) {
    compileContextHandle_ = handle;
  }
  std::unique_lock<std::mutex> lockCallbackMutex() const {
    return std::unique_lock<std::mutex>(callbackMutex_);
  }

 private:
  mutable std::mutex callbackMutex_;
  at::CallbackHandle annotationHandle_{0};
  at::CallbackHandle compileContextHandle_{0};
};

CallbackManager callbackManager;

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
    py_unique_frames.emplace_back(std::move(l));
  }

  std::vector<IValue> result;
  result.reserve(to_symbolize.size());
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

#define ADD_CALLBACK(callbackType) at::add##callbackType##Callback
at::CallbackHandle _initRecordAnnotations(bool useGlobalCallback) {
  auto addCallback =
      useGlobalCallback ? ADD_CALLBACK(Global) : ADD_CALLBACK(ThreadLocal);
  return addCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            c10::cuda::CUDACachingAllocator::recordAnnotation(
                {{"name", fn.name()}, {"stage", "START"}});
            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            c10::cuda::CUDACachingAllocator::recordAnnotation(
                {{"name", fn.name()}, {"stage", "END"}});
          })
          .scopes({at::RecordScope::USER_SCOPE}));
}

at::CallbackHandle _initCompileContexts() {
  return at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            std::string functionName = fn.name();
            const std::string functionNamePrefix = "Torch-Compiled Region";
            if (functionName.compare(
                    0, functionNamePrefix.size(), functionNamePrefix) == 0) {
              c10::cuda::CUDACachingAllocator::pushCompileContext(functionName);
            }
            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            std::string functionName = fn.name();
            const std::string functionNamePrefix = "Torch-Compiled Region";
            if (functionName.compare(
                    0, functionNamePrefix.size(), functionNamePrefix) == 0) {
              c10::cuda::CUDACachingAllocator::popCompileContext();
            }
          })
          .scopes({at::RecordScope::FUNCTION}));
}

void setRecordFunctionCallbacks(
    bool enabled,
    bool compileContext,
    bool globalRecordAnnotations) {
  // Handle Callbacks under mutex
  auto lock = callbackManager.lockCallbackMutex();
  if (enabled) {
    if (callbackManager.getAnnotationHandle() == 0) {
      callbackManager.setAnnotationHandle(
          _initRecordAnnotations(globalRecordAnnotations));
    }
    if (compileContext && callbackManager.getCompileContextHandle() == 0) {
      callbackManager.setCompileContextHandle(_initCompileContexts());
    }
  } else {
    if (callbackManager.getAnnotationHandle() != 0) {
      at::removeCallback(callbackManager.getAnnotationHandle());
      callbackManager.setAnnotationHandle(0);
    }
    if (callbackManager.getCompileContextHandle() != 0) {
      at::removeCallback(callbackManager.getCompileContextHandle());
      callbackManager.setCompileContextHandle(0);
    }
  }
}

} // namespace

void _record_memory_history(
    bool enabled,
    bool record_context,
    int64_t trace_alloc_max_entries,
    bool trace_alloc_record_context,
    bool record_cpp_context,
    bool clearHistory,
    bool compileContext,
    bool globalRecordAnnotations,
    const std::vector<std::string>& skip_actions) {
  c10::cuda::CUDACachingAllocator::CreateContextFn recorder = gather;
  if (enabled && record_cpp_context &&
      (trace_alloc_record_context || record_context)) {
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
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);

  setRecordFunctionCallbacks(enabled, compileContext, globalRecordAnnotations);
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled,
      recorder,
      trace_alloc_max_entries,
      when,
      clearHistory,
      skip_actions);
}

static void checkOptionIn(
    const std::string& option,
    std::initializer_list<std::string> valid,
    const char* error) {
  TORCH_CHECK(
      valid.end() != std::find(valid.begin(), valid.end(), option), error);
}

void _record_memory_history(
    std::optional<std::string> enabled,
    std::optional<std::string> context,
    const std::string& stacks,
    size_t max_entries,
    bool clearHistory,
    bool compileContext,
    bool globalRecordAnnotations,
    const std::vector<std::string>& skip_actions) {
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
  if (enabled && context && stacks == "all") {
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
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  setRecordFunctionCallbacks(
      enabled.has_value(), compileContext, globalRecordAnnotations);
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled.has_value(),
      recorder,
      max_entries,
      when,
      clearHistory,
      skip_actions);
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
  IValue time_us_s = "time_us";
  IValue compile_contexts_s = "compile_context";
  IValue user_metadata_s = "user_metadata";

  auto empty_frames = new_list();

  std::vector<CapturedTraceback*> frame_tracebacks;
  std::vector<Dict<IValue, IValue>> frame_dict;

  auto add_frame_key = [&](const c10::Dict<IValue, IValue>& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      frame_tracebacks.push_back(getCapturedTracebackFromContext(ctx));
      frame_dict.push_back(d);
    } else {
      d.insert(frames_s, empty_frames);
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    auto segmentDict = new_dict();
    segmentDict.insert(device_s, segmentInfo.device);
    segmentDict.insert(address_s, static_cast<int64_t>(segmentInfo.address));
    segmentDict.insert(
        total_size_s, static_cast<int64_t>(segmentInfo.total_size));
    segmentDict.insert(
        allocated_size_s, static_cast<int64_t>(segmentInfo.allocated_size));
    segmentDict.insert(
        active_size_s, static_cast<int64_t>(segmentInfo.active_size));
    segmentDict.insert(
        requested_size_s, static_cast<int64_t>(segmentInfo.requested_size));
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
      blockDict.insert(address_s, static_cast<int64_t>(address));
      blockDict.insert(size_s, static_cast<int64_t>(blockInfo.size));
      blockDict.insert(
          requested_size_s, static_cast<int64_t>(blockInfo.requested_size));
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
    TORCH_CHECK(false, "unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    auto trace = new_list();
    for (const auto& te : traceInfo) {
      auto trace_entry = new_dict();
      trace_entry.insert(action_s, action_to_str(te.action_));
      trace_entry.insert(
          TraceEntry::OOM == te.action_ ? device_free_s : addr_s,
          static_cast<int64_t>(te.addr_));
      trace_entry.insert(size_s, (int64_t)te.size_);
      trace_entry.insert(stream_s, int64_t(te.stream_));
      trace_entry.insert(compile_contexts_s, te.compile_context_);
      trace_entry.insert(user_metadata_s, te.user_metadata_);
      if (te.context_) {
        auto sc = getCapturedTracebackFromContext(te.context_);
        frame_tracebacks.push_back(sc);
        frame_dict.push_back(trace_entry);
      }
      trace_entry.insert(time_us_s, te.time_.t_);
      trace.push_back(trace_entry);
    }
    traces.push_back(trace);
  }

  auto external_annotations = new_list();
  for (const auto& ae : snapshot.external_annotations) {
    auto annotation_entry = new_dict();
    for (const auto& md : ae.metadata_) {
      annotation_entry.insert((IValue)md.first, md.second);
    }
    annotation_entry.insert(device_s, ae.device_);
    annotation_entry.insert(time_us_s, ae.time_.t_);
    external_annotations.push_back(annotation_entry);
  }

  auto allocator_settings = new_dict();
  IValue last_allocator_settings_s = "PYTORCH_CUDA_ALLOC_CONF";
  IValue max_split_size_s = "max_split_size";
  IValue garbage_collection_threshold_s = "garbage_collection_threshold";
  IValue expandable_segments_s = "expandable_segments";
  IValue pinned_num_register_threads_s = "pinned_num_register_threads";
  IValue release_lock_on_malloc_s = "release_lock_on_cudamalloc";
  IValue pinned_use_host_register_s = "pinned_use_cuda_host_register";
  IValue roundup_power2_divisions_s = "roundup_power2_divisions";
  IValue graph_capture_record_stream_reuse_s =
      "graph_capture_record_stream_reuse";

  allocator_settings.insert(
      last_allocator_settings_s,
      snapshot.config_metadata.last_allocator_settings);
  allocator_settings.insert(
      max_split_size_s, int64_t(snapshot.config_metadata.max_split_size));
  allocator_settings.insert(
      garbage_collection_threshold_s,
      snapshot.config_metadata.garbage_collection_threshold);
  allocator_settings.insert(
      expandable_segments_s, snapshot.config_metadata.expandable_segments);
  allocator_settings.insert(
      pinned_num_register_threads_s,
      int64_t(snapshot.config_metadata.pinned_num_register_threads));
  allocator_settings.insert(
      release_lock_on_malloc_s,
      snapshot.config_metadata.release_lock_on_malloc);
  allocator_settings.insert(
      pinned_use_host_register_s,
      snapshot.config_metadata.pinned_use_host_register);
  allocator_settings.insert(
      graph_capture_record_stream_reuse_s,
      snapshot.config_metadata.graph_capture_record_stream_reuse);
  unsigned int roundup_key = 1;
  auto roundup_settings = new_dict();
  for (const auto& v : snapshot.config_metadata.roundup_power2_divisions) {
    IValue roundup_key_s = std::to_string(roundup_key);
    roundup_settings.insert(roundup_key_s, int64_t(v));
    roundup_key *= 2;
  }
  allocator_settings.insert(roundup_power2_divisions_s, roundup_settings);

  auto result = new_dict();
  result.insert("segments", segments);
  result.insert("device_traces", traces);
  result.insert("allocator_settings", allocator_settings);
  result.insert("external_annotations", external_annotations);

  auto frames = ivalue_symbolize(frame_tracebacks);
  for (auto i : c10::irange(frames.size())) {
    frame_dict.at(i).insert(frames_s, frames.at(i));
  }

  return write_pickle(result);
}
} // namespace torch::cuda
