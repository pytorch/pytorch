#include <torch/csrc/profiler/collection.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <type_traits>

#include <fmt/format.h>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

#include <ATen/Context.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/kineto_shim.h>

namespace torch {
namespace profiler {
namespace impl {
using trace_ptr_t =
    std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>;

void InputOutputEncoder::push(c10::ArrayRef<const c10::IValue> values) {
  for (const auto& value : values) {
    if (value.isTensor()) {
      push(value.toTensor());
    } else if (value.isScalar()) {
      tags_.emplace_back(Tag::Scalar);
    } else if (value.isTensorList()) {
      tags_.emplace_back(Tag::TensorListBegin);
      // TODO: Skip TensorList for now.
      tags_.emplace_back(Tag::TERMINATOR);
    } else {
      tags_.emplace_back(Tag::Other);
    }
  }
  tags_.emplace_back(Tag::TERMINATOR);
}

void InputOutputEncoder::push(const at::Tensor& t) {
  if (t.defined()) {
    tags_.emplace_back(Tag::Tensor);
    const auto& sizes = t.sizes();
    const auto dim = sizes.size();
    TORCH_CHECK(
        dim <= std::numeric_limits<uint32_t>::max(),
        "Cannot profile Tensors of size > uint32 max. Got dim: ",
        dim);

    tensor_metadata_.emplace_back(
        /*ptr_=*/(void*)t.unsafeGetTensorImpl(),
        /*dtype_=*/t.scalar_type(),
        /*dim_=*/(uint32_t)dim,
        /*layout_=*/t.layout());

    for (const auto i : sizes) {
      tensor_sizes_.emplace_back(i);
    }
  } else {
    tags_.emplace_back(Tag::UndefinedTensor);
  }
}

// This is a custom-iterator-like getter to obtain input shapes and dtypes.
auto InputOutputEncoder::getNextShapesAndDtypes() {
  return [this,
          tag_it = tags_.begin(),
          tensor_metadata_it = tensor_metadata_.begin(),
          tensor_size_it = tensor_sizes_.begin()]() mutable {
    struct Inputs out;
    bool terminate = false;
    while (!terminate && tag_it != tags_.end()) {
      out.shapes_.emplace_back();
      switch (*tag_it) {
        case Tag::Tensor: {
          const auto& md = *tensor_metadata_it++;
          for (const auto _ : c10::irange(md.dim_)) {
            (void)_; // Suppress unused variable warning
            out.shapes_.back().push_back(*tensor_size_it++);
          }
          out.tensor_metadata_.emplace_back(md);
          out.dtypes_.emplace_back(scalarTypeToTypeMeta(md.dtype_).name());
        } break;

        case Tag::TensorListBegin:
          while (*(++tag_it) != Tag::TERMINATOR) {
            // TODO: Skip TensorLists for now.
          }
          out.dtypes_.emplace_back("TensorList");
          out.tensor_metadata_.emplace_back();
          break;

        case Tag::Scalar:
          out.dtypes_.emplace_back("Scalar");
          out.tensor_metadata_.emplace_back();
          break;

        case Tag::UndefinedTensor:
        case Tag::Other:
          out.dtypes_.emplace_back();
          out.tensor_metadata_.emplace_back();
          break;

        case Tag::TERMINATOR:
          // This marks the end of this op.
          out.shapes_.pop_back();
          terminate = true;
          break;

        default:
          break;
      }
      ++tag_it;
    }
    return out;
  };
}

void InputOutputEncoder::clear() {
  tags_.clear();
  tensor_metadata_.clear();
  tensor_sizes_.clear();
}

namespace {
// See `RecordQueue::getSubqueue()` for an overview of this cache.
struct SubQueueThreadCache {
  uint32_t key_;
  ThreadLocalSubqueue* ref_;
};

// The astute observer will note that this leaves a dangling reference; nothing
// in the teardown of `RecordQueue` or `ThreadLocalSubqueue` clears this value.
// (And the raw pointer in `SubQueueThreadCache` will not extend the lifetime
// of `*ref_`.) This is safe, however, because `getSubqueue` will check
// `sub_queue_cache_.key_` before attempting to access `ref_`, and if `key_`
// does not match the RecordQueue's *unique* `id_` it will evict
// `sub_queue_cache_` and fall back to a different mechanism.
std::atomic<uint32_t> queue_id_{0};
thread_local SubQueueThreadCache sub_queue_cache_{0, nullptr};
} // namespace

namespace python_tracer {
namespace {
GetFn get_fn;

struct NoOpPythonTracer : public PythonTracerBase {
  static NoOpPythonTracer& singleton() {
    static NoOpPythonTracer singleton_;
    return singleton_;
  }
  void start(RecordQueue*) override {}
  void stop() override {}
  void clear() override {}
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<time_t(approx_time_t)>,
      std::vector<CompressedEvent>&) override {
    return {};
  }
  ~NoOpPythonTracer() = default;
};
} // namespace

void registerTracer(GetFn get_tracer) {
  get_fn = get_tracer;
}

PythonTracerBase& PythonTracerBase::get() {
  if (get_fn == nullptr) {
    return NoOpPythonTracer::singleton();
  }
  return get_fn();
}
} // namespace python_tracer

#define OUT_T(method_name) decltype(std::declval<Result>().method_name())
#define DEFINE_VISITOR(                                                  \
    method_name,                                                         \
    torch_op_field,                                                      \
    backend_field,                                                       \
    allocation_field,                                                    \
    oom_field,                                                           \
    py_field,                                                            \
    py_c_field,                                                          \
    kineto_field)                                                        \
  OUT_T(method_name) Result::method_name() const {                       \
    using out_t = OUT_T(method_name);                                    \
    return c10::visit(                                                   \
        c10::overloaded(                                                 \
            [&](const ExtraFields<EventType::TorchOp>& e) -> out_t {     \
              (void)e;                                                   \
              return torch_op_field;                                     \
            },                                                           \
            [&](const ExtraFields<EventType::Backend>& e) -> out_t {     \
              (void)e;                                                   \
              return backend_field;                                      \
            },                                                           \
            [&](const ExtraFields<EventType::Allocation>& e) -> out_t {  \
              (void)e;                                                   \
              return allocation_field;                                   \
            },                                                           \
            [&](const ExtraFields<EventType::OutOfMemory>& e) -> out_t { \
              (void)e;                                                   \
              return oom_field;                                          \
            },                                                           \
            [&](const ExtraFields<EventType::PyCall>& e) -> out_t {      \
              (void)e;                                                   \
              return py_field;                                           \
            },                                                           \
            [&](const ExtraFields<EventType::PyCCall>& e) -> out_t {     \
              (void)e;                                                   \
              return py_c_field;                                         \
            },                                                           \
            [&](const ExtraFields<EventType::Kineto>& e) -> out_t {      \
              (void)e;                                                   \
              return kineto_field;                                       \
            }),                                                          \
        extra_fields_);                                                  \
  }

std::string toString(const ExtraFields<EventType::PyCall>& e) {
  if (e.module_.has_value()) {
    return fmt::format(
        "nn.Module: {}_{}", e.module_->cls_name_.str(), e.module_->id_);
  }
  return fmt::format(
      "{}({}): {}",
      e.callsite_.filename_.str(),
      e.callsite_.line_no_,
      e.callsite_.funcname_.str());
}

namespace {
auto scopeToType(at::RecordScope scope) {
  return scope == at::RecordScope::USER_SCOPE
      ? libkineto::ActivityType::USER_ANNOTATION
      : libkineto::ActivityType::CPU_OP;
}

auto torchOpEndNS(
    const ExtraFields<EventType::TorchOp>& e,
    const bool finished,
    const std::weak_ptr<Result>& parent) {
  if (finished && e.end_time_ns_ == std::numeric_limits<time_t>::min()) {
    auto p = parent.lock();
    if (p) {
      return p->endTimeNS();
    }
  }
  return e.end_time_ns_;
}

auto kinetoEventCorrelationID(
    const ExtraFields<EventType::Kineto>& e,
    const std::weak_ptr<Result>& parent) {
  if (e.correlation_id_) {
    return e.correlation_id_;
  }
  auto p = parent.lock();
  return p ? p->correlationID() : 0;
}
} // namespace

DEFINE_VISITOR(
    name,
    e.name_,
    e.name_,
    "[memory]",
    "[OutOfMemory]",
    toString(e),
    e.function_name_.str(),
    e.name_);
DEFINE_VISITOR(
    kinetoType,
    scopeToType(e.scope_),
    scopeToType(e.scope_),
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::PYTHON_FUNCTION,
    libkineto::ActivityType::PYTHON_FUNCTION,
    e.activity_type_);
DEFINE_VISITOR(
    correlationID,
    e.correlation_id_,
    0,
    0,
    0,
    0,
    0,
    kinetoEventCorrelationID(e, parent_));
DEFINE_VISITOR(
    endTimeNS,
    torchOpEndNS(e, finished_, parent_),
    e.end_time_us_ * 1000,
    start_time_ns_,
    start_time_ns_,
    e.end_time_ns_,
    e.end_time_ns_,
    start_time_ns_ + e.duration_us_ * 1000);
DEFINE_VISITOR(
    endTID,
    e.end_tid_,
    start_tid_,
    start_tid_,
    start_tid_,
    start_tid_,
    start_tid_,
    start_tid_);
DEFINE_VISITOR(
    deviceType,
    c10::DeviceType::CPU,
    c10::DeviceType::CPU,
    e.device_type_,
    e.device_type_,
    c10::DeviceType::CPU,
    c10::DeviceType::CPU,
    torch::autograd::profiler::deviceTypeFromActivity(e.activity_type_));
#undef DEFINE_VISITOR
#undef OUT_T

template <typename T, size_t ChunkSize>
ThreadLocalSubqueue::EventBlock<T, ChunkSize>::EventBlock() {
  static std::atomic<uint64_t> counter_{0};
  id_start_ = 1 + ChunkSize * counter_++;
}
template <class... Args>
std::pair<KinetoObserverContext::Event*, uint64_t> ThreadLocalSubqueue::OpList::
    emplace_back(Args&&... args) {
  maybe_grow();
  *next_ = {std::forward<Args>(args)...};
  auto corr_id = buffer_last_->correlation_id(next_);
  return {next_++, corr_id};
}
uint64_t ThreadLocalSubqueue::OpList::correlationID(const OpList::Iterator& e) {
  return e.address().first->correlation_id(&*e);
}

ThreadLocalSubqueue::ThreadLocalSubqueue(
    const uint64_t tid,
    const ProfilerConfig& config)
    : tid_{tid}, config_{config}, kineto_info_{kineto::kineto_ids()} {
  torch::profiler::impl::kineto::recordThreadInfo();
}

std::unique_ptr<KinetoObserverContext> ThreadLocalSubqueue::begin_op(
    const at::RecordFunction& fn) {
  KinetoObserverContext::Event* event;
  uint64_t corr_id;
  std::tie(event, corr_id) = op_events_.emplace_back(
      fn.seqNr(),
      fn.forwardThreadId(),
      fn.scope(),
      fn.isAsync(),
      fn.debugHandle(),
      fn.name());
  if (config_.report_input_shapes) {
    inputs_outputs_.push(fn.inputs());
  }
  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    torch::profiler::impl::kineto::pushUserCorrelationId(corr_id);
  } else {
    torch::profiler::impl::kineto::pushCorrelationId(corr_id);
  }

#if !defined BUILD_LITE_INTERPRETER && !defined C10_MOBILE
  // backward nodes source range corresponds to the forward node
  // TODO: consider using C++ stack trace
  if (config_.with_stack && fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    auto cs = torch::profiler::impl::prepareCallstack(jit::currentCallstack());
    jit_stack_.emplace_back(callstackStr(cs));
  }
  if (config_.with_modules &&
      fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    jit_modules_.emplace_back(jit::currentModuleHierarchy());
  }
#endif
  if (config_.with_flops) {
    extra_args_.emplace_back(torch::profiler::impl::saveExtraArgs(fn));
  }

  auto out = std::make_unique<KinetoObserverContext>(event);

  if (config_.state == ProfilerState::KINETO_GPU_FALLBACK) {
    try {
      out->fallback_ = gpu_fallback_.emplace_back();
      torch::profiler::impl::cudaStubs()->record(
          nullptr, &out->fallback_->cuda_event_start_, nullptr);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to record CUDA event. " << e.what();
    }
  }

  event->start_time_ = torch::profiler::impl::getApproximateTime();
  event->allow_tf32_cublas_ = at::globalContext().allowTF32CuBLAS();
  return out;
}

RecordQueue::RecordQueue(
    const ProfilerConfig& config,
    std::set<ActivityType> activities)
    : id_(++queue_id_), config_{config}, activities_{activities} {
  if (tracePython()) {
    python_tracer::PythonTracerBase::get().start(this);
  }
}

bool RecordQueue::tracePython() const {
  return config_.with_stack && activities_.count(ActivityType::CPU);
}

ThreadLocalSubqueue* RecordQueue::getSubqueue() {
  // In the most common case, a thread will want to write to the same sub-queue
  // that it wrote to last call. The only time that isn't true is if:
  //  A) The profiler context has ended and we are in a new one.
  //  B) Two profilers are active in different TLS contexts, and this thread
  //     is a worker helping with intra-op parallelism.
  // Since we expect this to be the OVERWHELMINGLY common case (>99%), we add a
  // special thread_local cache so that we can skip the overall `flat_hash_map`
  // (and corresponding lock).
  if (id_ == sub_queue_cache_.key_) {
    return sub_queue_cache_.ref_;
  }

  const auto tid = at::RecordFunction::currentThreadId();
  std::lock_guard<std::mutex> guard(sub_queue_mutex_);
  auto it = sub_queues_.find(tid);
  if (it == sub_queues_.end()) {
    it = sub_queues_
             .emplace(tid, std::make_unique<ThreadLocalSubqueue>(tid, config_))
             .first;
  }

  sub_queue_cache_ = SubQueueThreadCache{id_, it->second.get()};
  return it->second.get();
}

void RecordQueue::stop() {
  if (tracePython()) {
    python_tracer::PythonTracerBase::get().stop();
  }
}

namespace {
template <typename T>
auto steal_or_default(T& it) {
  if (it.exhausted()) {
    return typename T::value_type();
  } else {
    auto result = std::move(*it);
    ++it;
    return result;
  }
}

void mark_finished(std::shared_ptr<Result>& r) {
  TORCH_INTERNAL_ASSERT(!r->finished_, r->name());
  r->finished_ = true;
  TORCH_INTERNAL_ASSERT(r->endTimeNS() >= r->start_time_ns_, r->name());
}

static constexpr const char* indexKey = "Profiler Event Index";

void passEventsToKineto(
    const std::vector<std::shared_ptr<Result>>& results,
    uint64_t start_time_us,
    uint64_t end_time_us) {
  using namespace torch::profiler::impl::kineto;
  TraceWrapper cpu_trace(start_time_us, "PyTorch Profiler");

  // Generate Kineto events for each event recorded by the PyTorch profiler.
  for (const auto i : c10::irange(results.size())) {
    const auto& e = results[i];
    const auto* activity = cpu_trace.addCPUActivity(
        e->name(),
        e->kinetoType(),
        e->kineto_info_,
        e->correlationID(),
        e->start_time_ns_ / 1000,
        e->endTimeNS() / 1000);

    TORCH_INTERNAL_ASSERT(activity || !kKinetoAvailable);
    if (activity) {
      addMetadata(activity, indexKey, std::to_string(i));
    }
  }

  // Kineto adds the events that it collected.
  cpu_trace.transferCpuTrace(end_time_us);
}

#ifdef USE_KINETO
// There are two mechanisms that we use to connect Profiler and Kineto events.
// The first is the correlation ID. The profiler pushes a unique integer at the
// start of an op and pops it at the end. Kineto then associates the events
// that it collects with that correlation ID and sets the linked activity of
// the events that it collected to point to the profiler op.
//
// However, this is not a sufficient description because it does not retain
// dependency information between kineto ops. Consider a call to `torch.add`.
// Three events will be collected:
//   `aten::add`          (TorchOp, collected by profiler)
//   `cudaLaunchKernel`   (CUDA runtime event, collected by Kineto)
//   `at::vectorized_...` (GPU kernel, collected by Kineto)
// If we only relied on correlation IDs we would set both Kineto events as
// children of the `at::add`, rather than the correct
//   `at::add -> cudaLaunchKernel -> at::vectorized_...`
//
// Kineto surfaces this information through a second concept called a "flow".
// In this example, the `cudaLaunchKernel` event is the start of a flow and the
// GPU kernel has the same flow id but is not a start event. Thus, when merging
// the Kineto events into the call tree we first add all events which are flow
// start nodes. We then merge the rest, trying to pair them with flow starts
// and falling back to correlation ID if necessary. For any nodes without
// linked events the caller is determined using the normal tree construction
// algorithm.
class TransferEvents {
  using itrace_t = libkineto::ITraceActivity;
  using activity_t = torch::profiler::impl::kineto::activity_t;

 public:
  TransferEvents(
      std::vector<std::shared_ptr<Result>>& results,
      trace_ptr_t& trace)
      : results_{results} {
    auto* trace_activities_ptr = trace->get()->activities();
    TORCH_INTERNAL_ASSERT(trace_activities_ptr != nullptr);
    trace_activities_ = *trace_activities_ptr;
    reassociate();
    extractEventsFromTrace();
    setParents();
  }

 private:
  static long long extractIndex(const std::string& metadata_json) {
    static const auto prefix = fmt::format("\"{}\": ", indexKey);
    auto pos = metadata_json.find(prefix);
    return (pos == std::string::npos) ? unmatchedIndex : [&]() {
      auto end = metadata_json.find(",", pos);
      end = (end == std::string::npos) ? metadata_json.size() : end;
      return std::stoll(metadata_json.substr(pos + prefix.size(), end));
    }();
  }

  std::shared_ptr<Result> lookup(const itrace_t* key) {
    if (key == nullptr) {
      return nullptr;
    }

    // First check the map.
    auto it = kineto_events_.find(key);
    if (it != kineto_events_.end()) {
      return it->second;
    }

    // Then fallback to the encoded metadata.
    const auto index = extractIndex(key ? key->metadataJson() : "");
    if (index != unmatchedIndex) {
      auto out = results_.get().at(index);
      kineto_events_[key] = out;
      return out;
    }

    // And finally give up.
    return nullptr;
  }

  void reassociate() {
    // Match profiler events with the corresponding kineto events. Kineto may
    // have moved or copied the activities, so we have to recover the
    // relationship between `libkineto::ITraceActivity` and `Result`.
    for (const auto* activity : trace_activities_) {
      TORCH_INTERNAL_ASSERT(activity != nullptr);
      auto e = lookup(activity);
      if (e != nullptr) {
        TORCH_INTERNAL_ASSERT(e->kineto_activity_ == nullptr);
        e->kineto_activity_ = static_cast<const activity_t*>(activity);
      }
    }
    if (results_.get().size() != kineto_events_.size()) {
      TORCH_WARN(fmt::format(
          "Failed to recover relationship between all profiler and kineto events: "
          "{} vs. {}  reassociated.",
          results_.get().size(),
          kineto_events_.size()));
    }
  }

  std::shared_ptr<Result> resultFromActivity(const itrace_t* activity) {
    TORCH_INTERNAL_ASSERT(activity != nullptr);

    // Kineto is inconsistent with types, so we have to cast to int32.
    torch::profiler::impl::kineto::DeviceAndResource device_and_resource{
        static_cast<int32_t>(activity->deviceId()),
        static_cast<int32_t>(activity->resourceId())};

    auto event = Result::create(
        activity->timestamp() * 1000,
        noTID, // Placeholder
        device_and_resource,
        ExtraFields<EventType::Kineto>{
            activity->name(),
            activity->duration(),
            static_cast<uint64_t>(activity->correlationId()),
            activity->type(),
            {/*id=*/static_cast<uint32_t>(activity->flowId()),
             /*type=*/static_cast<uint32_t>(activity->flowType()),
             /*start=*/activity->flowStart()}});

    // NB: It's tempting to set `event->kineto_activity_`; however we can only
    // guarantee that the events we passed to Kineto are of type
    // `GenericTraceActivity`. Others may derive from ITraceActivity and thus
    // are not safe to cast.
    return event;
  }

  std::shared_ptr<Result> toResult(const itrace_t* activity) {
    auto e = lookup(activity);

    // Until we are very sure that we can reassociate kineto and profiler
    // events we need to be very defensive.
    const auto type = activity->type();
    if (e == nullptr &&
        (type == libkineto::ActivityType::CPU_OP ||
         type == libkineto::ActivityType::CPU_INSTANT_EVENT ||
         type == libkineto::ActivityType::USER_ANNOTATION ||
         type == libkineto::ActivityType::PYTHON_FUNCTION)) {
      TORCH_WARN_ONCE(
          "Detected an event which was likely passed to kineto by the PyTorch "
          "profiler, but is not present in the set of known events: ",
          activity->name(),
          " This most likely means that Kineto has not "
          "maintained address stability for this event. Please report this to "
          "the PyTorch team.");
      return nullptr;
    }

    if (e == nullptr) {
      e = resultFromActivity(activity);
      results_.get().push_back(e);
      kineto_events_[activity] = e;
    }
    return e;
  }

  void extractEventsFromTrace() {
    for (const auto* activity : trace_activities_) {
      auto e = toResult(activity);
      const auto* linked_activity = activity->linkedActivity();
      if (e && linked_activity) {
        c10::visit(
            c10::overloaded(
                [&](ExtraFields<EventType::Kineto>& i) {
                  i.linked_activity_ = toResult(linked_activity);
                },
                [](auto&) { TORCH_INTERNAL_ASSERT(false); }),
            e->extra_fields_);
      }
    }
  }

  void setKinetoTID(
      std::shared_ptr<Result>& r,
      std::shared_ptr<Result> parent) {
    auto f = [&](ExtraFields<EventType::Kineto>& i) {
      TORCH_INTERNAL_ASSERT(r->start_tid_ == noTID);
      r->start_tid_ =
          parent ? parent->start_tid_ : at::RecordFunction::currentThreadId();
    };
    c10::visit(c10::overloaded(f, [](auto&) {}), r->extra_fields_);

    for (auto& child : r->children_) {
      setKinetoTID(child, r);
    }
  }

  void setParents() {
    // First pass: Collect start events and set parent to linked event.
    ska::flat_hash_map<int, std::shared_ptr<Result>> flow_map;
    for (auto& e : results_.get()) {
      TORCH_INTERNAL_ASSERT(e != nullptr);
      auto f = [&](const ExtraFields<EventType::Kineto>& i) {
        if (i.flow.type == libkineto::kLinkAsyncCpuGpu && i.flow.start) {
          auto inserted = flow_map.insert({i.flow.id, e});
#ifdef USE_ROCM
          if (inserted.second) {
            TORCH_WARN_ONCE(
                "ROCTracer produced duplicate flow start: ", i.flow.id);
          }
#else // USE_ROCM
          TORCH_INTERNAL_ASSERT(inserted.second);
#endif // USE_ROCM
        }
        TORCH_INTERNAL_ASSERT(e->parent_.expired());
        e->parent_ = i.linked_activity_;
      };
      c10::visit(c10::overloaded(f, [](const auto&) {}), e->extra_fields_);
    }

    // Second pass
    for (auto& e : results_.get()) {
      auto f = [&](const ExtraFields<EventType::Kineto>& i) {
        // Flow takes priority over linked event.
        const auto it = flow_map.find(i.flow.id);
        if (it != flow_map.end() &&
            i.flow.type == libkineto::kLinkAsyncCpuGpu && !i.flow.start) {
          e->parent_ = it->second;
        }

        // If a parent was set we have to do some bookkeeping.
        auto parent = e->parent_.lock();
        if (parent) {
          parent->children_.push_back(e);
          mark_finished(e);
        }
      };
      c10::visit(c10::overloaded(f, [](const auto&) {}), e->extra_fields_);
    }

    // Set TIDs now that we have established lineage.
    for (auto& e : results_.get()) {
      if (e->parent_.expired()) {
        setKinetoTID(e, nullptr);
      }
    }
  }

  static constexpr long long unmatchedIndex = -1;
  static constexpr auto noTID = std::numeric_limits<uint64_t>::max();
  std::reference_wrapper<std::vector<std::shared_ptr<Result>>> results_;
  std::vector<const itrace_t*> trace_activities_;
  ska::flat_hash_map<const itrace_t*, std::shared_ptr<Result>> kineto_events_;
};
#else
class TransferEvents {
 public:
  template <class... Args>
  TransferEvents(Args&&...) {}
};
#endif

trace_ptr_t addKinetoEvents(
    std::vector<std::shared_ptr<Result>>& results,
    uint64_t start_time_us,
    uint64_t end_time_us,
    const ProfilerConfig& config) {
  using namespace torch::profiler::impl::kineto;
  passEventsToKineto(results, start_time_us, end_time_us);

  // In on demand mode kineto is directly controlled by other machinery.
  if (config.state == ProfilerState::KINETO_ONDEMAND) {
    return nullptr;
  }

  auto trace = std::make_unique<ActivityTraceWrapper>(stopTrace());
  TORCH_INTERNAL_ASSERT(trace || !kKinetoAvailable);
  TransferEvents transfer{results, trace};
  return trace;
}

struct EvaluateFunctionVisitor {
  void operator()(
      ExtraFields<EventType::TorchOp>& first,
      ExtraFields<EventType::TorchOp>& second) {
    if (first.scope_ == at::RecordScope::FUNCTION &&
        second.scope_ == at::RecordScope::BACKWARD_FUNCTION &&
        first.name_.rfind("autograd::engine::evaluate_function: ", 0) == 0) {
      first.sequence_number_ = second.sequence_number_;
      first.forward_tid_ = second.forward_tid_;
    }
  }

  template <typename T0, typename T1>
  void operator()(T0&, T1&) {}
};

void set_autograd_evaluate(std::vector<std::shared_ptr<Result>>& results) {
  auto end = results.size() > 2 ? results.end() - 1 : results.begin();
  for (auto it = results.begin(); it < end; ++it) {
    if ((*it)->start_tid_ == (*(it + 1))->start_tid_) {
      c10::visit(
          EvaluateFunctionVisitor(),
          (*it)->extra_fields_,
          (*(it + 1))->extra_fields_);
    }
  }
}

using result_ptr_t = std::shared_ptr<Result>;
struct ResultGreater {
  bool operator()(const result_ptr_t& a, const result_ptr_t& b) const {
    return a->endTimeNS() > b->endTimeNS();
  }
};

void build_tree(std::vector<std::shared_ptr<Result>>& events) {
  set_autograd_evaluate(events);
  std::stable_sort(
      events.begin(), events.end(), [](const auto& a, const auto& b) {
        return a->start_time_ns_ < b->start_time_ns_;
      });

  using op_fields = ExtraFields<EventType::TorchOp>;
  ska::flat_hash_map<uint64_t, std::shared_ptr<Result>> stacks;
  std::priority_queue<result_ptr_t, std::vector<result_ptr_t>, ResultGreater>
      end_events_;

  auto push_event = [&stacks, &end_events_](std::shared_ptr<Result>& event) {
    // Kineto builds subtrees using correlation ids and flows, so some Kineto
    // events are already marked finished before the main tree building
    // algorithm. It's fine to ignore them; the root event of these subtrees
    // not a Kineto op and will be handled normally.
    if (c10::holds_alternative<ExtraFields<EventType::Kineto>>(
            event->extra_fields_) &&
        event->finished_) {
      return;
    }

    TORCH_INTERNAL_ASSERT(event->parent_.expired());
    for (const auto& child : event->children_) {
      TORCH_INTERNAL_ASSERT(child->finished_);
    }
    TORCH_INTERNAL_ASSERT(!event->finished_);

    auto parent_it = stacks.find(event->start_tid_);
    if (parent_it == stacks.end()) {
      auto fwd_tid = c10::visit(
          c10::overloaded(
              [](const op_fields& i) { return i.forward_tid_; },
              [](const auto&) -> uint64_t { return 0; }),
          event->extra_fields_);
      if (fwd_tid) {
        parent_it = stacks.find(fwd_tid);
      }
    }

    if (parent_it != stacks.end()) {
      event->parent_ = parent_it->second;
      parent_it->second->children_.push_back(event);
    }

    if (event->endTimeNS() > event->start_time_ns_) {
      stacks[event->start_tid_] = event;
      end_events_.push(event);
    } else if (event->endTimeNS() == std::numeric_limits<time_t>::min()) {
      // We use min time to indicate the lack of a termination event, so if we
      // encounter such a case we don't push to `end_events_`.
      stacks[event->start_tid_] = event;
    } else {
      mark_finished(event);
    }
  };

  auto pop_event = [&stacks](std::shared_ptr<Result> event) {
    if (event->finished_) {
      // This event was marked finished by a previous `pop_event` call.
      return;
    }

    auto start_tid = event->start_tid_;
    auto frame = stacks.at(start_tid);

    while (frame.get() != event.get()) {
      TORCH_INTERNAL_ASSERT(frame != nullptr);
      mark_finished(frame);
      TORCH_INTERNAL_ASSERT(!frame->parent_.expired());
      frame = frame->parent_.lock();
    }

    mark_finished(event);
    stacks.erase(start_tid);
    auto new_frame = event->parent_.lock();
    if (new_frame != nullptr) {
      stacks[start_tid] = new_frame;
    }
  };

  // Stack replay loop.
  for (auto& event : events) {
    while (!end_events_.empty() &&
           end_events_.top()->endTimeNS() < event->start_time_ns_) {
      pop_event(end_events_.top());
      end_events_.pop();
    }
    push_event(event);
  }

  // Cleanup remaining exit events.
  while (!end_events_.empty()) {
    pop_event(end_events_.top());
    end_events_.pop();
  }
}
} // namespace

std::pair<
    std::vector<std::shared_ptr<Result>>,
    std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>>
RecordQueue::getRecords(
    std::function<time_t(approx_time_t)> time_converter,
    uint64_t start_time_us,
    uint64_t end_time_us) {
  auto converter = [&](approx_time_t t) {
    return t == std::numeric_limits<approx_time_t>::min()
        ? std::numeric_limits<time_t>::min()
        : time_converter(t);
  };
  std::vector<std::shared_ptr<Result>> out;
  std::vector<python_tracer::CompressedEvent> python_enters;
  for (auto& subqueue_it : sub_queues_) {
    auto& queue = *subqueue_it.second;
    for (auto& i : queue.backend_events_) {
      auto start_time = i.start_time_us_;
      out.emplace_back(Result::create(
          /*start_time_ns_=*/start_time * 1000,
          /*start_tid_=*/queue.tid(),
          /*kineto_info_=*/queue.kineto_info(),
          /*extra_fields_=*/std::move(i)));
    }
    queue.backend_events_.clear();

    auto input_getter = queue.inputs_outputs_.getNextShapesAndDtypes();
    auto jit_stack_it = queue.jit_stack_.begin();
    auto jit_module_it = queue.jit_modules_.begin();
    auto extra_args_it = queue.extra_args_.begin();
    auto gpu_fallback_it = queue.gpu_fallback_.begin();
    for (auto event = queue.op_events_.begin(); event != queue.op_events_.end();
         ++event) {
      auto& i = *event;
      auto start_time = converter(i.start_time_);
      out.emplace_back(Result::create(
          start_time,
          /*start_tid_=*/queue.tid(),
          /*kineto_info_=*/queue.kineto_info(),
          /*extra_fields_=*/
          ExtraFields<EventType::TorchOp>(
              std::move(i.basic_fields_),
              ThreadLocalSubqueue::OpList::correlationID(event),
              converter(i.end_time_),
              input_getter(),
              steal_or_default(jit_stack_it),
              steal_or_default(jit_module_it),
              steal_or_default(extra_args_it),
              steal_or_default(gpu_fallback_it),
              i.allow_tf32_cublas_)));
    }
    queue.op_events_.clear();
    queue.inputs_outputs_.clear();
    queue.jit_stack_.clear();
    queue.jit_modules_.clear();
    queue.extra_args_.clear();
    queue.gpu_fallback_.clear();

    for (auto& i : queue.allocations_) {
      auto start_time = converter(i.start_time_);
      out.emplace_back(Result::create(
          start_time,
          /*start_tid_=*/queue.tid(),
          /*kineto_info_=*/queue.kineto_info(),
          /*extra_fields_=*/std::move(i)));
    }
    queue.allocations_.clear();
    for (auto& i : queue.ooms_) {
      auto start_time = converter(i.start_time_);
      out.emplace_back(Result::create(
          start_time,
          /*start_tid_=*/queue.tid(),
          /*kineto_info_=*/queue.kineto_info(),
          /*extra_fields_=*/std::move(i)));
    }
    queue.ooms_.clear();

    for (auto& i : queue.py_calls_) {
      python_enters.push_back(
          {i.first, queue.tid(), queue.kineto_info(), converter(i.second)});
    }
  }

  if (tracePython()) {
    auto& tracer = python_tracer::PythonTracerBase::get();
    for (auto i : tracer.getEvents(converter, python_enters)) {
      out.push_back(i);
    }
    tracer.clear();
  }

  auto trace = addKinetoEvents(out, start_time_us, end_time_us, config_);
  build_tree(out);
  return {out, std::move(trace)};
}

} // namespace impl
} // namespace profiler
} // namespace torch
