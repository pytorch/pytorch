#include <torch/csrc/profiler/collection.h>

#include <algorithm>
#include <limits>
#include <queue>

#include <fmt/format.h>

#include <ATen/Context.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace profiler {
namespace impl {

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
    py_c_field)                                                          \
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
} // namespace

DEFINE_VISITOR(
    name,
    e.name_,
    e.name_,
    "[memory]",
    "[OutOfMemory]",
    toString(e),
    e.function_name_.str());
DEFINE_VISITOR(
    kinetoType,
    scopeToType(e.scope_),
    scopeToType(e.scope_),
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::PYTHON_FUNCTION,
    libkineto::ActivityType::PYTHON_FUNCTION);
DEFINE_VISITOR(correlationID, e.correlation_id_, 0, 0, 0, 0, 0);
DEFINE_VISITOR(
    endTimeNS,
    torchOpEndNS(e, finished_, parent_),
    e.end_time_us_ * 1000,
    start_time_ns_,
    start_time_ns_,
    e.end_time_ns_,
    e.end_time_ns_);
DEFINE_VISITOR(
    endTID,
    e.end_tid_,
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
    c10::DeviceType::CPU);
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

void addKinetoEvents(
    std::vector<std::shared_ptr<Result>>& results,
    uint64_t start_time_us,
    uint64_t end_time_us) {
  torch::profiler::impl::kineto::TraceWrapper cpu_trace(
      start_time_us, "PyTorch Profiler");

  for (auto& e : results) {
    e->kineto_activity_ = cpu_trace.addCPUActivity(
        e->name(),
        e->kinetoType(),
        e->kineto_info_,
        e->correlationID(),
        e->start_time_ns_ / 1000,
        e->endTimeNS() / 1000);
  }

  cpu_trace.transferCpuTrace(end_time_us);
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
    TORCH_INTERNAL_ASSERT(event->parent_.expired());
    TORCH_INTERNAL_ASSERT(event->children_.empty());
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

std::vector<std::shared_ptr<Result>> RecordQueue::getRecords(
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

  addKinetoEvents(out, start_time_us, end_time_us);
  build_tree(out);
  return out;
}

} // namespace impl
} // namespace profiler
} // namespace torch
