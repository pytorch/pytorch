#include <torch/csrc/profiler/collection.h>

#include <algorithm>

#include <ATen/record_function.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace profiler {
namespace impl {
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

std::string Result::name() const {
  return c10::visit([](auto& e){ return e.name_; }, event_);
}

uint64_t Result::correlation_id() const {
  return c10::visit(c10::overloaded(
      [](const OpEvent& e){ return e.correlation_id_; },
      [](const BackendEvent& e) { return std::numeric_limits<uint64_t>::max(); }
  ), event_);
}

ThreadLocalSubqueue::ThreadLocalSubqueue(
    const uint64_t tid,
    const ProfilerConfig& config)
    : tid_{tid}, config_{config}, kineto_info_{kineto::kineto_ids()} {}

std::unique_ptr<KinetoObserverContext> ThreadLocalSubqueue::begin_op(
    const at::RecordFunction& fn,
    uint64_t correlation_id) {
  auto event = op_events_.emplace_back(
      correlation_id,
      fn.threadId(),
      fn.seqNr(),
      fn.forwardThreadId(),
      fn.scope(),
      fn.isAsync(),
      fn.debugHandle(),
      fn.name());
  if (config_.report_input_shapes) {
    inputs_.emplace_back(
        torch::profiler::impl::inputSizes(fn),
        torch::profiler::impl::inputTypes(fn));
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
  return out;
}

RecordQueue::RecordQueue(const ProfilerConfig& config)
    : id_(++queue_id_), config_{config} {}

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
    it =
        sub_queues_.emplace(tid, std::make_unique<ThreadLocalSubqueue>(tid, config_)).first;
  }

  sub_queue_cache_ = SubQueueThreadCache{id_, it->second.get()};
  return it->second.get();
}

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

std::deque<Result> RecordQueue::getRecords(
    std::function<time_t(approx_time_t)> time_converter) {
  auto converter = [&](approx_time_t t) {
    return t == std::numeric_limits<approx_time_t>::min()
        ? std::numeric_limits<int64_t>::min()
        : time_converter(t) / 1000; // ns to ms
  };
  std::deque<Result> out;
  for (auto& subqueue_it : sub_queues_) {
    auto& queue = *subqueue_it.second;
    for (auto& i : queue.backend_events_) {
      Result r;
      r.start_time_us_ = i.start_time_us_;
      r.end_time_us_ = i.end_time_us_;
      r.start_tid_ = queue.tid();
      r.kineto_info_ = queue.kineto_info();
      r.event_ = std::move(i);
      out.push_back(std::move(r));
    }

    auto input_it = queue.inputs_.begin();
    auto jit_stack_it = queue.jit_stack_.begin();
    auto jit_module_it = queue.jit_modules_.begin();
    auto extra_args_it = queue.extra_args_.begin();
    auto gpu_fallback_it = queue.gpu_fallback_.begin();
    for (auto& i : queue.op_events_) {
      Result r;
      r.start_time_us_ = converter(i.start_time_);
      r.end_time_us_ = converter(i.end_time_);
      r.start_tid_ = queue.tid();
      r.kineto_info_ = queue.kineto_info();
      r.event_ = std::move(i);
      r.inputs_ = steal_or_default(input_it);
      r.jit_stack_ = steal_or_default(jit_stack_it);
      r.jit_modules_ = steal_or_default(jit_module_it);
      r.extra_args_ = steal_or_default(extra_args_it);
      r.gpu_fallback_ = steal_or_default(gpu_fallback_it);

      out.push_back(std::move(r));
    }
    queue.op_events_.clear();
    queue.inputs_.clear();
    queue.jit_stack_.clear();
    queue.jit_modules_.clear();
    queue.extra_args_.clear();
    queue.gpu_fallback_.clear();
  }

  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.start_time_us_ < b.start_time_us_;
  });
  return out;
}

} // namespace impl
} // namespace profiler
} // namespace torch
