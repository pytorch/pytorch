// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/FlightRecorderHook.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <thread>
#include <unordered_map>
#include <vector>

#include <c10/util/thread_name.h>

namespace c10d {

namespace {

constexpr int64_t kFlightRecorderHookId = 0x46524543; // 'FREC'

std::string_view hookOpName(HookOpName name) {
  switch (name) {
    case HookOpName::SEND:
      return "send";
    case HookOpName::RECV:
      return "recv";
    case HookOpName::BROADCAST:
      return "broadcast";
    case HookOpName::ALLREDUCE:
      return "all_reduce";
    case HookOpName::REDUCE:
      return "reduce";
    case HookOpName::ALLGATHER:
      return "all_gather";
    case HookOpName::REDUCE_SCATTER:
      return "reduce_scatter";
    case HookOpName::ALLTOALL:
      return "all_to_all";
    case HookOpName::BARRIER:
      return "barrier";
    case HookOpName::SCATTER:
      return "scatter";
    case HookOpName::GATHER:
      return "gather";
    case HookOpName::SPLIT:
      return "split";
    case HookOpName::NEW_WINDOW:
      return "new_window";
    case HookOpName::UNKNOWN:
      break;
  }
  return "unknown";
}

bool isP2POp(HookOpName name) {
  return name == HookOpName::SEND || name == HookOpName::RECV;
}

// FlightRecorder keys process groups by a per-recorder monotonic id.
std::atomic<size_t> next_pg_id{0};

using TraceIdentifier = FlightRecorder<c10::Event>::TraceIdentifier;

class WorkCompletionTracker {
 public:
  void track(
      c10::intrusive_ptr<Work> work,
      std::function<void()> onCompletion) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.push_back({std::move(work), std::move(onCompletion)});
    }
    cv_.notify_one();
  }

 private:
  struct PendingWork {
    c10::intrusive_ptr<Work> work;
    std::function<void()> onCompletion;
  };

  WorkCompletionTracker() {
    std::thread([this] { run(); }).detach();
  }

  void run() {
    c10::setThreadName("pt_fr_work");
    while (true) {
      std::vector<PendingWork> current;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !pending_.empty(); });
        current.swap(pending_);
      }

      std::vector<PendingWork> incomplete;
      incomplete.reserve(current.size());
      for (auto& pending : current) {
        bool completed = false;
        try {
          completed = pending.work->isCompleted();
        } catch (...) {
          // A backend that cannot report completion must not leave a trace
          // permanently active.
          completed = true;
        }
        if (completed) {
          pending.onCompletion();
        } else {
          incomplete.push_back(std::move(pending));
        }
      }

      if (!incomplete.empty()) {
        std::unique_lock<std::mutex> lock(mutex_);
        pending_.insert(
            pending_.end(),
            std::make_move_iterator(incomplete.begin()),
            std::make_move_iterator(incomplete.end()));
        cv_.wait_for(lock, std::chrono::milliseconds(100));
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<PendingWork> pending_;

  friend WorkCompletionTracker& completionTracker();
};

WorkCompletionTracker& completionTracker() {
  static auto* tracker = new WorkCompletionTracker();
  return *tracker;
}

class TracebackRegistry {
 public:
  void record(
      const c10::intrusive_ptr<c10::ivalue::Future>& future,
      TraceIdentifier trace) {
    if (!future || !trace.id || !trace.reset_epoch) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    traces_[future.get()] = {
        c10::weak_intrusive_ptr<c10::ivalue::Future>(future), trace};
    order_.push_back({future.get(), trace});

    const auto capacity = FlightRecorder<c10::Event>::get()->maxEntries();
    while (order_.size() > capacity) {
      auto oldest = order_.front();
      order_.pop_front();
      auto it = traces_.find(oldest.future);
      if (it != traces_.end() && it->second.trace.id == oldest.trace.id &&
          it->second.trace.reset_epoch == oldest.trace.reset_epoch) {
        traces_.erase(it);
      }
    }
  }

  std::string getTraceback(
      const c10::intrusive_ptr<c10::ivalue::Future>& future) {
    if (!future) {
      return "";
    }

    TraceIdentifier trace;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = traces_.find(future.get());
      if (it == traces_.end()) {
        return "";
      }
      auto registered = it->second.future.lock();
      if (!registered || registered.get() != future.get()) {
        traces_.erase(it);
        return "";
      }
      trace = it->second.trace;
    }

    auto entry = FlightRecorder<c10::Event>::get()->getEntry(
        trace.id, trace.reset_epoch);
    return entry ? entry->getTraceback() : "";
  }

 private:
  struct RegisteredTrace {
    c10::weak_intrusive_ptr<c10::ivalue::Future> future;
    TraceIdentifier trace;
  };

  struct OrderedTrace {
    c10::ivalue::Future* future;
    TraceIdentifier trace;
  };

  std::mutex mutex_;
  std::unordered_map<c10::ivalue::Future*, RegisteredTrace> traces_;
  std::deque<OrderedTrace> order_;
};

TracebackRegistry& tracebackRegistry() {
  static auto* registry = new TracebackRegistry();
  return *registry;
}

class FlightRecorderHookState
    : public std::enable_shared_from_this<FlightRecorderHookState> {
 public:
  explicit FlightRecorderHookState(ProcessGroup* pg)
      : pg_(pg),
        pg_id_(next_pg_id++),
        pg_status_(std::make_shared<ProcessGroupStatus>()) {
    TORCH_CHECK(pg_, "FlightRecorderHook: null process group");
    // Backend options are optional on custom backends (getBackendOptions
    // throws by default); fall back to identity ranks and the default timeout.
    std::vector<uint64_t> ranks;
    try {
      auto options = pg_->getDefaultBackend()->getBackendOptions();
      ranks = options->global_ranks_in_group;
      timeout_ = options->timeout;
    } catch (const std::exception&) {
      ranks.clear();
    }
    if (ranks.empty()) {
      ranks.reserve(pg_->getSize());
      for (int r = 0; r < pg_->getSize(); ++r) {
        ranks.push_back(static_cast<uint64_t>(r));
      }
    }
    FlightRecorder<c10::Event>::get()->record_pg_ranks(
        std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc()),
        std::move(ranks));
  }

  void onPre(const PreHookArgs& args) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool is_p2p = isP2POp(args.name);
    size_t collective_seq = is_p2p ? collective_seq_ : ++collective_seq_;
    size_t p2p_seq = is_p2p ? ++p2p_seq_ : p2p_seq_;

    pg_status_->lastEnqueuedSeq =
        static_cast<int64_t>(is_p2p ? p2p_seq : collective_seq);
    pg_status_->lastEnqueuedWorkName = std::string(hookOpName(args.name));

    auto inputs = args.input_tensors;
    auto outputs = args.output_tensors;
    if (inputs.empty() && args.name == HookOpName::RECV) {
      inputs = outputs;
    }
    if (outputs.empty() &&
        (args.name == HookOpName::BROADCAST ||
         args.name == HookOpName::ALLREDUCE ||
         args.name == HookOpName::REDUCE || args.name == HookOpName::BARRIER ||
         args.name == HookOpName::SEND)) {
      outputs = inputs;
    }
    auto backend = backendName(args);

    auto trace_id = FlightRecorder<c10::Event>::get()->recordWithResetEnabled(
        pg_id_,
        std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc()),
        collective_seq,
        p2p_seq,
        static_cast<size_t>(args.op_id),
        c10::str(backend, ":", hookOpName(args.name)),
        inputs,
        outputs,
        /*start=*/nullptr,
        /*end=*/nullptr,
        timeout_,
        pg_status_,
        is_p2p);
    inflight_.emplace(
        args.op_id,
        InflightTrace{
            trace_id,
            static_cast<int64_t>(is_p2p ? p2p_seq : collective_seq),
            std::string(hookOpName(args.name))});
  }

  void onPost(const PostHookArgs& args) {
    std::optional<InflightTrace> trace;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = inflight_.find(args.op_id);
      if (it == inflight_.end()) {
        return;
      }
      trace = std::move(it->second);
      inflight_.erase(it);
    }
    if (!trace->id.id || !trace->id.reset_epoch) {
      return;
    }

    auto trace_id = trace->id;
    auto retire = [self = shared_from_this(), trace = std::move(*trace)] {
      {
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->pg_status_->lastCompletedSeq = trace.sequence;
        self->pg_status_->lastCompletedWorkName = trace.name;
      }
      FlightRecorder<c10::Event>::get()->retire_id(
          trace.id.id, trace.id.reset_epoch, /*compute_duration=*/false);
    };
    if (!args.work) {
      retire();
      return;
    }

    c10::intrusive_ptr<c10::ivalue::Future> future;
    try {
      future = args.work->getFutureResult();
    } catch (...) {
      future = nullptr;
    }
    tracebackRegistry().record(future, trace_id);
    completionTracker().track(args.work, std::move(retire));
  }

 private:
  std::string backendName(const PreHookArgs& args) const {
    const at::Tensor* tensor = nullptr;
    if (!args.input_tensors.empty()) {
      tensor = &args.input_tensors.front();
    } else if (!args.output_tensors.empty()) {
      tensor = &args.output_tensors.front();
    }
    if (tensor && tensor->defined()) {
      return pg_->getBackend(tensor->device().type())->getBackendName();
    }
    return pg_->getDefaultBackend()->getBackendName();
  }

  struct InflightTrace {
    TraceIdentifier id;
    int64_t sequence;
    std::string name;
  };

  ProcessGroup* pg_;
  size_t pg_id_;
  std::shared_ptr<ProcessGroupStatus> pg_status_;
  std::chrono::milliseconds timeout_{kBackendDefaultTimeout};
  std::mutex mutex_;
  size_t collective_seq_{0};
  size_t p2p_seq_{0};
  std::unordered_map<int64_t, InflightTrace> inflight_;
};

} // namespace

std::shared_ptr<FlightRecorderHook> FlightRecorderHook::attach(
    c10::intrusive_ptr<ProcessGroup> pg) {
  TORCH_CHECK(pg, "FlightRecorderHook: null process group");
  if (!isInstalled(pg.get())) {
    install(pg.get());
  }
  return std::shared_ptr<FlightRecorderHook>(
      new FlightRecorderHook(std::move(pg)));
}

bool FlightRecorderHook::isEnabled() {
  return FlightRecorder<c10::Event>::get()->enabled_;
}

std::string FlightRecorderHook::getFlightRecorderTraceback(
    const c10::intrusive_ptr<c10::ivalue::Future>& future) {
  return tracebackRegistry().getTraceback(future);
}

FlightRecorderHook::FlightRecorderHook(c10::intrusive_ptr<ProcessGroup> pg)
    : pg_(std::move(pg)) {}

FlightRecorderHook::~FlightRecorderHook() = default;

void FlightRecorderHook::install(ProcessGroup* pg) {
  TORCH_CHECK(pg, "FlightRecorderHook: null process group");
  if (isInstalled(pg)) {
    return;
  }
  auto state = std::make_shared<FlightRecorderHookState>(pg);
  pg->registerPreHook(kFlightRecorderHookId, [state](const PreHookArgs& args) {
    state->onPre(args);
  });
  pg->registerPostHook(
      kFlightRecorderHookId,
      [state](const PostHookArgs& args) { state->onPost(args); });
}

bool FlightRecorderHook::isInstalled(ProcessGroup* pg) {
  return pg->preHooks_.contains(kFlightRecorderHookId) &&
      pg->postHooks_.contains(kFlightRecorderHookId);
}

void FlightRecorderHook::remove() {
  if (pg_) {
    pg_->unregisterPreHook(kFlightRecorderHookId);
    pg_->unregisterPostHook(kFlightRecorderHookId);
    pg_.reset();
  }
}

} // namespace c10d
