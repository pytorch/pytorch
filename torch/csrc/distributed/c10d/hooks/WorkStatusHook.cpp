// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/WorkStatusHook.hpp>

#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <ATen/Context.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/thread_name.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef _WIN32
#include <pthread.h>
#endif

namespace c10d {

namespace {

// Hook ids must not collide with user-registered or other built-in hooks.
std::atomic<int64_t> next_hook_id{0x57535441 /* 'WSTA' */};
// Stands in for stock ProcessGroupNCCL's local_id_, which numbers its groups
// per process.
std::atomic<size_t> next_pg_id{0};

// Cap on ops awaiting a completion push. One that never gets one (a hang, or a
// Work the backend never completes on its own) is evicted oldest first.
constexpr size_t kMaxInflight = 4096;
// Cap on completion keys pushed before the op's post-hook registered them.
constexpr size_t kMaxEarlyKeys = 1024;

const std::vector<std::string> TORCH_DIST_WORK_STATUS_PERIOD_MS = {
    "TORCH_DIST_WORK_STATUS_PERIOD_MS"};

// Stock ProcessGroupNCCL's NCCL_BACKEND_NAME; its header is CUDA-only.
constexpr const char* kStockNcclName = "nccl";

// Stock ProcessGroupNCCL's kWorkStatusUpdatePeriodMs.
constexpr int kDefaultPeriodMs = 30 * 1000;

std::chrono::milliseconds period() {
  static const int ms =
      getCvarInt(TORCH_DIST_WORK_STATUS_PERIOD_MS, kDefaultPeriodMs);
  return std::chrono::milliseconds(ms);
}

// The op names stock reports (opTypeToString).
OpType toOpType(HookOpName name, bool any_source) {
  switch (name) {
    case HookOpName::SEND:
      return OpType::SEND;
    case HookOpName::RECV:
      return any_source ? OpType::RECVANYSOURCE : OpType::RECV;
    case HookOpName::BROADCAST:
      return OpType::BROADCAST;
    case HookOpName::ALLREDUCE:
      return OpType::ALLREDUCE;
    case HookOpName::ALLREDUCE_COALESCED:
      return OpType::ALLREDUCE_COALESCED;
    case HookOpName::REDUCE:
      return OpType::REDUCE;
    case HookOpName::ALLGATHER:
      return OpType::ALLGATHER;
    case HookOpName::ALLGATHER_BASE:
      return OpType::_ALLGATHER_BASE;
    case HookOpName::ALLGATHER_COALESCED:
      return OpType::ALLGATHER_COALESCED;
    case HookOpName::ALLGATHER_INTO_TENSOR_COALESCED:
      return OpType::ALLGATHER_INTO_TENSOR_COALESCED;
    case HookOpName::REDUCE_SCATTER:
      return OpType::REDUCE_SCATTER;
    case HookOpName::REDUCE_SCATTER_BASE:
      return OpType::_REDUCE_SCATTER_BASE;
    case HookOpName::REDUCE_SCATTER_TENSOR_COALESCED:
      return OpType::REDUCE_SCATTER_TENSOR_COALESCED;
    case HookOpName::ALLTOALL:
      return OpType::ALLTOALL;
    case HookOpName::ALLTOALL_BASE:
      return OpType::ALLTOALL_BASE;
    case HookOpName::BARRIER:
      return OpType::BARRIER;
    case HookOpName::SCATTER:
      return OpType::SCATTER;
    case HookOpName::GATHER:
      return OpType::GATHER;
    case HookOpName::SPLIT:
    case HookOpName::NEW_WINDOW:
    case HookOpName::UNKNOWN:
      break;
  }
  return OpType::UNKNOWN;
}

// Ops on a single tensor list, which the pre-hook passes on one side only
// (inputs, or outputs for recv). Stock counts the list on both sides.
bool singleBuffer(HookOpName name) {
  return name == HookOpName::ALLREDUCE ||
      name == HookOpName::ALLREDUCE_COALESCED || name == HookOpName::REDUCE ||
      name == HookOpName::BROADCAST || name == HookOpName::SEND ||
      name == HookOpName::RECV;
}

size_t totalNumel(const std::vector<at::Tensor>& tensors) {
  size_t numel = 0;
  for (const auto& tensor : tensors) {
    if (tensor.defined()) {
      numel += static_cast<size_t>(tensor.numel());
    }
  }
  return numel;
}

c10::intrusive_ptr<Backend> tryGetBackend(
    ProcessGroup& pg,
    std::optional<c10::DeviceType> device) {
  try {
    return device ? pg.getBackend(*device) : pg.getDefaultBackend();
  } catch (const std::exception&) {
    return nullptr;
  }
}

// Whether the status is reported at all, so whether ops need tracking.
bool reporting() {
  return C10dLogger::getLogger() != nullptr;
}

// FlightRecorderHook's: whether the calling thread's current stream on the
// device is capturing a CUDA graph.
bool streamIsCapturing(c10::Device device) {
  if (device.is_cpu() || !c10::impl::hasDeviceGuardImpl(device.type())) {
    return false;
  }
  try {
    auto* guard_impl = c10::impl::getDeviceGuardImpl(device.type());
    return guard_impl->getStream(device).is_capturing();
  } catch (const std::exception&) {
    return false;
  }
}

// Whether the calling thread's current stream on its current device of the
// type is capturing. Devices without a context are skipped, as a stream there
// cannot be capturing, so that the check does not create one.
bool currentStreamIsCapturing(c10::DeviceType type) {
  if (type == c10::DeviceType::CPU || !c10::impl::hasDeviceGuardImpl(type)) {
    return false;
  }
  try {
    auto device = c10::impl::getDeviceGuardImpl(type)->getDevice();
    if (!at::globalContext()
             .getAcceleratorHooksInterface(type)
             .hasPrimaryContext(device.index())) {
      return false;
    }
    return streamIsCapturing(device);
  } catch (const std::exception&) {
    return false;
  }
}

std::optional<c10::Device> opDevice(const PreHookArgs& args) {
  for (const auto* tensors : {&args.input_tensors, &args.output_tensors}) {
    for (const auto& tensor : *tensors) {
      if (tensor.defined()) {
        return tensor.device();
      }
    }
  }
  return std::nullopt;
}

// One process-wide thread reports every attached group. Leaked, so neither it
// nor the hooks race static destruction; the thread itself is stopped at exit
// (see stopAtExit) because it calls the C10dLogger, which is a static.
struct Registry {
  std::mutex mutex;
  std::condition_variable cv;
  std::set<WorkStatusHook*> hooks;
  std::thread* thread = nullptr;
  std::atomic<bool> stop{false};
  bool exited = false;
};

Registry* registry_ptr = new Registry();

Registry& registry() {
  return *registry_ptr;
}

void emit(
    const std::vector<std::pair<std::string, C10dLoggingData>>& records,
    const std::atomic<bool>& stop) {
  auto* logger = C10dLogger::getLogger();
  for (const auto& [prefix, data] : records) {
    // A logger blocking on one group must not delay exit by the rest.
    if (stop) {
      return;
    }
    // Stock's line, verbatim for log parsers, and rate-limited per call site
    // like stock's.
#ifdef LOG_EVERY_MS
    C10_LOG_EVERY_MS(INFO, period().count())
        << prefix << "NCCL Work update periodically: "
        << "last enqueued NCCL work: " << data.integers.at("last_enqueued_work")
        << ", last completed NCCL work: "
        << data.integers.at("last_completed_work") << ".";
#endif // LOG_EVERY_MS
    if (logger == nullptr) {
      continue;
    }
    try {
      logger->log(data);
    } catch (const std::exception& e) {
      LOG(ERROR) << prefix << "WorkStatusHook: logger failed: " << e.what();
    }
  }
}

void runLoop() {
  c10::setThreadName("pt_work_status");
  auto& r = registry();
  std::unique_lock<std::mutex> lock(r.mutex);
  while (!r.cv.wait_for(lock, period(), [&] { return r.stop.load(); })) {
    if (!reporting()) {
      continue;
    }
    std::vector<std::pair<std::string, C10dLoggingData>> records;
    for (auto* hook : r.hooks) {
      records.emplace_back(hook->logPrefix(), hook->status());
    }
    // A slow or blocking logger holds up this thread only: attach and remove
    // need the registry lock, which is not held while logging.
    lock.unlock();
    emit(records, r.stop);
    lock.lock();
  }
  r.exited = true;
  r.cv.notify_all();
}

void stopAtExit() {
  auto& r = registry();
#if defined(_WIN32) && !defined(C10_USE_MSVC_STATIC_RUNTIME)
  // The CRT has already terminated the thread, possibly holding the mutex, as
  // Engine::~Engine explains. The thread object is leaked, never destroyed.
  r.stop = true;
#else
  std::unique_lock<std::mutex> lock(r.mutex);
  if (r.thread == nullptr) {
    return;
  }
  r.stop = true;
  r.cv.notify_all();
  // Bounded: a logger that never returns must not hang the exit. Leaving the
  // thread running then risks it touching the logger as it is destroyed,
  // which is no worse than the hang.
  if (r.thread->get_id() == std::this_thread::get_id() ||
      !r.cv.wait_for(
          lock, std::chrono::seconds(10), [&] { return r.exited; })) {
    r.thread->detach();
    return;
  }
  lock.unlock();
  r.thread->join();
#endif
}

// Starts the thread on first use, under the registry lock.
void ensureThreadLocked(Registry& r) {
  if (r.thread != nullptr) {
    return;
  }
  // Installed first, so that a child forked as the thread starts, with the
  // registry lock held, starts afresh too.
  static bool hooks_installed = [] {
    std::atexit(stopAtExit);
#ifndef _WIN32
    // A forked child has no thread, and the registry lock may have been held
    // by it at the fork. Start the child afresh; its inherited hooks stop
    // reporting.
    pthread_atfork(nullptr, nullptr, [] { registry_ptr = new Registry(); });
#endif
    return true;
  }();
  (void)hooks_installed;
  r.thread = new std::thread(runLoop);
}

} // namespace

struct WorkStatusHook::State {
  // How an op's completion is learnt.
  enum class Completion : uint8_t {
    // Pushed by key by the completion hook.
    kPush,
    // From a callback on the Work's future. CPU ops only: a device backend's
    // future may complete at launch and getFuture() may sync streams.
    kFuture,
    // Not at all; the op is only passed over by later completions.
    kNone,
  };

  struct OpInfo {
    // Sequence number reported for the op: like stock, collectives and p2p
    // ops are numbered by separate 1-based counters.
    int64_t seq = -1;
    HookOpName name = HookOpName::UNKNOWN;
    bool any_source = false;
    size_t numel_in = 0;
    size_t numel_out = 0;
    Completion completion = Completion::kNone;
    // Set once the op's Work is indexed in work_ids.
    std::optional<uint64_t> completion_key;
  };

  void onPre(const PreHookArgs& args);
  void onPost(const PostHookArgs& args, const std::weak_ptr<State>& weak);
  void onCompletionKey(uint64_t completion_key);
  void markCompleted(int64_t op_id, const OpInfo& info);
  void fill(C10dLoggingData& data);

  Completion completionFor(c10::DeviceType type) const;

  // Fixed before any hook is registered. Device types whose backend is the
  // group's default one, which alone has the completion hook registered, and
  // pushes completions.
  std::vector<c10::DeviceType> push_types;
  // Device types an op without tensors may have been issued on.
  std::vector<c10::DeviceType> device_types;
  // For ops without tensors (barrier), whose device is not passed: unknown if
  // the group has several backends.
  Completion barrier_completion = Completion::kNone;

  // Numbered whether or not the status is reported, so that ranks agree once
  // a logger is registered.
  std::atomic<int64_t> collective_seq{0};
  std::atomic<int64_t> p2p_seq{0};

  // Guards everything below. Lock order: the registry's mutex, then this one.
  // Nothing under it calls into a backend or takes the GIL, for the reasons
  // given on FlightRecorderHook::mutex_.
  std::mutex mutex;
  bool removed = false;
  OpInfo last_enqueued;
  OpInfo last_completed;
  int64_t last_completed_op_id = -1;
  // Ops awaiting completion, by op_id, and the completion key index. Bounded;
  // see onPre.
  std::map<int64_t, OpInfo> inflight;
  std::unordered_map<uint64_t, int64_t> work_ids;
  // Completion keys pushed before the op's post-hook indexed them, so that
  // the post-hook need not ask the Work, which is unsafe under graph capture
  // and wrong for nccl2's batched send/recv, whose placeholder Work reads
  // completed at once. Bounded, oldest first: keys of ops issued around the
  // hooks (directly on a backend) are never claimed.
  std::unordered_set<uint64_t> early_keys;
  std::deque<uint64_t> early_order;
  // Work types whose getFuture() threw (Work's default does, as do gloo's p2p
  // works), so it is not asked, and an exception thrown, for every op.
  std::unordered_set<std::type_index> futureless_types;
};

std::shared_ptr<WorkStatusHook> WorkStatusHook::attach(
    c10::intrusive_ptr<ProcessGroup> pg,
    std::vector<uint64_t> global_ranks) {
  TORCH_CHECK(pg, "WorkStatusHook: null process group");
  if (period().count() <= 0) {
    return nullptr;
  }
  // Stock ProcessGroupNCCL reports its own status. It only counts its own ops,
  // so a mixed group it serves is left to it entirely rather than reported
  // twice under the same pg_name.
  for (const auto& device : pg->getDeviceTypes()) {
    auto backend = tryGetBackend(*pg, device.type());
    if (backend && backend->getBackendName() == kStockNcclName) {
      return nullptr;
    }
  }

  auto hook = std::shared_ptr<WorkStatusHook>(
      new WorkStatusHook(std::move(pg), std::move(global_ranks)));
  auto& pg_ref = *hook->pg_;
  auto& state = *hook->state_;
  hook->push_completion_ = pg_ref.supportsCompletionHooks();
  auto default_backend = tryGetBackend(pg_ref, std::nullopt);
  std::set<Backend*> backends;
  for (const auto& device : pg_ref.getDeviceTypes()) {
    auto backend = tryGetBackend(pg_ref, device.type());
    backends.insert(backend.get());
    state.device_types.push_back(device.type());
    if (hook->push_completion_ && backend == default_backend) {
      state.push_types.push_back(device.type());
    }
  }
  if (backends.size() == 1) {
    state.barrier_completion = state.completionFor(state.device_types.front());
    // Its CPU ops, if it has any, complete their futures properly.
    if (state.barrier_completion == State::Completion::kNone &&
        std::find(
            state.device_types.begin(),
            state.device_types.end(),
            c10::DeviceType::CPU) != state.device_types.end()) {
      state.barrier_completion = State::Completion::kFuture;
    }
  }
  // The callbacks hold the state weakly, as FlightRecorderHook's hold it, so
  // that dropping the returned handle detaches the hook.
  std::weak_ptr<State> weak = hook->state_;
  pg_ref.registerPreHook(hook->hook_id_, [weak](const PreHookArgs& args) {
    if (auto self = weak.lock()) {
      self->onPre(args);
    }
  });
  pg_ref.registerPostHook(hook->hook_id_, [weak](const PostHookArgs& args) {
    if (auto self = weak.lock()) {
      self->onPost(args, weak);
    }
  });
  if (hook->push_completion_) {
    pg_ref.registerCompletionHook(
        hook->hook_id_, [weak](const CompletionHookArgs& args) {
          if (auto self = weak.lock()) {
            self->onCompletionKey(args.completionKey);
          }
        });
  }

  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  r.hooks.insert(hook.get());
  ensureThreadLocked(r);
  return hook;
}

void WorkStatusHook::logAll() {
  std::vector<std::pair<std::string, C10dLoggingData>> records;
  {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (auto* hook : r.hooks) {
      records.emplace_back(hook->logPrefix(), hook->status());
    }
  }
  emit(records, registry().stop);
}

WorkStatusHook::WorkStatusHook(
    c10::intrusive_ptr<ProcessGroup> pg,
    std::vector<uint64_t> global_ranks)
    : pg_(std::move(pg)),
      hook_id_(next_hook_id++),
      pg_id_(next_pg_id++),
      rank_(pg_->getRank()),
      pg_name_(pg_->getGroupName()),
      pg_desc_(pg_->getGroupDesc()),
      state_(std::make_shared<State>()) {
  if (global_ranks.empty()) {
    if (auto backend = tryGetBackend(*pg_, std::nullopt)) {
      try {
        global_ranks = backend->getBackendOptions()->global_ranks_in_group;
      } catch (const std::exception&) {
      }
    }
  }
  if (static_cast<size_t>(rank_) < global_ranks.size()) {
    global_rank_ = static_cast<int64_t>(global_ranks[rank_]);
  }
}

WorkStatusHook::~WorkStatusHook() {
  remove();
}

void WorkStatusHook::remove() {
  // Unregistered with the state mutex released, as FlightRecorderHook::remove
  // explains.
  c10::intrusive_ptr<ProcessGroup> pg;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!pg_) {
      return;
    }
    pg = std::move(pg_);
    state_->removed = true;
    state_->inflight.clear();
    state_->work_ids.clear();
    state_->early_keys.clear();
    state_->early_order.clear();
  }
  {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.hooks.erase(this);
  }
  pg->unregisterPreHook(hook_id_);
  pg->unregisterPostHook(hook_id_);
  if (push_completion_) {
    pg->unregisterCompletionHook(hook_id_);
  }
}

std::string WorkStatusHook::logPrefix() const {
  // Stock ProcessGroupNCCL::createLogPrefix.
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str(
        "[PG ID ",
        pg_id_,
        " PG GUID ",
        pg_name_,
        "(",
        pg_desc_,
        ") Rank ",
        rank_,
        "] ");
  }
  return c10::str(
      "[PG ID ", pg_id_, " PG GUID ", pg_name_, " Rank ", rank_, "] ");
}

C10dLoggingData WorkStatusHook::status() {
  C10dLoggingData data;
  data.integers["pg_id"] = static_cast<int64_t>(pg_id_);
  data.integers["rank"] = rank_;
  data.integers["global_rank"] = global_rank_;
  // Never observed; stock's defaults when collectives are not timed.
  data.integers["last_started_work"] = -1;
  data.integers["last_started_numel_in"] = 0;
  data.integers["last_started_numel_out"] = 0;
  data.strings["last_started_work_name"] = "";
  data.strings["pg_name"] = pg_name_;
  data.strings["pg_desc"] = pg_desc_;
  state_->fill(data);
  return data;
}

void WorkStatusHook::State::fill(C10dLoggingData& data) {
  std::lock_guard<std::mutex> lock(mutex);
  auto fill_op = [&](const std::string& which, const OpInfo& op) {
    data.integers[c10::str("last_", which, "_work")] = op.seq;
    data.integers[c10::str("last_", which, "_numel_in")] =
        static_cast<int64_t>(op.numel_in);
    data.integers[c10::str("last_", which, "_numel_out")] =
        static_cast<int64_t>(op.numel_out);
    data.strings[c10::str("last_", which, "_work_name")] =
        op.seq < 0 ? "" : opTypeToString(toOpType(op.name, op.any_source));
  };
  fill_op("enqueued", last_enqueued);
  fill_op("completed", last_completed);
}

WorkStatusHook::State::Completion WorkStatusHook::State::completionFor(
    c10::DeviceType type) const {
  if (std::find(push_types.begin(), push_types.end(), type) !=
      push_types.end()) {
    return Completion::kPush;
  }
  return type == c10::DeviceType::CPU ? Completion::kFuture : Completion::kNone;
}

void WorkStatusHook::State::onPre(const PreHookArgs& args) {
  const bool is_p2p =
      args.name == HookOpName::SEND || args.name == HookOpName::RECV;
  const int64_t seq = is_p2p ? ++p2p_seq : ++collective_seq;
  if (!reporting()) {
    return;
  }
  OpInfo op;
  op.seq = seq;
  op.name = args.name;
  op.any_source = args.name == HookOpName::RECV && args.root < 0;
  op.numel_in = totalNumel(args.input_tensors);
  op.numel_out = totalNumel(args.output_tensors);
  if (singleBuffer(args.name)) {
    op.numel_in = op.numel_out = std::max(op.numel_in, op.numel_out);
  }
  bool capturing = false;
  if (auto device = opDevice(args)) {
    capturing = streamIsCapturing(*device);
    op.completion = completionFor(device->type());
  } else {
    // barrier(): its device is not passed, but it is one of the group's.
    op.completion = barrier_completion;
    for (auto type : device_types) {
      capturing = capturing || currentStreamIsCapturing(type);
    }
  }

  std::lock_guard<std::mutex> lock(mutex);
  if (removed) {
    return;
  }
  if (capturing) {
    // Like stock: numbered, but neither enqueued nor completed.
    return;
  }
  last_enqueued = op;
  inflight[args.op_id] = op;
  while (inflight.size() > kMaxInflight) {
    auto oldest = inflight.begin();
    if (oldest->second.completion_key) {
      work_ids.erase(*oldest->second.completion_key);
    }
    inflight.erase(oldest);
  }
}

void WorkStatusHook::State::onPost(
    const PostHookArgs& args,
    const std::weak_ptr<State>& weak) {
  if (!reporting()) {
    return;
  }
  std::optional<std::type_index> work_type;
  if (args.work) {
    work_type.emplace(typeid(*args.work));
  }
  OpInfo op;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = inflight.find(args.op_id);
    if (it == inflight.end()) {
      return;
    }
    op = it->second;
    if (args.work && op.completion == Completion::kPush) {
      // Completion is pushed by key, possibly already.
      auto key = args.work->getCompletionKey();
      if (early_keys.erase(key) == 0) {
        work_ids[key] = args.op_id;
        it->second.completion_key = key;
        return;
      }
      inflight.erase(it);
      if (args.op_id > last_completed_op_id) {
        last_completed_op_id = args.op_id;
        last_completed = op;
      }
      return;
    }
    inflight.erase(it);
    if (!args.work || op.completion != Completion::kFuture ||
        futureless_types.count(*work_type)) {
      return;
    }
  }
  // Asked with mutex released: this calls into the backend.
  c10::intrusive_ptr<c10::ivalue::Future> future;
  try {
    future = args.work->getFuture();
  } catch (const std::exception&) {
    std::lock_guard<std::mutex> lock(mutex);
    futureless_types.insert(*work_type);
    return;
  }
  if (!future) {
    return;
  }
  // Runs inline if the future is already complete. uses_future=false: only
  // the error state is read, so CUDA futures need not sync streams for it.
  future->addCallback(
      [weak, op_id = args.op_id, op](c10::ivalue::Future& f) {
        if (f.hasError()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->markCompleted(op_id, op);
        }
      },
      /*uses_future=*/false);
}

void WorkStatusHook::State::onCompletionKey(uint64_t completion_key) {
  if (!reporting()) {
    return;
  }
  // May run on the backend's watchdog with its locks held: map lookups only.
  std::lock_guard<std::mutex> lock(mutex);
  if (removed) {
    return;
  }
  auto id_it = work_ids.find(completion_key);
  if (id_it == work_ids.end()) {
    if (early_keys.insert(completion_key).second) {
      early_order.push_back(completion_key);
    }
    while (early_order.size() > kMaxEarlyKeys) {
      early_keys.erase(early_order.front());
      early_order.pop_front();
    }
    return;
  }
  auto op_id = id_it->second;
  work_ids.erase(id_it);
  auto it = inflight.find(op_id);
  if (it == inflight.end()) {
    return;
  }
  if (op_id > last_completed_op_id) {
    last_completed_op_id = op_id;
    last_completed = it->second;
  }
  inflight.erase(it);
}

void WorkStatusHook::State::markCompleted(int64_t op_id, const OpInfo& info) {
  std::lock_guard<std::mutex> lock(mutex);
  // Futures may complete out of issue order; report the latest op.
  if (removed || op_id <= last_completed_op_id) {
    return;
  }
  last_completed_op_id = op_id;
  last_completed = info;
}

} // namespace c10d
