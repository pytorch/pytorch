// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/WorkStatusHook.hpp>

#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <ATen/Context.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/CallOnce.h>
#include <c10/util/thread_name.h>

#include <algorithm>
#include <array>
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

// Must not collide with other hooks' ids.
std::atomic<int64_t> next_hook_id{0x57535441 /* 'WSTA' */};
// Stands in for stock ProcessGroupNCCL's per-process local_id_.
std::atomic<size_t> next_pg_id{0};

// Caps on ops awaiting completion (one never completed, e.g. hung, is evicted
// oldest first) and on completion keys pushed before their post-hook ran.
constexpr size_t kMaxInflight = 4096;
constexpr size_t kMaxEarlyKeys = 1024;

const std::vector<std::string> TORCH_DIST_WORK_STATUS_PERIOD_MS = {
    "TORCH_DIST_WORK_STATUS_PERIOD_MS"};

// Stock ProcessGroupNCCL's NCCL_BACKEND_NAME; its header is CUDA-only.
constexpr const char* kStockNcclName = "nccl";
constexpr const char* kGlooName = "gloo";

// Stock ProcessGroupNCCL's kWorkStatusUpdatePeriodMs. Read once per process.
std::chrono::milliseconds period() {
  static const int ms = getCvarInt(TORCH_DIST_WORK_STATUS_PERIOD_MS, 30 * 1000);
  return std::chrono::milliseconds(ms);
}

// The op names stock reports, indexed by HookOpName.
static_assert(static_cast<int>(HookOpName::UNKNOWN) == 20, "update kOpTypes");
constexpr std::array<OpType, 21> kOpTypes = {
    OpType::SEND,
    OpType::RECV,
    OpType::BROADCAST,
    OpType::ALLREDUCE,
    OpType::REDUCE,
    OpType::ALLGATHER,
    OpType::REDUCE_SCATTER,
    OpType::ALLTOALL,
    OpType::BARRIER,
    OpType::SCATTER,
    OpType::GATHER,
    OpType::UNKNOWN, // SPLIT
    OpType::UNKNOWN, // NEW_WINDOW
    OpType::ALLREDUCE_COALESCED,
    OpType::_ALLGATHER_BASE,
    OpType::ALLGATHER_COALESCED,
    OpType::ALLGATHER_INTO_TENSOR_COALESCED,
    OpType::_REDUCE_SCATTER_BASE,
    OpType::REDUCE_SCATTER_TENSOR_COALESCED,
    OpType::ALLTOALL_BASE,
    OpType::UNKNOWN};

// Ops on one tensor list, which the pre-hook passes on one side only. Stock
// counts it on both.
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

// Whether the calling thread's current stream on the device is capturing.
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

// As above, on the current device of the type. One without a CUDA context
// cannot be capturing and is skipped, so that none is created.
bool currentStreamIsCapturing(c10::DeviceType type) {
  if (type == c10::DeviceType::CPU || !c10::impl::hasDeviceGuardImpl(type)) {
    return false;
  }
  try {
    auto device = c10::impl::getDeviceGuardImpl(type)->getDevice();
    if ((type == c10::DeviceType::CUDA || type == c10::DeviceType::HIP) &&
        !at::globalContext()
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

// One process-wide thread reports every attached group. Leaked, so that it
// and the hooks do not race static destruction; the thread is stopped at exit
// as it calls the C10dLogger, a static.
struct Registry {
  std::mutex mutex;
  std::condition_variable cv;
  std::set<WorkStatusHook*> hooks;
  std::thread* thread = nullptr;
  std::atomic<bool> stop{false};
  bool exited = false;
};

Registry* registry_ptr = new Registry();

using Records = std::vector<C10dLoggingData>;

Records snapshotLocked(Registry& r) {
  Records records;
  for (auto* hook : r.hooks) {
    records.push_back(hook->status());
  }
  return records;
}

void emit(const Records& records, const std::atomic<bool>& stop) {
  auto* logger = C10dLogger::getLogger();
  for (const auto& data : records) {
    // A logger blocking on one group must not delay exit by the rest.
    if (stop || logger == nullptr) {
      return;
    }
    try {
      logger->log(data);
    } catch (const std::exception& e) {
      LOG(ERROR) << "WorkStatusHook: logger failed for group "
                 << data.strings.at("pg_name") << ": " << e.what();
    }
  }
}

void runLoop() {
  c10::setThreadName("pt_work_status");
  auto& r = *registry_ptr;
  std::unique_lock<std::mutex> lock(r.mutex);
  while (!r.cv.wait_for(lock, period(), [&] { return r.stop.load(); })) {
    if (!reporting()) {
      continue;
    }
    auto records = snapshotLocked(r);
    // Logged unlocked, so a blocking logger holds up this thread only.
    lock.unlock();
    emit(records, r.stop);
    lock.lock();
  }
  r.exited = true;
  r.cv.notify_all();
}

void stopAtExit() {
  auto& r = *registry_ptr;
#if defined(_WIN32) && !defined(C10_USE_MSVC_STATIC_RUNTIME)
  // The CRT has already terminated the thread, possibly holding the mutex, as
  // Engine::~Engine explains.
  r.stop = true;
#else
  std::unique_lock<std::mutex> lock(r.mutex);
  if (r.thread == nullptr) {
    return;
  }
  r.stop = true;
  r.cv.notify_all();
  // Bounded, so a logger that never returns does not hang the exit.
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

void installProcessHooks() {
  static c10::once_flag once;
  c10::call_once(once, [] {
    std::atexit(stopAtExit);
#ifndef _WIN32
    // A forked child has no thread and may inherit a held registry lock.
    // Start it afresh; its inherited hooks stop reporting.
    pthread_atfork(nullptr, nullptr, [] { registry_ptr = new Registry(); });
#endif
  });
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
    // Like stock, collectives and p2p ops are numbered separately, from 1.
    int64_t seq = -1;
    OpType type = OpType::UNKNOWN;
    size_t numel_in = 0;
    size_t numel_out = 0;
    Completion completion = Completion::kNone;
    // Set once the op's Work is indexed in work_ids.
    std::optional<uint64_t> completion_key;
  };

  void onPre(const PreHookArgs& args);
  void onPost(const PostHookArgs& args, const std::weak_ptr<State>& weak);
  void onCompletionKey(uint64_t completion_key);
  void completeLocked(int64_t op_id, const OpInfo& info);
  void fill(C10dLoggingData& data);
  Completion completionFor(c10::DeviceType type) const;

  // Fixed before the hooks are registered. Device types served by the
  // default backend, which alone has the completion hook, if it pushes.
  std::vector<c10::DeviceType> push_types;
  std::vector<c10::DeviceType> device_types;
  // For barrier, whose device is not passed.
  Completion barrier_completion = Completion::kNone;

  // Numbered even when not reporting, so ranks agree once a logger is set.
  std::atomic<int64_t> collective_seq{0};
  std::atomic<int64_t> p2p_seq{0};

  // Guards everything below; taken after the registry's. Nothing under it
  // blocks, takes the GIL or takes a backend lock.
  std::mutex mutex;
  bool removed = false;
  OpInfo last_enqueued;
  OpInfo last_completed;
  int64_t last_completed_op_id = -1;
  // Ops awaiting completion, by op_id, and the completion key index.
  std::map<int64_t, OpInfo> inflight;
  std::unordered_map<uint64_t, int64_t> work_ids;
  // Keys pushed before the op's post-hook indexed them, so it need not ask
  // whether the Work completed. Bounded: keys of ops issued directly on a
  // backend are never claimed.
  std::unordered_set<uint64_t> early_keys;
  std::deque<uint64_t> early_order;
  // Work types whose getFuture() threw (e.g. gloo's p2p), so not asked again.
  std::unordered_set<std::type_index> futureless_types;
};

std::shared_ptr<WorkStatusHook> WorkStatusHook::attach(
    c10::intrusive_ptr<ProcessGroup> pg,
    std::vector<uint64_t> global_ranks) {
  TORCH_CHECK(pg, "WorkStatusHook: null process group");
  if (period().count() <= 0) {
    return nullptr;
  }
  // Stock ProcessGroupNCCL reports its own ops under the same pg_name, so a
  // group it serves is left to it entirely.
  std::set<Backend*> backends;
  for (const auto& device : pg->getDeviceTypes()) {
    auto backend = tryGetBackend(*pg, device.type());
    if (backend && backend->getBackendName() == kStockNcclName) {
      return nullptr;
    }
    backends.insert(backend.get());
  }

  auto hook = std::shared_ptr<WorkStatusHook>(
      new WorkStatusHook(std::move(pg), std::move(global_ranks)));
  auto& pg_ref = *hook->pg_;
  auto& state = *hook->state_;
  hook->push_completion_ = pg_ref.supportsCompletionHooks();
  auto default_backend = tryGetBackend(pg_ref, std::nullopt);
  bool all_cpu = true;
  for (const auto& device : pg_ref.getDeviceTypes()) {
    state.device_types.push_back(device.type());
    all_cpu = all_cpu && device.type() == c10::DeviceType::CPU;
    if (hook->push_completion_ &&
        tryGetBackend(pg_ref, device.type()) == default_backend) {
      state.push_types.push_back(device.type());
    }
  }
  // A barrier's backend is known only if the group has one. Its future
  // completes properly only if it runs on CPU, as gloo's always does.
  if (backends.size() == 1) {
    auto* backend = *backends.begin();
    const bool cpu_barrier =
        all_cpu || (backend && backend->getBackendName() == kGlooName);
    state.barrier_completion = !state.push_types.empty()
        ? State::Completion::kPush
        : cpu_barrier ? State::Completion::kFuture
                      : State::Completion::kNone;
  }
  // Held weakly, like FlightRecorderHook's state, so that dropping the
  // returned handle detaches the hook.
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

  installProcessHooks();
  auto& r = *registry_ptr;
  std::lock_guard<std::mutex> lock(r.mutex);
  r.hooks.insert(hook.get());
  if (r.thread == nullptr) {
    r.thread = new std::thread(runLoop);
  }
  return hook;
}

void WorkStatusHook::logAll() {
  if (!reporting()) {
    return;
  }
  auto& r = *registry_ptr;
  Records records;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    records = snapshotLocked(r);
  }
  emit(records, r.stop);
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
    auto& r = *registry_ptr;
    std::lock_guard<std::mutex> lock(r.mutex);
    r.hooks.erase(this);
  }
  pg->unregisterPreHook(hook_id_);
  pg->unregisterPostHook(hook_id_);
  if (push_completion_) {
    pg->unregisterCompletionHook(hook_id_);
  }
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
  auto fill_op = [&](const char* which, const OpInfo& op) {
    auto key = [&](const char* field) {
      return c10::str("last_", which, "_", field);
    };
    data.integers[key("work")] = op.seq;
    data.integers[key("numel_in")] = static_cast<int64_t>(op.numel_in);
    data.integers[key("numel_out")] = static_cast<int64_t>(op.numel_out);
    data.strings[key("work_name")] = op.seq < 0 ? "" : opTypeToString(op.type);
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
  op.type = args.name == HookOpName::RECV && args.root < 0
      ? OpType::RECVANYSOURCE
      : kOpTypes.at(static_cast<size_t>(args.name));
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
    // barrier(): its device is one of the group's.
    op.completion = barrier_completion;
    for (auto type : device_types) {
      capturing = capturing || currentStreamIsCapturing(type);
    }
  }

  std::lock_guard<std::mutex> lock(mutex);
  // Like stock, captured ops are numbered but neither enqueued nor completed.
  if (removed || capturing) {
    return;
  }
  last_enqueued = op;
  // Stays until evicted if the backend throws, as no post-hook follows.
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
    // Bound first: typeid evaluates an operand with side effects.
    const Work& work = *args.work;
    work_type.emplace(typeid(work));
  }
  OpInfo op;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = inflight.find(args.op_id);
    if (it == inflight.end()) {
      return;
    }
    op = it->second;
    const bool push = args.work && op.completion == Completion::kPush;
    // Completion is pushed by key, possibly already.
    const uint64_t key = push ? args.work->getCompletionKey() : 0;
    if (push && early_keys.erase(key) == 0) {
      work_ids[key] = args.op_id;
      it->second.completion_key = key;
      return;
    }
    inflight.erase(it);
    if (push) {
      completeLocked(args.op_id, op);
      return;
    }
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
  // Runs inline if already complete. uses_future=false: only the error state
  // is read, so CUDA futures need not sync streams.
  future->addCallback(
      [weak, op_id = args.op_id, op](c10::ivalue::Future& f) {
        if (f.hasError()) {
          return;
        }
        if (auto self = weak.lock()) {
          std::lock_guard<std::mutex> lock(self->mutex);
          if (!self->removed) {
            self->completeLocked(op_id, op);
          }
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
  completeLocked(op_id, it->second);
  inflight.erase(it);
}

void WorkStatusHook::State::completeLocked(int64_t op_id, const OpInfo& info) {
  // Ops may complete out of issue order; report the latest.
  if (op_id > last_completed_op_id) {
    last_completed_op_id = op_id;
    last_completed = info;
  }
}

} // namespace c10d
