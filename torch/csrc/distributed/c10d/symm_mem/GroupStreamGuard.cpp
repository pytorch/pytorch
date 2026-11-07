#include <torch/csrc/distributed/c10d/symm_mem/GroupStreamGuard.hpp>

#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <unordered_map>
#include <utility>

namespace c10d::symmetric_memory {

// The last guarded operation of one capture context: the event recorded
// after its launch and the stream it launched on. A wait binds to the most
// recent record, so one event per context suffices.
struct GroupStreamGuard::LastOp {
  c10::cuda::CUDAEvent done;
  std::optional<c10::cuda::CUDAStream> stream;
};

struct GroupStreamGuard::State {
  std::mutex mu;
  // An event can only be waited on from the capture context it was recorded
  // in, so eager operations and the current capture each keep their own last
  // operation. One capture is tracked at a time: a new capture id takes over
  // the capture slot, so captures interleaved from several threads are not
  // ordered within themselves.
  LastOp eager;
  std::optional<c10::CaptureId_t> capture_id;
  LastOp capture;
  // Owning group, for liveness only.
  std::optional<c10::weak_intrusive_ptr<c10d::ProcessGroup>> pg;
};

namespace {

// Keyed by ProcessGroup identity and device. The weak reference in State
// keeps the group's memory from being reused, so a live entry always names
// the group it was created for.
using StreamStateKey = std::pair<const c10d::ProcessGroup*, c10::DeviceIndex>;

std::mutex g_stream_map_mutex;

// Leaked deliberately. State owns a CUDAEvent, and destroying one at process
// exit runs cudaEventDestroy after the driver has shut down, which throws in
// ~CUDAEvent and aborts.
using StreamStateMap = std::unordered_map<
    StreamStateKey,
    std::shared_ptr<GroupStreamGuard::State>,
    c10::hash<StreamStateKey>>;
StreamStateMap& stream_states() {
  static auto* states = new StreamStateMap();
  return *states;
}

// State for (pg, device), created when absent or when the entry's group has
// expired. Returned by shared_ptr so an active guard survives another thread
// replacing the entry.
std::shared_ptr<GroupStreamGuard::State> get_group_stream_state(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(g_stream_map_mutex);
  const StreamStateKey key{pg.get(), device};
  auto it = stream_states().find(key);
  const bool stale = it == stream_states().end() ||
      !it->second->pg.has_value() || it->second->pg->expired();
  if (stale) {
    // Drop entries whose group is gone, otherwise transient groups leak an
    // event each. Insertions are rare, so the scan is off any hot path.
    for (auto i = stream_states().begin(); i != stream_states().end();) {
      if (!i->second->pg.has_value() || i->second->pg->expired()) {
        i = stream_states().erase(i);
      } else {
        ++i;
      }
    }
    auto state = std::make_shared<GroupStreamGuard::State>();
    state->pg.emplace(pg);
    it = stream_states().insert_or_assign(key, std::move(state)).first;
  }
  return it->second;
}

} // namespace

GroupStreamGuard::GroupStreamGuard(const std::string& group_name) {
  const auto pg = c10d::resolve_process_group(group_name);
  TORCH_CHECK(pg != nullptr, "GroupStreamGuard: null ProcessGroup");
  const auto cur = c10::cuda::getCurrentCUDAStream();
  state_ = get_group_stream_state(pg, cur.device_index());
  lock_ = std::unique_lock<std::mutex>(state_->mu);

  const auto capture = c10::cuda::captureIdMayInitCtx(cur.stream());
  last_ = &state_->eager;
  if (capture.has_value()) {
    if (state_->capture_id != capture) {
      // A new capture: the previous capture's event cannot be waited on from
      // here, so its slot starts over.
      state_->capture_id = capture;
      state_->capture.stream.reset();
    }
    last_ = &state_->capture;
  }
  if (last_->stream.has_value() && *last_->stream != cur) {
    // Waits for the previous pad operation of this context only: the event
    // was recorded just after its launch.
    last_->done.block(cur);
  }
  last_->stream = cur;
}

GroupStreamGuard::~GroupStreamGuard() {
  // Still under state_->mu, so the next guard's wait cannot be enqueued
  // before this record. Unconditional: the next stream is unknown here.
  if (last_ == nullptr || !last_->stream.has_value()) {
    return;
  }
  // record() throws on a stream carrying an earlier launch error, which in a
  // noexcept destructor is std::terminate. Demote to a warning, as
  // CUDAGuardImpl::uncheckedSetDevice does. c10_cuda_check_implementation
  // clears a non-sticky error, so this warning is its only report.
  bool recorded = false;
  try {
    last_->done.record(*last_->stream);
    recorded = true;
  } catch (const std::exception& e) {
    TORCH_WARN(
        "symm_mem: ignoring error recording the stream-ordering event: ",
        e.what());
  } catch (...) {
    TORCH_WARN(
        "symm_mem: ignoring unknown error recording the stream-ordering "
        "event");
  }
  if (!recorded) {
    // No event to wait on: the next operation of this context runs unordered
    // against this one.
    last_->stream.reset();
  }
}

} // namespace c10d::symmetric_memory
