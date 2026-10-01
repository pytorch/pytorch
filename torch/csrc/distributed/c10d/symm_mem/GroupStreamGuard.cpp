#include <torch/csrc/distributed/c10d/symm_mem/GroupStreamGuard.hpp>

#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <map>
#include <unordered_map>
#include <utility>

namespace c10d::symmetric_memory {

// The previous guarded operation of one capture context.
struct GroupStreamGuard::Frontier {
  // Recorded at the end of each guarded operation of the context on that
  // operation's stream. A wait binds to the most recent record, so one event
  // per context suffices.
  c10::cuda::CUDAEvent done;
  // Stream of the previous guarded operation of the context.
  std::optional<c10::cuda::CUDAStream> last_stream;
};

struct GroupStreamGuard::State {
  std::mutex mu;
  // One frontier per capture context, nullopt for eager. A single frontier
  // would let an operation of one context overwrite another's, and the next
  // operation of that other context would then wait on nothing.
  std::map<std::optional<c10::CaptureId_t>, Frontier> frontiers;
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

void GroupStreamGuard::init_(const c10::intrusive_ptr<c10d::ProcessGroup>& pg) {
  TORCH_CHECK(pg != nullptr, "GroupStreamGuard: null ProcessGroup");
  const auto cur = c10::cuda::getCurrentCUDAStream();
  state_ = get_group_stream_state(pg, cur.device_index());
  lock_ = std::unique_lock<std::mutex>(state_->mu);
  const auto capture = c10::cuda::captureIdMayInitCtx(cur.stream());

  // Drop the frontiers of captures that have ended, comparing streams rather
  // than querying them, since a stored stream may since have been destroyed
  // (an external stream). A stream is in at most one capture and stays in it
  // until the capture ends, so a capture frontier whose last stream now runs
  // in another context is never needed again. At most one dead frontier per
  // stream remains.
  auto& frontiers = state_->frontiers;
  std::erase_if(frontiers, [&](const auto& entry) {
    const auto& [ctx, f] = entry;
    return ctx.has_value() && ctx != capture &&
        (!f.last_stream.has_value() || *f.last_stream == cur);
  });
  auto it = frontiers.try_emplace(capture).first;

  auto& frontier = it->second;
  if (frontier.last_stream.has_value() && *frontier.last_stream != cur) {
    // Waits for the previous pad operation of this context only: the event
    // was recorded just after its launch.
    frontier.done.block(cur);
  }
  frontier.last_stream = cur;
  frontier_ = &frontier;
}

GroupStreamGuard::GroupStreamGuard(const std::string& group_name) {
  init_(c10d::resolve_process_group(group_name));
}

GroupStreamGuard::GroupStreamGuard(
    const std::string& /*group_name*/,
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg) {
  init_(pg);
}

GroupStreamGuard::~GroupStreamGuard() {
  // Still under state_->mu, so the next guard's wait cannot be enqueued
  // before this record. Unconditional: the next stream is unknown here.
  if (frontier_ == nullptr || !frontier_->last_stream.has_value()) {
    return;
  }
  // record() throws on a stream carrying an earlier launch error, which in a
  // noexcept destructor is std::terminate. Demote to a warning, as
  // CUDAGuardImpl::uncheckedSetDevice does. c10_cuda_check_implementation
  // clears a non-sticky error, so this warning is its only report.
  bool recorded = false;
  try {
    frontier_->done.record(*frontier_->last_stream);
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
    frontier_->last_stream.reset();
  }
}

} // namespace c10d::symmetric_memory
