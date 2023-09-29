#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Hooks.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {

namespace {

std::string exceptionPtrWhat(const std::exception_ptr& eptr) {
  try {
    std::rethrow_exception(eptr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "<unknown error>";
  }
}

void commonEventinit(
    ::c10d::EventInfo& evt,
    const Backend& backend,
    const Work& work) {
  evt.timestamp =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  evt.pg_name = backend.getGroupName();
  evt.backend = backend.getBackendName();
  evt.sequence_number = work.getSequencenumber();
  evt.operation = c10d::opTypeToString(work.retrieveOpType());
  // isCompleted is mutable :facepalm:
  if (const_cast<Work&>(work).isCompleted() && !work.isSuccess())
    evt.error_message = exceptionPtrWhat(work.exception());
}
} // namespace

Backend::Backend(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

Backend::~Backend() = default;

void Backend::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.backend_{}", getBackendName()));
}

void Backend::emitCollectiveStart(const Work& work) {
  EventInfo evt;
  commonEventinit(evt, *this, work);

  evt.event_kind = EventKind::CollectionStart;
  details::enqueue_c10d_event(std::move(evt));
}

void Backend::emitCollectiveEnd(const Work& work) {
  EventInfo evt;
  commonEventinit(evt, *this, work);

  evt.event_kind = EventKind::CollectionEnd;
  // FIXME change getDuration to return Optional<float>
  try {
    evt.duration_ms = work.getDuration();
  } catch (std::exception& e) {
    C10D_INFO("Duraction not available {}", e.what());
  }
  details::enqueue_c10d_event(std::move(evt));
}

} // namespace c10d
