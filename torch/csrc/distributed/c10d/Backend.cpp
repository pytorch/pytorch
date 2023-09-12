#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Hooks.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {

Backend::Backend(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

Backend::~Backend() = default;

void Backend::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.backend_{}", getBackendName()));
}

void Backend::emitCollectiveStart(const Work& work) {
  details::EventInfo evt;
  evt.event_kind = details::EventKind::CollectionStart;
  evt.timestamp =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  evt.pg_name = getGroupName();
  evt.backend = getBackendName();
  evt.sequence_number = work.getSequencenumber();
  evt.operation = c10d::opTypeToString(work.retrieveOpType());

  details::enqueue_c10d_event(std::move(evt));
}

void Backend::emitCollectiveEnd(const Work& work) {
  details::EventInfo evt;
  evt.event_kind = details::EventKind::CollectionEnd;
  evt.timestamp =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  evt.pg_name = getGroupName();
  evt.backend = getBackendName();
  evt.sequence_number = work.getSequencenumber();
  evt.operation = c10d::opTypeToString(work.retrieveOpType());
  // FIXME change getDuration to return Optional<float>
  try {
    evt.duration_ms = work.getDuration();
  } catch (std::exception& e) {
    C10D_INFO("Duraction not available {}", e.what());
  }

  details::enqueue_c10d_event(std::move(evt));
}

} // namespace c10d
