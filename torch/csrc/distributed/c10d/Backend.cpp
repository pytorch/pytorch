#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Hooks.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {

namespace {
void commonEventinit(
    details::EventInfo& evt,
    const Backend& backend,
    const Work& work) {
  evt.timestamp =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  evt.pg_name = backend.getGroupName();
  evt.backend = backend.getBackendName();
  evt.sequence_number = work.getSequencenumber();
  evt.operation = c10d::opTypeToString(work.retrieveOpType());
  evt.drop_count = 0;
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
  details::EventInfo evt;
  commonEventinit(evt, *this, work);

  evt.event_kind = ::c10d::EventKind::CollectiveStart;
  details::enqueue_c10d_event(std::move(evt));
}

void Backend::emitCollectiveEnd(const Work& work) {
  details::EventInfo evt;
  commonEventinit(evt, *this, work);

  evt.event_kind = ::c10d::EventKind::CollectiveEnd;
  evt.duration_ms = work.getDuration();
  details::enqueue_c10d_event(std::move(evt));
}

} // namespace c10d
