#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

Backend::Backend(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

Backend::~Backend() = default;

void Backend::incref_pyobject() const noexcept {
  pyobj_slot_.incref();
}

void Backend::decref_pyobject() const noexcept {
  pyobj_slot_.decref();
}

bool Backend::try_incref_pyobject() const noexcept {
  return pyobj_slot_.try_incref();
}

void Backend::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.backend_{}", getBackendName()));
}

} // namespace c10d
