#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

Backend::Backend(int rank, int size) : rank_(rank), size_(size) {
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

Backend::~Backend() {}

void Backend::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.backend_{}", getBackendName()));
}

} // namespace c10d
