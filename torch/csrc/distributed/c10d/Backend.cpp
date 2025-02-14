#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

Backend::Backend(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

Backend::~Backend() = default;

void Backend::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.backend_{}", getBackendName()));
}

void getGlobalRankStartAndStride(
    const std::vector<uint64_t>& globalRanksInGroup,
    int& globalRankStart,
    int& globalRankStride) {
  if (globalRanksInGroup.empty()) {
    globalRankStart = 0;
  } else {
    globalRankStart = static_cast<int>(globalRanksInGroup[0]);
  }

  if (globalRanksInGroup.empty()) {
    globalRankStride = 1;
  } else if (globalRanksInGroup.size() == 1) {
    globalRankStride = 0;
  } else {
    bool ranksAreStrided = true;
    auto startRank = globalRanksInGroup[0];
    auto stride = globalRanksInGroup[1] - globalRanksInGroup[0];
    for (std::vector<uint64_t>::size_type i = 0; i < globalRanksInGroup.size();
         i++) {
      if (globalRanksInGroup[i] != startRank + i * stride) {
        ranksAreStrided = false;
        break;
      }
    }

    if (ranksAreStrided) {
      globalRankStride =
          static_cast<int>(globalRanksInGroup[1] - globalRanksInGroup[0]);
    } else {
      globalRankStride = -1;
    }
  }
}

} // namespace c10d
