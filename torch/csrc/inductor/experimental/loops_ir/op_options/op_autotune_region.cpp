// Anchor translation unit for the header-only autotune region helper.
// Owns nothing new: guarantees the header compiles standalone in the
// isolated extension build and pins the helper version string.

#include <string>

#include "op_autotune_region.h"

namespace torch::inductor::loops_ir::op_options {

namespace {
const char* kPinnedVersion = "autotune_region_v1";
static_assert(true, "header-only autotune region anchor");
} // namespace

const char* autotuneRegionAnchorVersion() {
  if (autotuneRegionVersion() != std::string(kPinnedVersion)) {
    reject(RejectReason::Context, "op_autotune_region version mismatch");
  }
  return kPinnedVersion;
}

} // namespace torch::inductor::loops_ir::op_options
