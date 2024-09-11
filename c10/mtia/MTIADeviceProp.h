#pragma once

#include <c10/mtia/MTIAMacros.h>

namespace c10::mtia {
struct C10_MTIA_API MTIADeviceProp {
  char name[256]; // NOLINT
  int total_memory; // NOLINT
};

} // namespace c10::mtia
