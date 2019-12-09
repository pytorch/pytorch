#include <cpuinfo.h>
#include <stdint.h>
#include <stdlib.h>

#include "caffe2/core/logging.h"
#include "caffe2/utils/bench_utils.h"

namespace caffe2 {

uint32_t wipe_cache() {
  static uint32_t* wipe_buffer = nullptr;
  static size_t wipe_size = 0;

  if (wipe_buffer == nullptr) {
    CAFFE_ENFORCE(cpuinfo_initialize(), "failed to initialize cpuinfo");
    const cpuinfo_processor* processor = cpuinfo_get_processor(0);
    if (processor->cache.l4 != nullptr) {
      wipe_size = processor->cache.l4->size;
    } else if (processor->cache.l3 != nullptr) {
      wipe_size = processor->cache.l3->size;
    } else if (processor->cache.l2 != nullptr) {
      wipe_size = processor->cache.l2->size;
    } else {
      wipe_size = processor->cache.l1d->size;
    }
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
    /*
     * On ARM precise cache size is not available, and cpuinfo may
     * underestimate. Use max for uArch (see src/arm/cache.c)
     */
    switch (processor->core->uarch) {
      case cpuinfo_uarch_cortex_a5:
        wipe_size = 512 * 1024; /* Max observed */
        break;
      case cpuinfo_uarch_cortex_a7:
        wipe_size = 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a8:
        wipe_size = 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a9:
        wipe_size = 1024 * 1024; /* Max observed */
        break;
      case cpuinfo_uarch_cortex_a12:
      case cpuinfo_uarch_cortex_a17:
        wipe_size = 8 * 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a15:
        wipe_size = 4 * 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a35:
        wipe_size = 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a53:
        wipe_size = 2 * 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a57:
        wipe_size = 2 * 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a72:
        wipe_size = 4 * 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a73:
        wipe_size = 8 * 1024 * 1024; /* uArch max */
        break;
      case cpuinfo_uarch_cortex_a55:
      case cpuinfo_uarch_cortex_a75:
      case cpuinfo_uarch_meerkat_m3:
        wipe_size = 4 * 1024 * 1024; /* DynamIQ max */
        break;
      default:
        wipe_size = 60 * 1024 * 1024;
        break;
    }
#endif
    LOG(INFO) << "Allocating cache wipe buffer of size " << wipe_size;
    wipe_buffer = static_cast<uint32_t*>(malloc(wipe_size));
    CAFFE_ENFORCE(wipe_buffer != nullptr);
  }
  uint32_t hash = 0;
  for (uint32_t i = 0; i * sizeof(uint32_t) < wipe_size; i += 8) {
    hash ^= wipe_buffer[i];
    wipe_buffer[i] = hash;
  }
  /* Make sure compiler doesn't optimize the loop away */
  return hash;
}

} /* namespace caffe2 */
