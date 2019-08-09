#include "ATen/ThreadLocalDebugInfo.h"

namespace at {

namespace internal {
CAFFE2_API thread_local std::shared_ptr<ThreadLocalDebugInfoBase> debug_info;
}
} // namespace at
