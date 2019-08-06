#pragma once
#include "caffe2/utils/threadpool/pthreadpool.h"

namespace at {
pthreadpool_t mobile_threadpool();
} // namespace at
