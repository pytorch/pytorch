#ifndef CAFFE2_CORE_NET_ASYNC_GPU_THREAD_POOL_H_
#define CAFFE2_CORE_NET_ASYNC_GPU_THREAD_POOL_H_

#include "caffe2/core/net_async_base.h"

namespace caffe2 {

std::shared_ptr<TaskThreadPoolBase>
GetAsyncNetGPUThreadPool(int gpu_id, int pool_size, bool create_new);

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_GPU_THREAD_POOL_H_
