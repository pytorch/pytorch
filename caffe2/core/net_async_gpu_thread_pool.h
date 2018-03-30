#ifndef CAFFE2_CORE_NET_ASYNC_GPU_THREAD_POOL_H_
#define CAFFE2_CORE_NET_ASYNC_GPU_THREAD_POOL_H_

#include "caffe2/core/net_async_polling.h"

namespace caffe2 {

std::shared_ptr<TaskThreadPool> GetAsyncNetGPUThreadPool(int gpu_id);

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_GPU_THREAD_POOL_H_
