#pragma once
#include <caffe2/utils/threadpool/pthreadpool.h>

// TODO Implement a parallel_for version for Mobile here, add to Aten/Parallel.h

namespace at {
// This implementation uses the threadpool resources from caffe2::Threadpool
// It should be replaced with a mobile version of "at::parallel_for" using
// caffe2::ThreadPool so all ATen/TH multithreading usage is mobile friendly.
pthreadpool_t mobile_threadpool();
} // namespace at
