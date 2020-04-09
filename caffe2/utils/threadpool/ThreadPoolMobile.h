#pragma once
#include <caffe2/utils/threadpool/pthreadpool.h>

// TODO Implement a parallel_for version for Mobile here, add to Aten/Parallel.h

namespace caffe2 {

class ThreadPool;

// Return a singleton instance of caffe2::ThreadPool for ATen/TH multithreading.
ThreadPool* mobile_threadpool();

// NOTE: This interface is temporary and should not be used.
// Please use Aten/Parallel.h for parallel primitives in pytorch.
// This implementation will be used by pytorch mobile, specifically
// NNPACK/QNNPACK. For mobile we need to use caffe2::ThreadPool instead of the
// 3rd party pthreadpool. Future work (TODO) Implement a mobile version of
// "at::parallel_for" using caffe2::ThreadPool so all ATen/TH multithreading
// usage is mobile friendly; Refactor QNNPACK or pthreadpool to explicitly using
// "at::parallel_for" primitive to replace pthreadpool_compute_1d for Pytorch;
pthreadpool_t mobile_pthreadpool();

size_t getDefaultNumThreads();
} // namespace caffe2
