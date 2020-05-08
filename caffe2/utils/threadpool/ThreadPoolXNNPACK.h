#pragma once
// Creating a separate .h/.cc file for creating threadpool for XNNPACK
// to avoid touching existing internal builds.
// When we unify threadpools this should all go away.
namespace caffe2 {
pthreadpool_t xnnpack_threadpool();
} // namespace caffe2
