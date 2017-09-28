/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_UTILS_PTHREADPOOL_IMPL_H_
#define CAFFE2_UTILS_PTHREADPOOL_IMPL_H_

#include "ThreadPoolCommon.h"

#ifndef CAFFE2_THREADPOOL_MOBILE
#error "mobile build state not defined"
#endif

#if CAFFE2_THREADPOOL_MOBILE

namespace caffe2 {

struct ThreadPool;

} // namespace caffe2

extern "C" {

// Wrapper for the caffe2 threadpool for the usage of NNPACK
struct pthreadpool {
  pthreadpool(caffe2::ThreadPool* pool) : pool_(pool) {}
  caffe2::ThreadPool* pool_;
};

} // extern "C"

#endif // CAFFE2_THREADPOOL_MOBILE

#endif  // CAFFE2_UTILS_PTHREADPOOL_IMPL_H_
