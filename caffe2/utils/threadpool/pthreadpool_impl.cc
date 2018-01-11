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

#include "caffe2/utils/threadpool/pthreadpool.h"
#include "caffe2/utils/threadpool/pthreadpool_impl.h"
#include "caffe2/utils/threadpool/ThreadPool.h"


//
// External API
//

void pthreadpool_compute_1d(struct pthreadpool* threadpool,
                            pthreadpool_function_1d_t function,
                            void* argument,
                            size_t range) {
    threadpool->pool_->run(
      [function, argument](int threadId, size_t workId) {
        function(argument, workId);
      },
      range);
}

size_t pthreadpool_get_threads_count(struct pthreadpool* threadpool) {
  return threadpool->pool_->getNumThreads();
}
