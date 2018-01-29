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

#include <atomic>

#include "caffe2/core/common.h"

namespace caffe2 {

// A global variable to mark if Caffe2 has cuda linked to the current runtime.
// Do not directly use this variable, but instead use the HasCudaRuntime()
// function below.
std::atomic<bool> g_caffe2_has_cuda_linked{false};

bool HasCudaRuntime() {
  return g_caffe2_has_cuda_linked.load();
}

namespace internal {
void SetCudaRuntimeFlag() {
  g_caffe2_has_cuda_linked.store(true);
}
} // namespace internal

const std::map<string, string>& GetBuildOptions() {
#ifndef CAFFE2_BUILD_STRINGS
#define CAFFE2_BUILD_STRINGS {}
#endif
  static const std::map<string, string> kMap = CAFFE2_BUILD_STRINGS;
  return kMap;
}

} // namespace caffe2
