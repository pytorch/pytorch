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

#ifndef CAFFE2_OPERATORS_CONV_OP_SHARED_H_
#define CAFFE2_OPERATORS_CONV_OP_SHARED_H_

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

/**
 * Creates a mutex and shared buffer in the workspace.
 * Not thread-safe, must be called from the constructor.
 */
template <typename Context>
void createSharedBuffer(Workspace* ws);

/**
 * Thread-safe, can be invoked from RunOnDevice() to serialize
 * access to shared buffer.
 */
template <typename Context>
void runWithSharedBuffer(
    Workspace* ws,
    std::function<void(Tensor<Context>* buffer)> f);
} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_SHARED_H_
