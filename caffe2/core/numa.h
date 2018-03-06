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

#ifndef CAFFE2_CORE_NUMA_H_
#define CAFFE2_CORE_NUMA_H_

#include "caffe2/core/logging.h"

CAFFE2_DECLARE_bool(caffe2_cpu_numa_enabled);

namespace caffe2 {

bool IsNUMAEnabled();

void NUMABind(int numa_node_id);

int GetNUMANode(const void* ptr);

int GetNumNUMANodes();

void NUMAMove(void* ptr, size_t size, int numa_node_id);

int GetCurrentNUMANode();

} // namespace caffe2

#endif // CAFFE2_CORE_NUMA_H_
