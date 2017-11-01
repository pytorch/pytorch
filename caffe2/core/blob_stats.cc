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

#include "caffe2/core/blob_stats.h"

namespace caffe2 {

const BlobStatGetter* BlobStatRegistry::get(CaffeTypeId id) {
  auto it = map_.find(id);
  if (it == map_.end()) {
    return nullptr;
  }
  return it->second.get();
}

BlobStatRegistry& BlobStatRegistry::instance() {
  static BlobStatRegistry registry;
  return registry;
}

void BlobStatRegistry::doRegister(
    CaffeTypeId id,
    std::unique_ptr<BlobStatGetter>&& v) {
  // don't use CAFFE_ENFORCE_EQ to avoid static initialization order fiasco.
  if (map_.count(id) > 0) {
    throw std::runtime_error("BlobStatRegistry: Type already registered.");
  }
  map_[id] = std::move(v);
}

namespace BlobStat {

size_t sizeBytes(const Blob& blob) {
  auto* p = BlobStatRegistry::instance().get(blob.meta().id());
  return p ? p->sizeBytes(blob) : 0;
}

} // namespace BlobStats
}
