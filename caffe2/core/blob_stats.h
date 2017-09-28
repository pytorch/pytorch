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

#pragma once

#include "caffe2/core/blob.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/typeid.h"

#include <unordered_map>

namespace caffe2 {

struct BlobStatGetter {
  virtual size_t sizeBytes(const Blob& blob) const = 0;
  virtual ~BlobStatGetter() {}
};

struct BlobStatRegistry {
 private:
  std::unordered_map<CaffeTypeId, std::unique_ptr<BlobStatGetter>> map_;
  void doRegister(CaffeTypeId id, std::unique_ptr<BlobStatGetter>&& v);

 public:
  template <typename T, typename Getter>
  struct Registrar {
    Registrar() {
      BlobStatRegistry::instance().doRegister(
          TypeMeta::Id<T>(), std::unique_ptr<Getter>(new Getter));
    }
  };

  const BlobStatGetter* get(CaffeTypeId id);
  static BlobStatRegistry& instance();
};

#define REGISTER_BLOB_STAT_GETTER(Type, BlobStatGetterClass)    \
  static BlobStatRegistry::Registrar<Type, BlobStatGetterClass> \
      CAFFE_ANONYMOUS_VARIABLE(BlobStatRegistry)

namespace BlobStat {

/**
 * Return size in bytes of the blob, if available for a blob of given type.
 * If not available, return 0.
 */
size_t sizeBytes(const Blob& blob);
}
}
