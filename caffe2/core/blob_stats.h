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
