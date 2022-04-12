#pragma once

#include "c10/util/Registry.h"
#include "caffe2/core/blob.h"
#include <c10/util/typeid.h>

#include <unordered_map>

namespace caffe2 {

struct BlobStatGetter {
  virtual size_t sizeBytes(const Blob& blob) const = 0;
  virtual ~BlobStatGetter() {}
};

struct BlobStatRegistry {
 private:
  std::unordered_map<TypeIdentifier, std::unique_ptr<BlobStatGetter>> map_;
  void doRegister(TypeIdentifier id, std::unique_ptr<BlobStatGetter>&& v);

 public:
  template <typename T, typename Getter>
  struct Registrar {
    Registrar() {
      BlobStatRegistry::instance().doRegister(
          TypeMeta::Id<T>(), std::unique_ptr<Getter>(new Getter));
    }
  };

  const BlobStatGetter* get(TypeIdentifier id);
  static BlobStatRegistry& instance();
};

#define REGISTER_BLOB_STAT_GETTER(Type, BlobStatGetterClass)    \
  static BlobStatRegistry::Registrar<Type, BlobStatGetterClass> \
      C10_ANONYMOUS_VARIABLE(BlobStatRegistry)

namespace BlobStat {

/**
 * Return size in bytes of the blob, if available for a blob of given type.
 * If not available, return 0.
 */
TORCH_API size_t sizeBytes(const Blob& blob);
}
}
