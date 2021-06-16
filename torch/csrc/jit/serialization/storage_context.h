#pragma once

#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {

// Used in torch.package and TorchScript serialization to coordinate 
// sharing of storages between models. Also used to create deterministic
// naming for storages.
class TORCH_API SerializationStorageContext {
 public:
  explicit SerializationStorageContext() : unique_id(0) {}

  uint64_t addStorage(c10::Storage storage) {
    TORCH_INTERNAL_ASSERT(!hasStorage(storage));
    uint64_t id = unique_id++;
    storage_id_map_.insert({storage, id});
    return id;
  }

  bool hasStorage(c10::Storage storage) {
    return storage_id_map_.find(storage) != storage_id_map_.end();
  }

  uint64_t getId(c10::Storage storage) {
    TORCH_INTERNAL_ASSERT(hasStorage(storage));
    return storage_id_map_.find(storage)->second;
  }

  ~SerializationStorageContext() = default;

 private:
  class StorageSerializationComparator {
   public:
    bool operator()(const c10::Storage& lhs, const c10::Storage& rhs) const {
      return rhs.unsafeGetStorageImpl() < lhs.unsafeGetStorageImpl();
    }
  };

  uint64_t unique_id;
  std::map<c10::Storage, uint64_t, StorageSerializationComparator>
      storage_id_map_;
};

// Used in torch.package and TorchScript deserialization to coordinate 
// sharing of storages between models.
class TORCH_API DeserializationStorageContext {
 public:
  explicit DeserializationStorageContext() {}

  void addStorage(const std::string& name, c10::Storage storage) {
    TORCH_INTERNAL_ASSERT(!hasStorage(name));
    name_storage_map_.insert({name, storage});
  }

  bool hasStorage(const std::string& name) {
    return name_storage_map_.find(name) != name_storage_map_.end();
  }

  c10::Storage getStorage(const std::string& name) {
    TORCH_INTERNAL_ASSERT(hasStorage(name));
    return name_storage_map_.find(name)->second;
  }
  ~DeserializationStorageContext() = default;

 private:
  std::map<std::string, c10::Storage> name_storage_map_;
};

} // namespace jit
} // namespace torch
