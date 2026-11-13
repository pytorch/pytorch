#pragma once

#include <ATen/core/ivalue.h>

namespace torch::jit {

// Used in torch.package and TorchScript serialization to coordinate
// sharing of storages between models. Also used to create deterministic
// naming for storages.
class TORCH_API SerializationStorageContext {
 public:
  explicit SerializationStorageContext() = default;
  SerializationStorageContext operator=(const SerializationStorageContext&) =
      delete;
  SerializationStorageContext(const SerializationStorageContext&) = delete;

  uint64_t getOrAddStorage(const c10::Storage& storage) {
    if (!hasStorage(storage)) {
      uint64_t size = storage_id_map_.size();
      storage_id_map_[storage] = size;
    }
    return storage_id_map_[storage];
  }

  bool hasStorage(const c10::Storage& storage) {
    return storage_id_map_.find(storage) != storage_id_map_.end();
  }

  ~SerializationStorageContext() = default;

 private:
  class StorageSerializationHash {
   public:
    size_t operator()(const c10::Storage& storage) const {
      return std::hash<void*>()(
          reinterpret_cast<void*>(storage.unsafeGetStorageImpl()));
    }
  };

  class StorageSerializationEqual {
   public:
    bool operator()(const c10::Storage& lhs, const c10::Storage& rhs) const {
      return lhs.unsafeGetStorageImpl() == rhs.unsafeGetStorageImpl();
    }
  };

  std::unordered_map<
      c10::Storage,
      uint64_t,
      StorageSerializationHash,
      StorageSerializationEqual>
      storage_id_map_;
};

// Used in torch.package and TorchScript deserialization to coordinate
// sharing of storages between models.
class TORCH_API DeserializationStorageContext {
 public:
  explicit DeserializationStorageContext() = default;
  DeserializationStorageContext operator=(
      const DeserializationStorageContext&) = delete;
  DeserializationStorageContext(const DeserializationStorageContext&) = delete;

  void addStorage(std::string name, c10::Storage storage) {
    TORCH_INTERNAL_ASSERT(!hasStorage(name));
    name_storage_map_.emplace(std::move(name), std::move(storage));
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
  std::unordered_map<std::string, c10::Storage> name_storage_map_;
};

} // namespace torch::jit
