#pragma once

#include <c10/core/Device.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace c10::impl {

using GeneratorFactory = std::function<c10::intrusive_ptr<c10::GeneratorImpl>(c10::Device)>;

// Thread-safe registry to map DeviceType -> factory
class C10_API GeneratorRegistry {
 public:
  static GeneratorRegistry& instance();

  // Registers a factory function for a specific device type.
  void register_factory(c10::DeviceType device_type, GeneratorFactory factory);
  // Creates a generator for a given device using the registered factory.
  c10::intrusive_ptr<c10::GeneratorImpl> make(c10::Device device) const;
  // Checks if a factory exists for the specified device type.
  bool has(c10::DeviceType device_type) const;

 private:
  GeneratorRegistry() = default;
  GeneratorRegistry(const GeneratorRegistry&) = delete;
  GeneratorRegistry& operator=(const GeneratorRegistry&) = delete;

  mutable std::mutex mu_;
  std::unordered_map<c10::DeviceType, GeneratorFactory> factories_;
};

// Convenience free functions
C10_API void registerGenerator(c10::DeviceType device_type, GeneratorFactory factory);
C10_API c10::intrusive_ptr<c10::GeneratorImpl> makeGenerator(c10::Device device);
C10_API bool hasGenerator(c10::DeviceType device_type);

} // namespace c10::impl