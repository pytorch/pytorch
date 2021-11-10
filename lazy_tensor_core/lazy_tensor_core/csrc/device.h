#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <c10/util/Deprecated.h>

namespace torch_lazy_tensors {

// Backend can extend it and define their own supported hardware types.
struct BackendDeviceType {
  int8_t type {0};
  virtual std::string toString() const { return "Unknown"; }
};

// TODO(alanwaketan): Rename it to BackendDevice.
class Device {
 public:
  // The default constructor will set both the device type and ordinal
  // to backend specific defaults.
  Device();
  Device(std::shared_ptr<BackendDeviceType>&& type, int ordinal);

  int8_t type() const;
  int64_t ordinal() const { return ordinal_;  }

  bool operator==(const Device& other) const { return compare(other) == 0; }
  bool operator!=(const Device& other) const { return compare(other) != 0; }
  bool operator<(const Device& rhs) const { return compare(rhs) < 0; }

  std::string toString() const;

  // The string -> Device conversion should be handled by the backend interface.
  C10_DEPRECATED explicit Device(const std::string& device_spec);

 private:
  int compare(const Device& rhs) const;

  // Use shared_ptr instead of unique_ptr so that Device can be copied.
  std::shared_ptr<BackendDeviceType> type_;
  int64_t ordinal_ {0};
};

std::ostream& operator<<(std::ostream& os, const Device& device);

}  // namespace torch_lazy_tensors
