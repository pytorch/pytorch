#pragma once

#include <optional>

#include <ATen/ATen.h>

namespace c10d {

struct TORCH_API DMAConnectivity : c10::intrusive_ptr_target {
  c10::DeviceType device_type;
  std::string connection_type;
  std::vector<std::vector<int>> matrix;

  explicit DMAConnectivity(
      c10::DeviceType device_type,
      std::string connection_type,
      std::vector<std::vector<int>> matrix);
};

struct DMAConnectivityDetector : c10::intrusive_ptr_target {
  virtual c10::intrusive_ptr<DMAConnectivity> detect() = 0;
  virtual ~DMAConnectivityDetector() {}
};

C10_EXPORT void register_dma_connectivity_detector(
    c10::DeviceType device_type,
    const std::string& connection_type,
    c10::intrusive_ptr<DMAConnectivityDetector> detector);

TORCH_API c10::intrusive_ptr<DMAConnectivity> detect_dma_connectivity(
    c10::DeviceType device_type,
    const std::string& connection_type);

} // namespace c10d
