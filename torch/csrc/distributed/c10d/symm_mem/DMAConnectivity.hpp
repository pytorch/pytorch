#pragma once

#include <ATen/ATen.h>

namespace c10d {

struct TORCH_API DMAConnectivity : c10::intrusive_ptr_target {
  c10::DeviceType device_type;
  std::string connection_type;

  // This is an NxN matrix representing the connectivity between N devices,
  // where each element matrix[i][j] indicates the connectivity between device
  // i and device j. A value of 0 denotes that there is no connection between
  // device i and j. The meaning of non-zero values are specific to the
  // connection type (e.g., for NVLink it represents the number of NVLinks).
  std::vector<std::vector<int>> matrix;

  explicit DMAConnectivity(
      c10::DeviceType device_type,
      std::string connection_type,
      std::vector<std::vector<int>> matrix);
};

struct DMAConnectivityDetector : c10::intrusive_ptr_target {
  virtual c10::intrusive_ptr<DMAConnectivity> detect() = 0;
  ~DMAConnectivityDetector() override = default;
};

C10_EXPORT void register_dma_connectivity_detector(
    c10::DeviceType device_type,
    const std::string& connection_type,
    c10::intrusive_ptr<DMAConnectivityDetector> detector);

TORCH_API c10::intrusive_ptr<DMAConnectivity> detect_dma_connectivity(
    c10::DeviceType device_type,
    const std::string& connection_type);

} // namespace c10d
