#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <torch/csrc/distributed/c10d/DMAConnectivity.hpp>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>

#include <cuda_runtime.h>
#include <nvml.h>

namespace {

constexpr int max_nvlinks = 64;

std::string get_bus_id(int device_idx) {
  // NOLINTNEXTLINE(*array*)
  char bus_id[80];
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_idx));
  snprintf(
      bus_id,
      sizeof(bus_id),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);
  return std::string(bus_id);
}

struct C10_EXPORT NVLinkDetector : public c10d::DMAConnectivityDetector {
  c10::intrusive_ptr<c10d::DMAConnectivity> detect() override {
    int num_devices = 0;
    C10_CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    std::vector<std::vector<int>> matrix;
    matrix.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      matrix.emplace_back(num_devices, 0);
    }

    // Obtain the bus_id for all visible devices
    std::unordered_map<std::string, int> bus_id_to_device_idx;
    std::vector<std::string> bus_ids;
    bus_ids.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      auto bus_id = get_bus_id(i);
      bus_id_to_device_idx.emplace(bus_id, i);
      bus_ids.push_back(std::move(bus_id));
    }

    static const char* warning_msg =
        "PyTorch features that use NVLinkDetector may assume no NVLink presence.";

    auto driver_api = c10::cuda::DriverAPI::get();
    if (driver_api->nvmlInit_v2_() != NVML_SUCCESS) {
      LOG(WARNING)
          << "NVLinkDetector: Failed to initialize NVML via nvmlInit_v2. "
          << warning_msg;
      return c10::make_intrusive<c10d::DMAConnectivity>(
          c10::DeviceType::CUDA, "nvlink", std::move(matrix));
    }

    // Obtain the nvml device for all bus_ids
    std::vector<nvmlDevice_t> nvml_devices(num_devices, nullptr);
    for (int i = 0; i < num_devices; ++i) {
      auto res = driver_api->nvmlDeviceGetHandleByPciBusId_v2_(
          bus_ids[i].c_str(), &nvml_devices[i]);
      if (res != NVML_SUCCESS) {
        LOG(WARNING) << "NVLinkDetector: Failed to obtain NVML device via "
                     << "nvmlDeviceGetHandleByPciBusId_v2. " << warning_msg;
        return c10::make_intrusive<c10d::DMAConnectivity>(
            c10::DeviceType::CUDA, "nvlink", std::move(matrix));
      }
    }

    std::vector<int> switch_link_count(num_devices, 0);
    for (int i = 0; i < num_devices; ++i) {
      for (int link = 0; link < max_nvlinks; ++link) {
        nvmlIntNvLinkDeviceType_t deviceType{};
        auto ret = driver_api->nvmlDeviceGetNvLinkRemoteDeviceType_(
            nvml_devices[i], link, &deviceType);
        if (ret != NVML_SUCCESS) {
          // We've exhausted the NVLinks connected to this device. This error
          // is benign. There doesn't seem to be a reliable way to obtain the
          // maximum link value that can be passed to the API. Therefore, we
          // simply increment the link value until the API fails or we reach a
          // predefined maximum value.
          break;
        }
        // Remote device is GPU
        if (deviceType == NVML_NVLINK_DEVICE_TYPE_GPU) {
          nvmlPciInfo_t pciInfo;
          auto res = driver_api->nvmlDeviceGetNvLinkRemotePciInfo_v2_(
              nvml_devices[i], link, &pciInfo);
          if (res != NVML_SUCCESS) {
            LOG(WARNING) << "NVLinkDetector: Failed to obtain NVML device via "
                         << "nvmlDeviceGetHandleByPciBusId_v2. " << warning_msg;
            return c10::make_intrusive<c10d::DMAConnectivity>(
                c10::DeviceType::CUDA, "nvlink", std::move(matrix));
          }
          auto it = bus_id_to_device_idx.find(pciInfo.busId);
          if (it != bus_id_to_device_idx.end()) {
            if (i != it->second) {
              matrix[i][it->second] += 1;
            }
          }
          // Remote device is NVSwitch
        } else if (deviceType == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
          switch_link_count[i] += 1;
        }
      }
    }

    // Process NVSwitch connections.
    // For simplicity, we assume that all NVSwitches are interconnected.
    for (int i = 0; i < num_devices; ++i) {
      for (int j = 0; j < num_devices; ++j) {
        if (i == j) {
          continue;
        }
        matrix[i][j] += std::min(switch_link_count[i], switch_link_count[j]);
      }
    }

    return c10::make_intrusive<c10d::DMAConnectivity>(
        c10::DeviceType::CUDA, "nvlink", std::move(matrix));
  }
};

struct RegisterDetector {
  RegisterDetector() {
    register_dma_connectivity_detector(
        c10::DeviceType::CUDA, "nvlink", c10::make_intrusive<NVLinkDetector>());
  }
};

static RegisterDetector register_detector_;

} // namespace
#endif
