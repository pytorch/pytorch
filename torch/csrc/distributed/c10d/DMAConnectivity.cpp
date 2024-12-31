#include <torch/csrc/distributed/c10d/DMAConnectivity.hpp>
#include <utility>

namespace {

std::string get_detector_key(
    c10::DeviceType device_type,
    const std::string& connection_type) {
  std::ostringstream oss;
  oss << device_type << "/" << connection_type;
  return oss.str();
}

class DetectorMap {
 public:
  DetectorMap(const DetectorMap&) = delete;
  DetectorMap& operator=(const DetectorMap&) = delete;
  static DetectorMap& get() {
    static DetectorMap instance;
    return instance;
  }

  void register_detector(
      c10::DeviceType device_type,
      const std::string& connection_type,
      c10::intrusive_ptr<c10d::DMAConnectivityDetector> detector) {
    auto key = get_detector_key(device_type, connection_type);
    detector_map_[key] = std::move(detector);
  }

  c10::intrusive_ptr<c10d::DMAConnectivity> detect(
      c10::DeviceType device_type,
      const std::string& connection_type) {
    auto key = get_detector_key(device_type, connection_type);
    {
      auto it = cached_.find(key);
      if (it != cached_.end()) {
        return it->second;
      }
    }

    auto it = detector_map_.find(key);
    TORCH_CHECK(
        it != detector_map_.end(),
        "DMA connectivity detector for ",
        device_type,
        " over ",
        connection_type,
        " is not available");
    auto detector = it->second;
    auto connectivity = detector->detect();
    cached_[key] = connectivity;
    return connectivity;
  }

 private:
  DetectorMap() = default;

  std::unordered_map<
      std::string,
      c10::intrusive_ptr<c10d::DMAConnectivityDetector>>
      detector_map_;

  std::unordered_map<std::string, c10::intrusive_ptr<c10d::DMAConnectivity>>
      cached_;
};

} // namespace

namespace c10d {

DMAConnectivity::DMAConnectivity(
    c10::DeviceType device_type,
    std::string connection_type,
    std::vector<std::vector<int>> matrix)
    : device_type(device_type),
      connection_type(std::move(connection_type)),
      matrix(std::move(matrix)) {}

void register_dma_connectivity_detector(
    c10::DeviceType device_type,
    const std::string& connection_type,
    c10::intrusive_ptr<DMAConnectivityDetector> detector) {
  return DetectorMap::get().register_detector(
      device_type, connection_type, std::move(detector));
}

c10::intrusive_ptr<DMAConnectivity> detect_dma_connectivity(
    c10::DeviceType device_type,
    const std::string& connection_type) {
  return DetectorMap::get().detect(device_type, connection_type);
}

} // namespace c10d
