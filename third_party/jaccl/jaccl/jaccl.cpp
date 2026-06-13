// Copyright © 2025 Apple Inc.

#include <fstream>
#include <sstream>

#include <json.hpp>

#include "jaccl/jaccl.h"
#include "jaccl/mesh.h"
#include "jaccl/rdma.h"
#include "jaccl/ring.h"

using json = nlohmann::json;

namespace {

std::vector<std::vector<std::vector<std::string>>> parse_devices_json(
    const char* dev_file) {
  std::ifstream f(dev_file);
  json devices = json::parse(f);
  if (!devices.is_array()) {
    throw std::runtime_error(
        "[jaccl] The device file should start with an array");
  }

  std::vector<std::vector<std::vector<std::string>>> result(devices.size());
  for (int rank = 0; rank < devices.size(); rank++) {
    auto conn = devices[rank];
    if (!conn.is_array()) {
      throw std::runtime_error(
          "[jaccl] The device file should have an array of arrays");
    }
    if (conn.size() != devices.size()) {
      std::ostringstream msg;
      msg << "[jaccl] The device file should contain the connectivity of each rank to "
          << "all other ranks but rank " << rank << " contains only "
          << conn.size() << " entries.";
      throw std::runtime_error(msg.str());
    }

    result[rank].resize(conn.size());
    for (int dst = 0; dst < conn.size(); dst++) {
      auto names = conn[dst];
      if (names.is_string()) {
        result[rank][dst].push_back(names);
      } else if (names.is_array()) {
        for (auto name_it = names.begin(); name_it != names.end(); name_it++) {
          result[rank][dst].push_back(*name_it);
        }
      } else if (!names.is_null()) {
        throw std::runtime_error(
            "[jaccl] Device names should be null, a string or array of strings.");
      }
    }
  }

  return result;
}

template <typename First, typename... Rest>
const char* getenv(First first, Rest... rest) {
  const char* rs = std::getenv(first);
  if (rs != nullptr) {
    return rs;
  }
  if constexpr (sizeof...(rest) > 0) {
    return getenv(rest...);
  }
  return rs;
}

} // namespace

namespace jaccl {

Config::Config() : rank_(0), size_(0) {}

Config& Config::set_rank(int rank) {
  rank_ = rank;
  return *this;
}

Config& Config::set_coordinator(std::string coordinator) {
  coordinator_ = std::move(coordinator);
  return *this;
}

Config& Config::set_devices(
    std::vector<std::vector<std::vector<std::string>>> devices) {
  devices_ = std::move(devices);
  size_ = devices_.size();
  for (int r = 0; r < size_; r++) {
    if (size_ != devices_[r].size()) {
      std::ostringstream msg;
      msg << "[jaccl] The full connectivity matrix should be provided but we have "
          << size_ << " rows and row " << r << " has " << devices_[r].size()
          << " columns.";
      throw std::invalid_argument(msg.str());
    }
  }
  return *this;
}

Config& Config::prefer_ring(bool prefer /* = true */) {
  prefer_ring_ = prefer;
  return *this;
}

bool Config::is_valid_mesh() const {
  if (size_ < 2) {
    return false;
  }

  for (int src = 0; src < size_; src++) {
    for (int dst = 0; dst < size_; dst++) {
      if ((src == dst && devices_[src][dst].size() != 0) ||
          (src != dst && devices_[src][dst].size() == 0)) {
        return false;
      }
    }
  }
  return true;
}

bool Config::is_valid_ring() const {
  if (size_ < 2) {
    return false;
  }

  int num_connections = devices_[0][1].size();
  for (int src = 0; src < size_; src++) {
    int left = (src + size_ - 1) % size_;
    int right = (src + 1) % size_;
    for (int dst = 0; dst < size_; dst++) {
      if (dst == left || dst == right) {
        if (devices_[src][dst].size() != num_connections) {
          return false;
        }
      }
    }
  }
  return true;
}

std::vector<std::string> Config::get_mesh_connectivity() const {
  if (!is_valid_mesh()) {
    throw std::runtime_error("[jaccl] The devices do not form a valid mesh.");
  }
  std::vector<std::string> devices(size_);
  for (int dst = 0; dst < size_; dst++) {
    if (dst != rank_) {
      devices[dst] = devices_[rank_][dst][0];
    }
  }
  return devices;
}

std::pair<std::vector<std::string>, std::vector<std::string>>
Config::get_ring_connectivity() const {
  if (!is_valid_ring()) {
    throw std::runtime_error("[jaccl] The devices do not form a valid ring.");
  }
  int left = (rank_ + size_ - 1) % size_;
  int right = (rank_ + 1) % size_;

  return std::make_pair(devices_[rank_][left], devices_[rank_][right]);
}

std::optional<Config> Config::from_env() {
  const char* dev_file = getenv("JACCL_IBV_DEVICES", "MLX_IBV_DEVICES");
  const char* coordinator =
      getenv("JACCL_COORDINATOR", "MLX_JACCL_COORDINATOR");
  const char* rank_str = getenv("JACCL_RANK", "MLX_RANK");
  const char* ring = getenv("JACCL_RING", "MLX_JACCL_RING");

  if (!dev_file || !coordinator || !rank_str) {
    return std::nullopt;
  }

  return Config()
      .set_rank(std::atoi(rank_str))
      .set_coordinator(coordinator)
      .set_devices(parse_devices_json(dev_file))
      .prefer_ring(ring != nullptr);
}

bool is_available() {
  return ibv().is_available();
}

std::shared_ptr<Group> init(bool strict /* = false */) {
  auto cfg = Config::from_env();
  if (!cfg.has_value()) {
    if (strict) {
      std::ostringstream msg;
      msg << "[jaccl] You need to provide via environment variables a rank "
          << "(JACCL_RANK/MLX_RANK), a device file (JACCL_IBV_DEVICES/"
          << "MLX_IBV_DEVICES) and a coordinator ip/port (JACCL_COORDINATOR/"
          << "MLX_JACCL_COORDINATOR).";
      throw std::runtime_error(msg.str());
    }
    return nullptr;
  }

  return init(*cfg, strict);
}

std::shared_ptr<Group> init(const Config& cfg, bool strict /* = false */) {
  if (cfg.get_prefer_ring() && cfg.is_valid_ring()) {
    auto [left, right] = cfg.get_ring_connectivity();
    return std::make_shared<RingGroup>(
        cfg.get_rank(), cfg.get_size(), left, right, cfg.get_coordinator());
  } else if (cfg.is_valid_mesh()) {
    auto mesh = cfg.get_mesh_connectivity();
    return std::make_shared<MeshGroup>(
        cfg.get_rank(), mesh, cfg.get_coordinator());
  } else if (cfg.is_valid_ring()) {
    auto [left, right] = cfg.get_ring_connectivity();
    return std::make_shared<RingGroup>(
        cfg.get_rank(), cfg.get_size(), left, right, cfg.get_coordinator());
  } else {
    if (!strict) {
      return nullptr;
    }

    throw std::runtime_error(
        "[jaccl] The configuration should define a valid mesh or a valid ring.");
  }
}

} // namespace jaccl
