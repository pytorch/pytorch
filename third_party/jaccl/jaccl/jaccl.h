// Copyright © 2025 Apple Inc.

#pragma once

#include <memory>
#include <vector>

#include "jaccl/group.h"

namespace jaccl {

class Config {
 public:
  Config();

  Config& set_rank(int rank);
  Config& set_coordinator(std::string coordinator);
  Config& set_devices(
      std::vector<std::vector<std::vector<std::string>>> devices);
  Config& prefer_ring(bool prefer = true);

  bool is_valid_mesh() const;
  bool is_valid_ring() const;

  int get_rank() const {
    return rank_;
  }

  int get_size() const {
    return size_;
  }

  std::string get_coordinator() const {
    return coordinator_;
  }

  bool get_prefer_ring() const {
    return prefer_ring_;
  }

  static std::optional<Config> from_env();

  friend std::shared_ptr<Group> init(const Config& cfg, bool strict);

 private:
  std::vector<std::string> get_mesh_connectivity() const;
  std::pair<std::vector<std::string>, std::vector<std::string>>
  get_ring_connectivity() const;

  int rank_;
  int size_;
  std::string coordinator_;
  std::vector<std::vector<std::vector<std::string>>> devices_;
  bool prefer_ring_;
};

/**
 * Check if JACCL (RDMA over Thunderbolt) is available on this system.
 */
bool is_available();

/**
 * Initialize a JACCL communication group from environment variables.
 *
 * Reads configuration from environment variables:
 *   - JACCL_RANK / MLX_RANK: The rank of this process
 *   - JACCL_IBV_DEVICES / MLX_IBV_DEVICES: Path to the device connectivity
 *     JSON file
 *   - JACCL_COORDINATOR / MLX_JACCL_COORDINATOR: IP:port of the coordinator
 *   - JACCL_RING / MLX_JACCL_RING: If set, prefer ring topology
 *
 * Args:
 *   strict: If true, throw on failure. If false, return nullptr.
 *
 * Returns:
 *   A shared_ptr to the Group, or nullptr on failure.
 */
std::shared_ptr<Group> init(bool strict = false);

/**
 * Initialize a JACCL communication group from an explicit Config object.
 */
std::shared_ptr<Group> init(const Config& cfg, bool strict = false);

} // namespace jaccl
