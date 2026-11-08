#pragma once

#include <cstdint>

#if defined(USE_ROCM) && !defined(_WIN32) && __has_include(<hsa/hsa_ext_amd.h>)
#include <ATen/DynamicLibrary.h>
#include <hsa/hsa_ext_amd.h>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace at::cuda::detail {

struct HsaCacheApi {
  decltype(&hsa_init) init;
  decltype(&hsa_shut_down) shut_down;
  decltype(&hsa_iterate_agents) iterate_agents;
  decltype(&hsa_agent_get_info) agent_get_info;
};

// HIP device properties have no LLC field. Reuse its runtime without adding a
// link-time dependency or a second management library.
inline HsaCacheApi loadHsaCacheApi() {
  try {
    static at::DynamicLibrary library(
        "libhsa-runtime64.so.1", "libhsa-runtime64.so", /*leak_handle=*/true);
    return {
        reinterpret_cast<decltype(&hsa_init)>(library.sym("hsa_init")),
        reinterpret_cast<decltype(&hsa_shut_down)>(
            library.sym("hsa_shut_down")),
        reinterpret_cast<decltype(&hsa_iterate_agents)>(
            library.sym("hsa_iterate_agents")),
        reinterpret_cast<decltype(&hsa_agent_get_info)>(
            library.sym("hsa_agent_get_info"))};
  } catch (const c10::DynamicLibraryError&) {
    return {};
  }
}

class LastLevelCache {
 public:
  explicit LastLevelCache(HsaCacheApi api) : api_(api) {}

  int64_t get(const char* uuid) {
    // HIP stores the 16 bytes following "GPU-" in HSA_AMD_AGENT_INFO_UUID.
    // This identity survives visibility remapping; a device ordinal does not.
    std::string key(uuid, 16);
    if (key.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos ||
        key == "0000000000000000") {
      return 0;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = sizes_.find(key);
    if (it == sizes_.end()) {
      it = sizes_.emplace(key, query(uuid)).first;
    }
    return it->second;
  }

 private:
  struct Query {
    const HsaCacheApi& api;
    const char* uuid;
    uint32_t size = 0;
    bool found = false;
  };

  static hsa_status_t visit(hsa_agent_t agent, void* data) {
    auto& query = *static_cast<Query*>(data);
    hsa_device_type_t type;
    auto status = query.api.agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (status != HSA_STATUS_SUCCESS || type != HSA_DEVICE_TYPE_GPU) {
      return status;
    }
    char uuid[21] = {};
    status = query.api.agent_get_info(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID), uuid);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    if (std::memcmp(uuid, "GPU-", 4) != 0 ||
        std::memcmp(uuid + 4, query.uuid, 16) != 0) {
      return HSA_STATUS_SUCCESS;
    }
    if (query.found) {
      // An ambiguous identity must not borrow another partition's cache size.
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    uint32_t sizes[4] = {};
    status = query.api.agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE, sizes);
    query.found = true;
    // L1..L4, in bytes, aggregated per agent. Prefer level over capacity.
    query.size = sizes[3] ? sizes[3] : sizes[2];
    return status;
  }

  int64_t query(const char* uuid) const {
    if (!api_.init || !api_.shut_down || !api_.iterate_agents ||
        !api_.agent_get_info || api_.init() != HSA_STATUS_SUCCESS) {
      return 0;
    }
    Query query{api_, uuid};
    const auto status = api_.iterate_agents(visit, &query);
    // HSA initialization is reference-counted. Release only our reference.
    api_.shut_down();
    return status == HSA_STATUS_SUCCESS ? query.size : 0;
  }

  HsaCacheApi api_;
  std::mutex mutex_;
  std::unordered_map<std::string, int64_t> sizes_;
};

inline int64_t getLastLevelCacheSize(const char* uuid) {
  static LastLevelCache cache(loadHsaCacheApi());
  return cache.get(uuid);
}

} // namespace at::cuda::detail
#else
namespace at::cuda::detail {
inline int64_t getLastLevelCacheSize(const char*) {
  return 0;
}
} // namespace at::cuda::detail
#endif
