#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADevice.h>

TEST(CudaDeviceTest, getDeviceFromPtr_fails_with_host_memory) {
  if (!at::cuda::is_available()) {
    return;
  }

  int dummy = 0;

  ASSERT_THROW(at::cuda::getDeviceFromPtr(&dummy), c10::Error);
}

#if defined(USE_ROCM) && !defined(_WIN32) && __has_include(<hsa/hsa_ext_amd.h>)
#include <ATen/cuda/detail/LastLevelCache.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using at::cuda::detail::HsaCacheApi;
using at::cuda::detail::LastLevelCache;

constexpr uint32_t kMiB = 1024 * 1024;

std::array<char, 16> rawUuid(const char* text) {
  std::array<char, 16> uuid;
  std::memcpy(uuid.data(), text, uuid.size());
  return uuid;
}

// Deliberately omit a NUL terminator, as in hipUUID::bytes.
const auto kUuidA = rawUuid("0123456789abcdef");
const auto kUuidB = rawUuid("fedcba9876543210");
const auto kUnknownUuid = rawUuid("1111111111111111");

struct FakeHsaAgent {
  uint64_t handle;
  std::string uuid;
  std::array<uint32_t, 4> sizes;
  hsa_device_type_t type = HSA_DEVICE_TYPE_GPU;
  hsa_status_t device_status = HSA_STATUS_SUCCESS;
  hsa_status_t uuid_status = HSA_STATUS_SUCCESS;
  hsa_status_t cache_status = HSA_STATUS_SUCCESS;
};

struct FakeHsa {
  static FakeHsa* active;

  std::vector<FakeHsaAgent> agents;
  hsa_status_t init_status = HSA_STATUS_SUCCESS;
  hsa_status_t iteration_status = HSA_STATUS_SUCCESS;
  std::atomic<int> init_calls{0};
  std::atomic<int> shutdown_calls{0};
  std::atomic<int> iteration_calls{0};
  std::atomic<int> device_calls{0};
  std::atomic<int> uuid_calls{0};
  std::atomic<int> cache_calls{0};
  // Model an existing HIP consumer whose reference must survive our queries.
  std::atomic<int> references{1};

  void add(const std::array<char, 16>& uuid, uint32_t l3) {
    agents.push_back(
        {agents.size() + 1,
         "GPU-" + std::string(uuid.data(), uuid.size()),
         {32 * 1024, 8 * kMiB, l3, 0}});
  }

  std::array<int, 6> calls() const {
    return {
        init_calls.load(),
        shutdown_calls.load(),
        iteration_calls.load(),
        device_calls.load(),
        uuid_calls.load(),
        cache_calls.load()};
  }

  static hsa_status_t init() {
    auto& mock = *active;
    ++mock.init_calls;
    if (mock.init_status == HSA_STATUS_SUCCESS) {
      ++mock.references;
    }
    return mock.init_status;
  }

  static hsa_status_t shutDown() {
    ++active->shutdown_calls;
    --active->references;
    return HSA_STATUS_SUCCESS;
  }

  static hsa_status_t iterate(
      hsa_status_t (*callback)(hsa_agent_t, void*),
      void* data) {
    ++active->iteration_calls;
    for (const auto& agent : active->agents) {
      auto status = callback(hsa_agent_t{agent.handle}, data);
      if (status != HSA_STATUS_SUCCESS) {
        return status;
      }
    }
    // Allow an error after callbacks to test rejection of partial results.
    return active->iteration_status;
  }

  static hsa_status_t getInfo(
      hsa_agent_t handle,
      hsa_agent_info_t attribute,
      void* value) {
    auto& mock = *active;
    auto it = std::find_if(
        mock.agents.begin(), mock.agents.end(), [&](const auto& agent) {
          return agent.handle == handle.handle;
        });
    if (it == mock.agents.end()) {
      return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    switch (static_cast<int>(attribute)) {
      case HSA_AGENT_INFO_DEVICE:
        ++mock.device_calls;
        *static_cast<hsa_device_type_t*>(value) = it->type;
        return it->device_status;
      case HSA_AMD_AGENT_INFO_UUID:
        ++mock.uuid_calls;
        std::memcpy(value, it->uuid.c_str(), it->uuid.size() + 1);
        return it->uuid_status;
      case HSA_AGENT_INFO_CACHE_SIZE:
        ++mock.cache_calls;
        std::memcpy(value, it->sizes.data(), sizeof(it->sizes));
        return it->cache_status;
      default:
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  static HsaCacheApi api() {
    return {init, shutDown, iterate, getInfo};
  }
};

FakeHsa* FakeHsa::active = nullptr;

class LastLevelCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FakeHsa::active = &hsa;
  }

  void TearDown() override {
    EXPECT_EQ(hsa.references.load(), 1);
    FakeHsa::active = nullptr;
  }

  FakeHsa hsa;
};

TEST_F(LastLevelCacheTest, reports_bytes_for_distinct_reordered_devices) {
  hsa.add(kUuidA, 64 * kMiB);
  hsa.add(kUuidB, 8 * kMiB);
  std::reverse(hsa.agents.begin(), hsa.agents.end());
  LastLevelCache cache(FakeHsa::api());

  EXPECT_EQ(cache.get(kUuidA.data()), 64 * kMiB);
  EXPECT_EQ(cache.get(kUuidB.data()), 8 * kMiB);
  EXPECT_EQ(hsa.cache_calls.load(), 2);
  EXPECT_EQ(hsa.init_calls.load(), 2);
  EXPECT_EQ(hsa.shutdown_calls.load(), 2);
}

TEST_F(LastLevelCacheTest, returns_l3_bytes_when_l4_is_absent) {
  for (uint32_t l3 : {0u, 1u, std::numeric_limits<uint32_t>::max()}) {
    SCOPED_TRACE(l3);
    hsa.agents.clear();
    hsa.add(kUuidA, l3);
    hsa.agents[0].sizes[3] = 0;
    LastLevelCache cache(FakeHsa::api());
    EXPECT_EQ(cache.get(kUuidA.data()), static_cast<int64_t>(l3));
  }
}

TEST_F(LastLevelCacheTest, prefers_outermost_level_over_larger_capacity) {
  hsa.add(kUuidA, 64 * kMiB);
  hsa.agents[0].sizes[3] = 4 * kMiB;
  LastLevelCache cache(FakeHsa::api());

  // L4 wins even when it is smaller than both L3 and L2.
  EXPECT_EQ(cache.get(kUuidA.data()), 4 * kMiB);
}

TEST_F(LastLevelCacheTest, ignores_cpu_agents_even_with_matching_uuid) {
  hsa.add(kUuidA, 128 * kMiB);
  hsa.agents[0].type = HSA_DEVICE_TYPE_CPU;
  hsa.agents[0].uuid_status = HSA_STATUS_ERROR;
  hsa.agents[0].cache_status = HSA_STATUS_ERROR;
  hsa.add(kUuidA, 64 * kMiB);
  LastLevelCache cache(FakeHsa::api());

  EXPECT_EQ(cache.get(kUuidA.data()), 64 * kMiB);
  EXPECT_EQ(hsa.uuid_calls.load(), 1);
  EXPECT_EQ(hsa.cache_calls.load(), 1);
}

TEST_F(LastLevelCacheTest, rejects_duplicate_uuid_even_when_sizes_agree) {
  for (uint32_t second_size : {8 * kMiB, 64 * kMiB}) {
    SCOPED_TRACE(second_size);
    hsa.agents.clear();
    hsa.add(kUuidA, 64 * kMiB);
    hsa.add(kUuidA, second_size);
    LastLevelCache cache(FakeHsa::api());
    EXPECT_EQ(cache.get(kUuidA.data()), 0);
  }
}

TEST_F(LastLevelCacheTest, rejects_invalid_and_unsupported_uuid) {
  auto placeholder = std::array<char, 16>{};
  placeholder[0] = 'X';
  placeholder[1] = 'X';
  for (const auto& uuid :
       {rawUuid("0000000000000000"),
        rawUuid("0123456789abcdeg"),
        std::array<char, 16>{},
        placeholder}) {
    LastLevelCache cache(FakeHsa::api());
    EXPECT_EQ(cache.get(uuid.data()), 0);
  }
  EXPECT_EQ(hsa.init_calls.load(), 0);
}

TEST_F(LastLevelCacheTest, requires_gpu_prefix_in_reported_uuid) {
  hsa.add(kUuidA, 64 * kMiB);
  hsa.agents[0].uuid.replace(0, 4, "CPU-");
  LastLevelCache cache(FakeHsa::api());
  EXPECT_EQ(cache.get(kUuidA.data()), 0);
  EXPECT_EQ(hsa.cache_calls.load(), 0);
}

TEST_F(LastLevelCacheTest, missing_functions_do_not_initialize_runtime) {
  for (int missing = 0; missing < 4; ++missing) {
    SCOPED_TRACE(missing);
    auto api = FakeHsa::api();
    switch (missing) {
      case 0:
        api.init = nullptr;
        break;
      case 1:
        api.shut_down = nullptr;
        break;
      case 2:
        api.iterate_agents = nullptr;
        break;
      case 3:
        api.agent_get_info = nullptr;
        break;
    }
    LastLevelCache cache(api);
    EXPECT_EQ(cache.get(kUuidA.data()), 0);
  }
  EXPECT_EQ(hsa.init_calls.load(), 0);
  EXPECT_EQ(hsa.shutdown_calls.load(), 0);
}

TEST_F(LastLevelCacheTest, failed_init_is_cached_without_shutdown) {
  hsa.init_status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  LastLevelCache cache(FakeHsa::api());
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(cache.get(kUuidA.data()), 0);
  }
  EXPECT_EQ(hsa.init_calls.load(), 1);
  EXPECT_EQ(hsa.iteration_calls.load(), 0);
  EXPECT_EQ(hsa.shutdown_calls.load(), 0);
}

TEST_F(
    LastLevelCacheTest,
    failed_agent_queries_discard_results_and_release_reference) {
  for (int failed = 0; failed < 3; ++failed) {
    SCOPED_TRACE(failed);
    hsa.agents.clear();
    hsa.add(kUuidA, 64 * kMiB);
    auto& agent = hsa.agents[0];
    switch (failed) {
      case 0:
        agent.device_status = HSA_STATUS_ERROR;
        break;
      case 1:
        agent.uuid_status = HSA_STATUS_ERROR;
        break;
      case 2:
        agent.cache_status = HSA_STATUS_ERROR;
        break;
    }
    LastLevelCache cache(FakeHsa::api());
    EXPECT_EQ(cache.get(kUuidA.data()), 0);
    const auto calls = hsa.calls();
    EXPECT_EQ(cache.get(kUuidA.data()), 0);
    EXPECT_EQ(hsa.calls(), calls);
    EXPECT_EQ(hsa.references.load(), 1);
  }
  EXPECT_EQ(hsa.init_calls.load(), 3);
  EXPECT_EQ(hsa.shutdown_calls.load(), 3);
}

TEST_F(LastLevelCacheTest, enumeration_failure_discards_already_matched_size) {
  hsa.add(kUuidA, 64 * kMiB);
  hsa.iteration_status = HSA_STATUS_ERROR;
  LastLevelCache cache(FakeHsa::api());
  EXPECT_EQ(cache.get(kUuidA.data()), 0);
  EXPECT_EQ(hsa.cache_calls.load(), 1);
  EXPECT_EQ(hsa.shutdown_calls.load(), 1);
}

TEST_F(LastLevelCacheTest, caches_positive_absent_and_unknown_results) {
  hsa.add(kUuidA, 64 * kMiB);
  hsa.add(kUuidB, 0);
  LastLevelCache cache(FakeHsa::api());
  EXPECT_EQ(cache.get(kUuidA.data()), 64 * kMiB);
  EXPECT_EQ(cache.get(kUuidB.data()), 0);
  EXPECT_EQ(cache.get(kUnknownUuid.data()), 0);
  const auto calls = hsa.calls();
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(cache.get(kUuidA.data()), 64 * kMiB);
    EXPECT_EQ(cache.get(kUuidB.data()), 0);
    EXPECT_EQ(cache.get(kUnknownUuid.data()), 0);
  }
  EXPECT_EQ(hsa.calls(), calls);
  EXPECT_EQ(hsa.init_calls.load(), 3);
  EXPECT_EQ(hsa.shutdown_calls.load(), 3);
}

TEST_F(LastLevelCacheTest, concurrent_first_reads_query_each_device_once) {
  hsa.add(kUuidA, 64 * kMiB);
  hsa.add(kUuidB, 0);
  LastLevelCache cache(FakeHsa::api());
  constexpr int thread_count = 8;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([&, i] {
      ++ready;
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (int j = 0; j < 1000; ++j) {
        // Contend on both positive and negative entries, in opposite orders.
        const bool positive = (i + j) % 2 == 0;
        const auto& uuid = positive ? kUuidA : kUuidB;
        if (cache.get(uuid.data()) != (positive ? 64 * kMiB : 0)) {
          ++failures;
        }
      }
    });
  }
  while (ready.load() != thread_count) {
    std::this_thread::yield();
  }
  start = true;
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(hsa.init_calls.load(), 2);
  EXPECT_EQ(hsa.shutdown_calls.load(), 2);
  EXPECT_EQ(hsa.iteration_calls.load(), 2);
  EXPECT_EQ(hsa.cache_calls.load(), 2);
}

} // namespace
#endif
