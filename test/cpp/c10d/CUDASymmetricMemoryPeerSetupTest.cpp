#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/driver_api.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <gtest/gtest.h>
#include "StoreTestCommon.hpp"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace c10d::symmetric_memory {
namespace {

// Preserve history and reference counts across cleanup and numeric-ID reuse.
template <typename Key>
class ResourceLog {
 public:
  void acquire(Key key) {
    ++entries_[key].acquired;
  }

  void released(Key key, CUresult result) {
    auto& entry = entries_[key];
    ++entry.attempted;
    if (result == CUDA_SUCCESS) {
      ++entry.succeeded;
    } else {
      ++entry.errors;
    }
  }

  size_t live() const {
    size_t result = 0;
    for (const auto& [key, entry] : entries_) {
      result += entry.acquired - std::min(entry.acquired, entry.succeeded);
    }
    return result;
  }

  bool clean() const {
    for (const auto& [key, entry] : entries_) {
      if (entry.acquired != entry.attempted ||
          entry.acquired != entry.succeeded || entry.errors != 0) {
        return false;
      }
    }
    return true;
  }

 private:
  struct Entry {
    size_t acquired = 0;
    size_t attempted = 0;
    size_t succeeded = 0;
    size_t errors = 0;
  };
  std::map<Key, Entry> entries_;
};

TEST(PeerResourceLogTest, DetectsDuplicateFailedCleanup) {
  ResourceLog<int> resources;
  resources.acquire(7);
  resources.released(7, CUDA_SUCCESS);
  EXPECT_TRUE(resources.clean());
  resources.released(7, CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(resources.live(), 0);
  EXPECT_FALSE(resources.clean());
}

TEST(PeerResourceLogTest, CountsAliasedReferencesAndReusedIdentities) {
  ResourceLog<int> resources;
  resources.acquire(7);
  resources.acquire(7);
  resources.released(7, CUDA_SUCCESS);
  EXPECT_EQ(resources.live(), 1);
  EXPECT_FALSE(resources.clean());
  resources.released(7, CUDA_SUCCESS);
  EXPECT_TRUE(resources.clean());
  resources.acquire(7);
  resources.released(7, CUDA_SUCCESS);
  EXPECT_TRUE(resources.clean());
}

std::vector<int> rollback_calls;

CUresult CUDAAPI record_unmap(CUdeviceptr ptr, size_t size) {
  EXPECT_EQ(ptr, 4096);
  EXPECT_EQ(size, 4096);
  rollback_calls.push_back(1);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI record_free(CUdeviceptr ptr, size_t size) {
  EXPECT_EQ(ptr, 4096);
  EXPECT_EQ(size, 4096);
  rollback_calls.push_back(2);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI record_release(CUmemGenericAllocationHandle handle) {
  EXPECT_EQ(handle, 7);
  rollback_calls.push_back(3);
  return CUDA_SUCCESS;
}

TEST(PeerRollbackTest, OwnerFailureAfterMappingCleansInOrder) {
  rollback_calls.clear();
  rollback_calls.reserve(3);
  c10::cuda::DriverAPI driver{};
  driver.cuMemUnmap_ = record_unmap;
  driver.cuMemAddressFree_ = record_free;
  driver.cuMemRelease_ = record_release;
  void* ptr = nullptr;
  const auto owner_error = std::make_exception_ptr(std::bad_alloc());
  EXPECT_THROW(
      map_and_own_peer(
          [&] { ptr = reinterpret_cast<void*>(4096); },
          [&]() -> std::unique_ptr<int> {
            EXPECT_NE(ptr, nullptr);
            std::rethrow_exception(owner_error);
          },
          [&]() noexcept { release_unowned_peer(ptr, 7, 4096, &driver); }),
      std::bad_alloc);
  EXPECT_EQ(rollback_calls, (std::vector<int>{1, 2, 3}));
}

TEST(PeerRollbackTest, UnpublishedMappingOnlyReleasesImport) {
  rollback_calls.clear();
  rollback_calls.reserve(3);
  c10::cuda::DriverAPI driver{};
  driver.cuMemUnmap_ = record_unmap;
  driver.cuMemAddressFree_ = record_free;
  driver.cuMemRelease_ = record_release;
  release_unowned_peer(nullptr, 7, 4096, &driver);
  EXPECT_EQ(rollback_calls, (std::vector<int>{3}));
}

enum class Fault { None, Import, Map, Owner };
struct Scenario {
  Fault fault;
  int world_size;
  int preceding_peers;
};

void PrintTo(const Scenario& scenario, std::ostream* out) {
  *out << "world_size=" << scenario.world_size
       << ", fault=" << static_cast<int>(scenario.fault)
       << ", preceding_peers=" << scenario.preceding_peers;
}
auto* api = static_cast<c10::cuda::DriverAPI*>(nullptr);
decltype(api->cuMemCreate_) real_create;
decltype(api->cuMemImportFromShareableHandle_) real_import;
decltype(api->cuMemMap_) real_map;
decltype(api->cuMemUnmap_) real_unmap;
decltype(api->cuMemRelease_) real_release;
decltype(api->cuMemAddressReserve_) real_reserve;
decltype(api->cuMemAddressFree_) real_free;
Fault fault;
bool injected;
int preceding_peers;
int peer_imports = 0;
int peer_maps = 0;
ResourceLog<CUmemGenericAllocationHandle> handles;
ResourceLog<CUdeviceptr> reservations;
ResourceLog<CUdeviceptr> mappings;
std::map<CUmemGenericAllocationHandle, size_t> imports;
struct ReceivedFd {
  int fd;
  struct stat identity;
};
std::vector<ReceivedFd> received_fds;

CUresult CUDAAPI create_allocation(
    CUmemGenericAllocationHandle* handle,
    size_t size,
    const CUmemAllocationProp* properties,
    unsigned long long flags) {
  auto result = real_create(handle, size, properties, flags);
  if (result == CUDA_SUCCESS) {
    handles.acquire(*handle);
  }
  return result;
}

CUresult CUDAAPI import_peer(
    CUmemGenericAllocationHandle* handle,
    void* token,
    CUmemAllocationHandleType type) {
  EXPECT_EQ(type, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  ReceivedFd received{static_cast<int>(reinterpret_cast<uintptr_t>(token)), {}};
  EXPECT_EQ(fstat(received.fd, &received.identity), 0);
  received_fds.push_back(received);
  if (fault == Fault::Import && !injected && peer_imports == preceding_peers) {
    injected = true;
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto result = real_import(handle, token, type);
  if (result == CUDA_SUCCESS) {
    handles.acquire(*handle);
    ++imports[*handle];
    ++peer_imports;
  }
  return result;
}

CUresult CUDAAPI map_peer(
    CUdeviceptr ptr,
    size_t size,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    unsigned long long flags) {
  const bool peer = imports.contains(handle);
  if (fault == Fault::Map && !injected && peer &&
      peer_maps == preceding_peers) {
    injected = true;
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto result = real_map(ptr, size, offset, handle, flags);
  if (result == CUDA_SUCCESS) {
    mappings.acquire(ptr);
    peer_maps += peer;
  }
  return result;
}

CUresult CUDAAPI unmap_peer(CUdeviceptr ptr, size_t size) {
  auto result = real_unmap(ptr, size);
  mappings.released(ptr, result);
  return result;
}

CUresult CUDAAPI release_peer(CUmemGenericAllocationHandle handle) {
  auto result = real_release(handle);
  handles.released(handle, result);
  return result;
}

CUresult CUDAAPI reserve_peer(
    CUdeviceptr* ptr,
    size_t size,
    size_t alignment,
    CUdeviceptr address,
    unsigned long long flags) {
  auto result = real_reserve(ptr, size, alignment, address, flags);
  if (result == CUDA_SUCCESS) {
    reservations.acquire(*ptr);
  }
  return result;
}

CUresult CUDAAPI free_peer(CUdeviceptr ptr, size_t size) {
  auto result = real_free(ptr, size);
  reservations.released(ptr, result);
  return result;
}

void expect_received_fds_closed() {
  for (const auto& received : received_fds) {
    struct stat current{};
    if (fstat(received.fd, &current) == -1) {
      EXPECT_EQ(errno, EBADF);
    } else {
      // Numeric FD reuse by a different file does not mean the old FD leaked.
      EXPECT_TRUE(
          current.st_dev != received.identity.st_dev ||
          current.st_ino != received.identity.st_ino ||
          current.st_rdev != received.identity.st_rdev ||
          current.st_mode != received.identity.st_mode);
    }
  }
  received_fds.clear();
}

void expect_clean() {
  EXPECT_EQ(handles.live(), 0);
  EXPECT_EQ(reservations.live(), 0);
  EXPECT_EQ(mappings.live(), 0);
  EXPECT_TRUE(handles.clean());
  EXPECT_TRUE(reservations.clean());
  EXPECT_TRUE(mappings.clean());
}

class DriverHooks {
 public:
  DriverHooks() {
    api = c10::cuda::DriverAPI::get();
    real_create = api->cuMemCreate_;
    real_import = api->cuMemImportFromShareableHandle_;
    real_map = api->cuMemMap_;
    real_unmap = api->cuMemUnmap_;
    real_release = api->cuMemRelease_;
    real_reserve = api->cuMemAddressReserve_;
    real_free = api->cuMemAddressFree_;
    api->cuMemCreate_ = create_allocation;
    api->cuMemImportFromShareableHandle_ = import_peer;
    api->cuMemMap_ = map_peer;
    api->cuMemUnmap_ = unmap_peer;
    api->cuMemRelease_ = release_peer;
    api->cuMemAddressReserve_ = reserve_peer;
    api->cuMemAddressFree_ = free_peer;
  }
  ~DriverHooks() {
    api->cuMemCreate_ = real_create;
    api->cuMemImportFromShareableHandle_ = real_import;
    api->cuMemMap_ = real_map;
    api->cuMemUnmap_ = real_unmap;
    api->cuMemRelease_ = real_release;
    api->cuMemAddressReserve_ = real_reserve;
    api->cuMemAddressFree_ = real_free;
  }
};

void check_peer_data(
    const c10::intrusive_ptr<SymmetricMemory>& handle,
    int rank,
    int world_size) {
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer != rank) {
      EXPECT_TRUE(handle->get_buffer(peer, {32}, at::kFloat, 0)
                      .cpu()
                      .eq(peer + 1)
                      .all()
                      .item<bool>());
    }
  }
}

// Compose the production handoff, rollback, and agreement with a real map.
// The fixture mapper below does not call the private, unexported map_block.
// This is not an injected end-to-end rendezvous failure.
void check_owner_failure(
    const c10::intrusive_ptr<FileStore>& store,
    int rank,
    int device) {
  CUmemAllocationProp properties{};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;
  properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t size = 0;
  C10_CUDA_DRIVER_CHECK(api->cuMemGetAllocationGranularity_(
      &size, &properties, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  CUmemGenericAllocationHandle backing{};
  C10_CUDA_DRIVER_CHECK(api->cuMemCreate_(&backing, size, &properties, 0));
  auto release_backing = c10::make_scope_exit(
      [&]() noexcept { EXPECT_EQ(api->cuMemRelease_(backing), CUDA_SUCCESS); });
  int fd = -1;
  C10_CUDA_DRIVER_CHECK(api->cuMemExportToShareableHandle_(
      &fd, backing, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  auto close_fd =
      c10::make_scope_exit([&]() noexcept { EXPECT_EQ(close(fd), 0); });
  CUmemGenericAllocationHandle imported{};
  C10_CUDA_DRIVER_CHECK(api->cuMemImportFromShareableHandle_(
      &imported,
      reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  void* ptr = nullptr;
  struct PeerOwner {
    void* ptr;
    CUmemGenericAllocationHandle handle;
    size_t size;
    ~PeerOwner() {
      release_unowned_peer(ptr, handle, size, api);
    }
  };
  std::unique_ptr<PeerOwner> owner;
  std::exception_ptr local_error;
  const auto owner_error = std::make_exception_ptr(std::bad_alloc());
  try {
    owner = map_and_own_peer(
        [&]() {
          CUdeviceptr address = 0;
          C10_CUDA_DRIVER_CHECK(
              api->cuMemAddressReserve_(&address, size, 0, 0, 0));
          bool mapped = false;
          auto rollback = c10::make_scope_exit([&]() noexcept {
            const auto result =
                mapped ? api->cuMemUnmap_(address, size) : CUDA_SUCCESS;
            EXPECT_EQ(result, CUDA_SUCCESS);
            if (result == CUDA_SUCCESS) {
              EXPECT_EQ(api->cuMemAddressFree_(address, size), CUDA_SUCCESS);
            }
          });
          C10_CUDA_DRIVER_CHECK(api->cuMemMap_(address, size, 0, imported, 0));
          mapped = true;
          CUmemAccessDesc access{};
          access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
          access.location.id = device;
          access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          C10_CUDA_DRIVER_CHECK(
              api->cuMemSetAccess_(address, size, &access, 1));
          ptr = reinterpret_cast<void*>(
              address); // NOLINT(performance-no-int-to-ptr)
          rollback.release();
        },
        [&]() {
          EXPECT_NE(ptr, nullptr);
          EXPECT_EQ(mappings.live(), 1);
          if (rank == 0) {
            injected = true;
            std::rethrow_exception(owner_error);
          }
          return std::unique_ptr<PeerOwner>(new PeerOwner{ptr, imported, size});
        },
        [&]() noexcept { release_unowned_peer(ptr, imported, size, api); });
  } catch (...) {
    local_error = std::current_exception();
  }
  StoreExchange exchange("owner-failure");
  bool observed_failure = false;
  try {
    observed_failure = exchange.barrier_with_error(store, 2, local_error);
    EXPECT_NE(rank, 0);
  } catch (const std::bad_alloc&) {
    EXPECT_EQ(rank, 0);
    observed_failure = true;
  }
  EXPECT_TRUE(observed_failure);
  EXPECT_EQ(injected, rank == 0);
  owner.reset();
  EXPECT_EQ(mappings.live(), 0);
  EXPECT_EQ(reservations.live(), 0);
  EXPECT_EQ(handles.live(), 1);
  store->barrier("owner-cleaned", 2);
}

int runRank(const std::string& path, int rank, Scenario scenario) {
  alarm(90);
  const auto world_size = scenario.world_size;
  auto store = c10::make_intrusive<FileStore>(path, world_size);
  store->setTimeout(std::chrono::seconds(30));
  bool supported = at::cuda::device_count() >= 2;
  const int device = rank % 2;
  if (supported) {
    supported = !at::cuda::get_fabric_access(device) &&
        at::cuda::get_p2p_access(device, 1 - device);
  }
  store->add("unsupported", supported ? 0 : 1);
  store->barrier("capability", world_size);
  if (store->add("unsupported", 0) != 0) {
    return 77;
  }
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  DriverHooks hooks;
  fault = rank == 0 ? scenario.fault : Fault::None;
  preceding_peers = scenario.preceding_peers;
  injected = false;
  if (scenario.fault == Fault::Owner) {
    check_owner_failure(store, rank, device);
    expect_received_fds_closed();
    expect_clean();
    return ::testing::Test::HasFailure() ? 1 : 0;
  }
  auto group = c10::make_intrusive<ProcessGroup>(store, rank, world_size);
  const std::string name = "peer-setup-test";
  register_process_group(name, group);
  auto unregister =
      c10::make_scope_exit([&]() noexcept { unregister_process_group(name); });
  if (get_backend(c10::Device(c10::kCUDA, device)) != "CUDA") {
    set_backend("CUDA");
  }
  auto make_tensor = [&]() {
    auto tensor = empty_strided_p2p(
        {32},
        {1},
        at::kFloat,
        c10::Device(c10::kCUDA, device),
        name,
        std::nullopt);
    tensor.fill_(rank + 1);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    return tensor;
  };
  auto tensor = make_tensor();
  bool failed = false;
  std::string error_message;
  c10::intrusive_ptr<SymmetricMemory> handle;
  try {
    handle = rendezvous(tensor, name);
  } catch (const c10::Error& error) {
    failed = true;
    error_message = error.what_without_backtrace();
  }
  expect_received_fds_closed();
  EXPECT_EQ(failed, scenario.fault != Fault::None);
  EXPECT_EQ(injected, rank == 0 && scenario.fault != Fault::None);
  if (scenario.fault != Fault::None) {
    const auto expected = rank == 0
        ? "CUDA driver error:"
        : "a peer failed to import, map, or own an allocation";
    EXPECT_NE(error_message.find(expected), std::string::npos);
    EXPECT_EQ(handles.live(), 1);
    EXPECT_EQ(reservations.live(), 1);
    EXPECT_EQ(mappings.live(), 1);
    if (rank == 0) {
      EXPECT_EQ(peer_maps, scenario.preceding_peers);
    }
  } else {
    EXPECT_EQ(handles.live(), world_size);
    EXPECT_EQ(reservations.live(), world_size);
    EXPECT_EQ(mappings.live(), world_size);
    check_peer_data(handle, rank, world_size);
  }
  store->barrier("observed", world_size);
  tensor.reset();
  handle.reset();
  expect_clean();
  if (scenario.fault != Fault::None) {
    // A fresh buffer diagnoses sequence alignment, not failed-tensor retry.
    fault = Fault::None;
    tensor = make_tensor();
    handle = rendezvous(tensor, name);
    expect_received_fds_closed();
    check_peer_data(handle, rank, world_size);
    store->barrier("retry-observed", world_size);
    tensor.reset();
    handle.reset();
    expect_clean();
  }
  store->barrier("cleaned", world_size);
  return ::testing::Test::HasFailure() ? 1 : 0;
}

class PeerSetupTest : public ::testing::TestWithParam<Scenario> {};

TEST_P(PeerSetupTest, ReleasesPeerResources) {
  c10d::test::TemporaryFile file;
  const auto scenario = GetParam();
  std::vector<pid_t> children(scenario.world_size, -1);
  for (int rank = 0; rank < scenario.world_size; ++rank) {
    children[rank] = fork();
    if (children[rank] == -1) {
      for (int previous = 0; previous < rank; ++previous) {
        kill(children[previous], SIGKILL);
        while (waitpid(children[previous], nullptr, 0) == -1 &&
               errno == EINTR) {
        }
      }
      FAIL() << "fork failed";
    }
    if (children[rank] == 0) {
      setenv("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES", "1", 1);
      setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False", 1);
      try {
        _exit(runRank(file.path, rank, scenario));
      } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        _exit(1);
      }
    }
  }
  int skipped = 0;
  for (auto child : children) {
    int status = 0;
    pid_t waited;
    do {
      waited = waitpid(child, &status, 0);
    } while (waited == -1 && errno == EINTR);
    EXPECT_EQ(waited, child);
    if (waited != child) {
      continue;
    }
    EXPECT_TRUE(WIFEXITED(status));
    if (!WIFEXITED(status)) {
      continue;
    }
    const auto code = WEXITSTATUS(status);
    EXPECT_TRUE(code == 0 || code == 77) << code;
    skipped += code == 77;
  }
  EXPECT_TRUE(skipped == 0 || skipped == scenario.world_size);
  if (skipped != 0 && !::testing::Test::HasFailure()) {
    GTEST_SKIP() << "Requires two P2P-capable GPUs without Fabric handles";
  }
}

INSTANTIATE_TEST_SUITE_P(
    CUDA,
    PeerSetupTest,
    ::testing::Values(
        Scenario{Fault::None, 2, 0},
        Scenario{Fault::Import, 2, 0},
        Scenario{Fault::Map, 2, 0},
        Scenario{Fault::None, 3, 0},
        Scenario{Fault::Import, 3, 0},
        Scenario{Fault::Import, 3, 1},
        Scenario{Fault::Map, 3, 1},
        Scenario{Fault::Owner, 2, 0}));

} // namespace
} // namespace c10d::symmetric_memory
