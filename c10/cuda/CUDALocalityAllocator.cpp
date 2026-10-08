#include <c10/cuda/CUDALocalityAllocator.h>

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>

namespace c10::cuda::LocalityAllocator {

namespace {

// 64 GiB of VA per device; physical pages are mapped as the bump pointer grows.
constexpr size_t kArenaBytes = size_t(64) << 30;

struct Arena {
  std::mutex mutex;
  CUdeviceptr base = 0;
  size_t top = 0; // bytes handed out by the bump pointer; pages below are mapped
  std::multimap<size_t, uintptr_t> free_blocks; // size -> base
  std::unordered_map<uintptr_t, size_t> live; // base -> size
  // Published for the lock-free contains() check; end == 0 until reserved.
  std::atomic<uintptr_t> begin{0}, end{0};
};

std::array<Arena, C10_COMPILE_TIME_MAX_GPUS>& arenas() {
  static std::array<Arena, C10_COMPILE_TIME_MAX_GPUS> instance;
  return instance;
}

size_t round_up(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN first appears in the CUDA 13.4
// headers this prototype was built with.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13040
// Maps [base + begin, base + end) with 2 MiB pages, page P on locality domain
// P & 1. On failure, unmaps what it mapped and returns the driver status.
CUresult map_pages(CUdeviceptr base, int device, size_t begin, size_t end) {
  auto* driver = DriverAPI::get();
  CUresult status = CUDA_SUCCESS;
  size_t offset = begin;
  for (; offset < end && status == CUDA_SUCCESS; offset += kPageBytes) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    prop.location.localized.deviceId = static_cast<unsigned char>(device);
    prop.location.localized.localityDomainId =
        static_cast<unsigned char>(((base + offset) >> 21) & 1);
    CUmemGenericAllocationHandle handle = 0;
    status = driver->cuMemCreate_(&handle, kPageBytes, &prop, 0);
    if (status == CUDA_SUCCESS) {
      status = driver->cuMemMap_(base + offset, kPageBytes, 0, handle, 0);
      // The mapping, if any, keeps the memory alive.
      C10_CUDA_DRIVER_CHECK(driver->cuMemRelease_(handle));
    }
  }
  if (status != CUDA_SUCCESS) {
    const size_t mapped = offset - kPageBytes - begin; // pages before the failed one
    if (mapped > 0) {
      C10_CUDA_DRIVER_CHECK(driver->cuMemUnmap_(base + begin, mapped));
    }
    return status;
  }
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(
      driver->cuMemSetAccess_(base + begin, end - begin, &access, 1));
  return CUDA_SUCCESS;
}

// Grows the arena by `bytes` at `arena.top`. False when the device is out of
// memory; other driver errors throw. The arena is unchanged on failure.
bool grow(Arena& arena, int device, size_t bytes) {
  const CUresult status = map_pages(arena.base, device, arena.top, arena.top + bytes);
  if (status == CUDA_ERROR_OUT_OF_MEMORY) {
    return false;
  }
  C10_CUDA_DRIVER_CHECK(status);
  arena.top += bytes;
  return true;
}

void reserve(Arena& arena, int device) {
  auto* driver = DriverAPI::get();
  CUdevice cu_device = 0;
  C10_CUDA_DRIVER_CHECK(driver->cuDeviceGet_(&cu_device, device));
  int domains = 0;
  C10_CUDA_DRIVER_CHECK(driver->cuDeviceGetAttribute_(
      &domains, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device));
  TORCH_CHECK(
      domains == 2,
      "LocalityInterleavedAllocator needs a device with 2 locality domains; device ",
      device,
      " has ",
      domains);
  size_t granularity = 0;
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  prop.location.localized.deviceId = static_cast<unsigned char>(device);
  C10_CUDA_DRIVER_CHECK(driver->cuMemGetAllocationGranularity_(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  TORCH_CHECK(
      kPageBytes % granularity == 0,
      "locality-domain allocation granularity ",
      granularity,
      " does not divide 2 MiB");
  // 4 MiB alignment makes page parity equal to address bit 21.
  CUdeviceptr base = 0;
  C10_CUDA_DRIVER_CHECK(
      driver->cuMemAddressReserve_(&base, kArenaBytes, kBlockAlign, 0, 0));
  // The two probe pages (sides 0 and 1) used by the SM side map.
  const CUresult status = map_pages(base, device, 0, kBlockAlign);
  if (status != CUDA_SUCCESS) {
    C10_CUDA_DRIVER_CHECK(driver->cuMemAddressFree_(base, kArenaBytes));
    C10_CUDA_DRIVER_CHECK(status);
  }
  arena.base = base;
  arena.top = kBlockAlign;
  arena.begin.store(arena.base);
  arena.end.store(arena.base + kArenaBytes);
}
#else
bool grow(Arena&, int, size_t) {
  return false;
}
void reserve(Arena&, int) {
  TORCH_CHECK(false, "LocalityInterleavedAllocator needs CUDA >= 13.4");
}
#endif

} // namespace

void* raw_alloc(size_t size, int device, cudaStream_t /*stream*/) {
  if (size == 0) {
    return nullptr;
  }
  TORCH_CHECK(device >= 0 && device < C10_COMPILE_TIME_MAX_GPUS);
  Arena& arena = arenas()[device];
  std::lock_guard<std::mutex> lock(arena.mutex);
  CUDAGuard guard(static_cast<DeviceIndex>(device));
  if (arena.base == 0) {
    reserve(arena, device);
  }
  const size_t bytes = round_up(size, kBlockAlign);
  uintptr_t block = 0;
  if (auto it = arena.free_blocks.lower_bound(bytes);
      it != arena.free_blocks.end()) {
    // Best fit; the remainder stays 4 MiB aligned and returns to the free list.
    block = it->second;
    const size_t remainder = it->first - bytes;
    arena.free_blocks.erase(it);
    if (remainder > 0) {
      arena.free_blocks.emplace(remainder, block + bytes);
    }
  } else {
    block = arena.base + arena.top;
    if (bytes > kArenaBytes - arena.top || !grow(arena, device, bytes)) {
      return nullptr; // the caching allocator reports OOM
    }
  }
  arena.live.emplace(block, bytes);
  return reinterpret_cast<void*>(block);
}

void raw_free(void* ptr, size_t /*size*/, int device, cudaStream_t /*stream*/) {
  if (ptr == nullptr) {
    return;
  }
  Arena& arena = arenas()[device];
  std::lock_guard<std::mutex> lock(arena.mutex);
  auto it = arena.live.find(reinterpret_cast<uintptr_t>(ptr));
  TORCH_CHECK(
      it != arena.live.end(),
      "LocalityInterleavedAllocator: freeing an unknown pointer");
  arena.free_blocks.emplace(it->second, it->first);
  arena.live.erase(it);
}

bool contains(int device, const void* ptr, size_t bytes) {
  if (device < 0 || device >= C10_COMPILE_TIME_MAX_GPUS) {
    return false;
  }
  const Arena& arena = arenas()[device];
  const uintptr_t begin = arena.begin.load(std::memory_order_relaxed);
  const uintptr_t end = arena.end.load(std::memory_order_relaxed);
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  return end != 0 && addr >= begin && addr < end && bytes <= end - addr;
}

void* arena_base(int device) {
  if (device < 0 || device >= C10_COMPILE_TIME_MAX_GPUS) {
    return nullptr;
  }
  return reinterpret_cast<void*>(arenas()[device].begin.load());
}

} // namespace c10::cuda::LocalityAllocator
