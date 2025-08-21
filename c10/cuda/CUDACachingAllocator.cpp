#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Gauge.h>
#include <c10/util/Logging.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/env.h>
#include <c10/util/error.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/static_tracepoint.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <regex>
#include <set>
#include <stack>
#include <thread>
#include <utility>
#include <vector>

TORCH_SDT_DEFINE_SEMAPHORE(malloc)
TORCH_SDT_DEFINE_SEMAPHORE(free)

// add these definitions so that we can compile with CUDA < 12.3
// borrowed from
// https://github.com/NVIDIA/nccl/blob/3ea7eedf3b9b94f1d9f99f4e55536dfcbd23c1ca/src/include/p2p.h#L20
#if CUDA_VERSION < 12030
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_IPC_HANDLE_SIZE 64
typedef struct CUmemFabricHandle_st {
  unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

namespace c10 {

C10_DEFINE_REGISTRY(FreeMemoryCallbacksRegistry, FreeMemoryCallback)

namespace cuda::CUDACachingAllocator {

using namespace c10::CachingAllocator;
using namespace c10::CachingDeviceAllocator;

namespace Native {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= max_split_size are not allowed
//   to be split. These oversize cached blocks will still satisfy requests
//   within 1MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

/**
 * Note [Interaction with CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Graph capture performs a dry run of a region of execution, freezing all CUDA
 * work (and virtual addresses used during that work) into a "graph." The graph
 * may be "replayed" like a single giant kernel, with greatly reduced CPU
 * overhead as well as modestly improved GPU performance.
 *
 * Because capture bakes in memory addresses, the memory used during capture
 * must be available for the graph to use during replay. DeviceCachingAllocator
 * assigns and frees memory eagerly and dynamically, so if we're not careful
 * about managing graphs' memory, at replay time those memory addresses could be
 * used by other tensors.
 *
 * To guarantee a graph's baked in addresses are safe to reuse in replay,
 * DeviceAllocator satisfies allocations from a graph-private memory pool during
 * capture, and doesn't begin cudaFreeing those addresses until the graph is
 * destroyed.
 *
 * Within the private pool, allocations are freed and reassigned as usual during
 * capture. Memory regions will be used in a consistent order during replay. So
 * a private pool doesn't use memory more wastefully than the default pools
 * during capture, but it does reserve its high-water mark of used memory away
 * from the default pools as long as the capture(s) it served survive
 * (regardless whether those captures are idle or replaying).
 *
 * CUDAGraph's requests for private pools are mediated by
 * DeviceAllocator::notifyCaptureBegin,
 *                  notifyCaptureAboutToEnd,
 *                  notifyCaptureEnded,
 *                  notifyCaptureDestroy.
 */

static char SHAREABLE_HANDLE_VERSION = 2;
enum ShareableHandleType : char {
  SHAREABLE_CUDA_MALLOC = 'c',
  SHAREABLE_CUDA_EXPANDABLE_SEGMENT = 'e'
};

namespace {

using stream_set = ska::flat_hash_set<cuda::CUDAStream>;

using Block = DeviceBlock<cuda::CUDAStream>;

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)

/*
Note [Expandable Segments]

Rationale

For large (>2MB) allocations, the allocator calls cudaMalloc to get allocations
that are the same size as what the user requests. In the future, parts of these
allocations can be reused for other requests if they are free. This works well
when the program makes many requests of exactly the same size or of sizes that
even multiples of that size. Many deep learning models follow this behavior.
However, one common exception is when the batch size changes slightly from one
iteration to the next, e.g. in batched inference. When the program runs
initially with batch size N, it will make allocations appropriate for that size.
If in the future, it runs at size N - 1, the existing allocations will still be
big enough. However, if it runs at size N + 1, then it will have to make new
allocations that are slightly larger. Not all the tensors are the same size.
Some might be (N + 1)*A and others (N + 1)*A*B where A and B are some non-batch
dimensions in the model. Because the allocator reuses existing allocations when
they are big enough, some number of (N + 1)*A allocations will actually fit in
the already existing N*B*A segments, though not perfectly. As the model runs it
will partially fill up all of these segments leaving unusable free slices of
memory at the end of these segments. The allocator at some point will need to
cudaMalloc a new (N + 1)*A*B segment. If there is not enough memory, there is
now no way to recover the slices of memory that are free at the end of existing
segments. With models 50+ layers deep, this pattern might repeat 50+ times
creating many slivers.

Approach

Expandable segments allows the allocator to create a segment initially and then
expand its size later when more memory is needed. Instead of making one segment
per allocation, it tries to make one segment (per stream) that grows as
necessary. Now when the N + 1 case runs, the allocations will tile nicely into
the one large segment until it fills up. Then more memory is requested and
appended to the end of the segment. This process does not create as many slivers
of unusable memory, so it is more likely to succeed at finding this memory.

Implementation

The expandable_segments:True option is used to enable/disable this behavior. We
use cuda's low-level memory APIs, which are similar to mmap, to extend the
memory segments. These APIs separate the allocation of physical memory
(cuMemCreate) from the allocation of virtual address space (cuMemAddressReserve)
and the associate between them cuMemMap/cuMemSetAccess.

When we allocate a new segment, we allocate enough address space to map
basically the entire physical memory of the GPU (there is 256TiB of address
space), but we only map enough physical memory to handle the current amount of
memory needed by the program. As more is requested, we add more physical memory
to the segment. This can work at the granularity of GPU pages which are 2MiB
currently.

If we end up out of memory, we can unmap all the memory in our segment
corresponding to empty physical pages, and return it to CUDA for use at another
address in the segment or in a segment for a different stream.

A current limitation of CUDA's API is that physical memory
(CUmemGenericAllocationHandle) cannot be split up after it is mapped even if the
handle holds multiple GPU pages. The cost to map/unmap memory is proportional to
the number of physical memory chunks that were allocated (mapping 10 separately
allocated 2MiB pages takes 10x time compared to mapping one 20MiB physical
allocation of 10 pages).  Changing memory mappings also appears to involve at
least some synchronous actions with the GPU and so should be considered an
expensive operation. To limit overhead, we use 2MiB pages for our small pool and
20MiB pages for our large pool. Initially allocation using expandable_blocks
will be slower than cudaMalloc, though still in the milliseconds range for
mapping the entire memory.

When mapping new memory to expand the segment, we look for the lowest address at
which we can fit a new allocation by adding new pages. Normally this will be at
the end of the block. But if have previously unmapped blocks earlier in the
segment during an OOM, it will first try to fill in those gaps to keep the
segment as a single block. By allocating at the lowest address we encourage
the split up parts of the block to merge into a single block again, reducing
fragmentation potential.

Allocation of blocks in the segment uses the same best-fit heuristics of the
rest of the allocator.

Expandable blocks can be enabled/disabled throughout the run of a program. When
disabled, the allocator will not put new allocations in an expandable block.

Limitations

* Slightly slower initial memory allocation speed.
* IPC of cuda tensors (e.g. for multiprocess dataloaders) is not supported.
However, it is possible to temporarily disable (expandable_segments:False) the
bevhavior for allocator tensors that need to be used cross-process.
* CUDA runtime APIs related to sharing memory across process
(cudaDeviceEnablePeerAccess) do not work for memory allocated with cuMemMap.
Instead these mapping have to be done manually. The allocator now has an
`enablePeerAccess` method to do this.
*/

template <>
struct ExpandableSegmentTraits<cuda::CUDAStream> {
  struct Handle {
    CUmemGenericAllocationHandle handle;
    std::optional<std::variant<int, CUmemFabricHandle>> shareable_handle;
  };
  using HandleT = Handle*;
};

struct CUDAExpandableSegment : ExpandableSegment<cuda::CUDAStream> {
  SegmentRange map(SegmentRange range) override {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }

    // if the handle type is not specified, try to use fabric handle first.
    // if it fails, use posix file handle
    if (CUDAAllocatorConfig::expandable_segments_handle_type() ==
        Expandable_Segments_Handle_Type::UNSPECIFIED) {
      CUDAAllocatorConfig::set_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::FABRIC_HANDLE);
      auto output = map(range);
      if (output.ptr != nullptr) {
        return output;
      }
      // if fabric handle is not supported, use posix file handle.
      CUDAAllocatorConfig::set_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::POSIX_FD);
      return map(range);
    }

    while (end > handles_.size()) {
      handles_.emplace_back(std::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      CUmemGenericAllocationHandle handle = 0;
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
#ifndef FBCODE_CAFFE2
      if (CUDAAllocatorConfig::expandable_segments_handle_type() !=
          Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      } else {
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
      }
#endif
      int flag = 0;
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuDeviceGetAttribute_(
          &flag,
          CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
          device_));
      if (flag)
        prop.allocFlags.gpuDirectRDMACapable = 1;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
      prop.location.id = static_cast<int>(device_);
      auto status =
          DriverAPI::get()->cuMemCreate_(&handle, segment_size_, &prop, 0);
      if (status != CUDA_SUCCESS) {
        if (status == CUDA_ERROR_OUT_OF_MEMORY) {
          for (auto j : c10::irange(begin, i)) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            auto h = handles_.at(j).value();
            handles_.at(j) = std::nullopt;
            C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h.handle));
          }
          trimHandles();
          return rangeFromHandles(begin, begin);
        } else if (
            CUDAAllocatorConfig::expandable_segments_handle_type() ==
            Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
          // we are testing if we can use fabric handle.
          // if we can, we will use it.
          // if we can't, we will use posix file handle.
          // so we should not return an error here.
          // in practice, we can get CUDA_ERROR_NOT_SUPPORTED or
          // CUDA_ERROR_NOT_PERMITTED to be safe, any non out-of-memory error is
          // considered as the handle type is not supported. if the handle type
          // is not supported, return a null range to indicate it.
          return SegmentRange(nullptr, 0);
        } else {
          C10_CUDA_DRIVER_CHECK(status);
        }
      }
      handles_.at(i) = Handle{handle, std::nullopt};
    }
    mapHandles(begin, end);
    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
    return rangeFromHandles(begin, end);
  }

  // Setup IPC sharing for range.
  // Returns the (larger) range that was actually shared.
  // Serializes data to std::ostream that can be passed to the
  // other process, and then restored as an exapandable segment
  // via ExpandableSegment::fromShared(istream);
  SegmentRange share(SegmentRange range, std::ostream& buf) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    ShareHeader header{getpid(), segment_size_, end - begin};
    buf.write((const char*)&header, sizeof(ShareHeader));
    for (auto i : c10::irange(begin, end)) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      auto& handle = handles_.at(i).value();
      if (CUDAAllocatorConfig::expandable_segments_handle_type() !=
          Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
        if (!handle.shareable_handle) {
          int fd = 0;
          C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemExportToShareableHandle_(
              &fd, handle.handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
          handle.shareable_handle = fd;
          LOG(INFO) << "use posix fd to share expandable segments.";
        }
        TORCH_CHECK(
            handle.shareable_handle != std::nullopt,
            "shareable_handle is null");
        buf.write((const char*)&*handle.shareable_handle, sizeof(int));
      } else {
        if (!handle.shareable_handle) {
          CUmemFabricHandle fabric_handle;
          C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemExportToShareableHandle_(
              &fabric_handle, handle.handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
          handle.shareable_handle = fabric_handle;
          LOG(INFO) << "use fabric handle to share expandable segments.";
        }
        TORCH_CHECK(
            handle.shareable_handle != std::nullopt,
            "shareable_handle is null");
        buf.write(
            (const char*)&*handle.shareable_handle, sizeof(CUmemFabricHandle));
      }
    }
    return rangeFromHandles(begin, end);
  }

  static std::unique_ptr<CUDAExpandableSegment> fromShared(
      c10::DeviceIndex device,
      std::vector<c10::DeviceIndex> peers,
      std::istream& buf) {
    ShareHeader header{};
    buf.read((char*)&header, sizeof(ShareHeader));
    auto segment = std::make_unique<CUDAExpandableSegment>();
    segment->init(device, std::nullopt, header.segment_size, std::move(peers));
// older build setups (e.g. multiwheels) do not have this syscall, added 2020
// but the kernel on the system might still support it.
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif
    if (CUDAAllocatorConfig::expandable_segments_handle_type() !=
        Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
      auto pidfd = syscall(SYS_pidfd_open, header.pid, 0);
      TORCH_CHECK(
          pidfd != -1 || errno != ENOSYS,
          "The kernel on this machine does not support the pidfd_open syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. "
          "Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.");
      TORCH_CHECK(pidfd != -1, "pidfd_open:", c10::utils::str_error(errno));
      for (auto i : c10::irange(header.num_handles)) {
        (void)i;
        int fd = 0;
        buf.read((char*)&fd, sizeof(int));
        auto myfd = syscall(SYS_pidfd_getfd, pidfd, fd, 0);
        if (myfd == -1) {
          auto err = errno;
          close((int)pidfd);
          for (auto& h : segment->handles_) {
            C10_CUDA_DRIVER_CHECK(
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                DriverAPI::get()->cuMemRelease_(h.value().handle));
            h = std::nullopt;
          }
          TORCH_CHECK(
              err != ENOSYS,
              "The kernel on this machine does not support the pidfd_getfd syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. "
              "Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.");
          TORCH_CHECK(false, "pidfd_getfd: ", c10::utils::str_error(err));
        }
        CUmemGenericAllocationHandle handle = 0;
        C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemImportFromShareableHandle_(
            &handle,
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            (void*)(uintptr_t)myfd,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        LOG(INFO) << "use posix fd to import expandable segments.";
        close((int)myfd);
        segment->handles_.emplace_back(Handle{handle, std::nullopt});
      }
      close((int)pidfd);
    } else {
      for (auto i : c10::irange(header.num_handles)) {
        (void)i;
        CUmemFabricHandle fabric_handle;
        buf.read((char*)&fabric_handle, sizeof(CUmemFabricHandle));
        CUmemGenericAllocationHandle handle = 0;
        C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemImportFromShareableHandle_(
            &handle,
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            (void*)&fabric_handle,
            CU_MEM_HANDLE_TYPE_FABRIC));
        LOG(INFO) << "use fabric handle to import expandable segments.";
        segment->handles_.emplace_back(Handle{handle, std::nullopt});
      }
    }
    segment->mapHandles(0, header.num_handles);
    segment->setAccess(device_, 0, header.num_handles);
    for (auto p : peers_) {
      segment->setAccess(p, 0, header.num_handles);
    }
    return segment;
  }

 private:
  size_t getReservedVirtualMemorySize(c10::DeviceIndex device) override {
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    return prop.totalGlobalMem + prop.totalGlobalMem / 8;
  }

  void createVirtualMemoryAddress(void** ptr) override {
    CUdeviceptr devPtr{};
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressReserve_(
        &devPtr, segment_size_ * max_handles_, 0ULL, 0, 0ULL));
    *ptr = reinterpret_cast<void*>(devPtr);
  }

  void releaseVirtualMemoryAddress(void* ptr) override {
    CUdeviceptr devPtr = reinterpret_cast<CUdeviceptr>(ptr);
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressFree_(
        devPtr, segment_size_ * max_handles_));
  }

  void mapHandles(size_t begin, size_t end) override {
    CUdeviceptr devPtr = reinterpret_cast<CUdeviceptr>(ptr_);
    for (auto i : c10::irange(begin, end)) {
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemMap_(
          devPtr + i * segment_size_,
          segment_size_,
          0,
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          handles_.at(i).value().handle,
          0ULL));
    }
  }

  void unmapHandles(size_t begin, size_t end) override {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    if (stream_) {
      C10_CUDA_CHECK(cudaStreamSynchronize(*stream_));
    } else {
      cuda::CUDAGuard device_guard(device_);
      C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUdeviceptr devPtr = reinterpret_cast<CUdeviceptr>(ptr_);
    for (auto i : c10::irange(begin, end)) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      Handle h = handles_.at(i).value();
      handles_.at(i) = std::nullopt;
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemUnmap_(
          devPtr + segment_size_ * i, segment_size_));
      if (h.shareable_handle) {
        close(std::get<int>(*h.shareable_handle));
      }
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h.handle));
    }
    trimHandles();
  }

  void setAccess(c10::DeviceIndex device, size_t begin, size_t end) override {
    CUmemAccessDesc desc{};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    desc.location.id = static_cast<int>(device);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUdeviceptr devPtr = reinterpret_cast<CUdeviceptr>(ptr_);
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
        devPtr + begin * segment_size_,
        (end - begin) * segment_size_,
        &desc,
        1));
  }

  struct ShareHeader {
    pid_t pid;
    size_t segment_size;
    size_t num_handles;
  };
};
#else
struct CUDAExpandableSegment : ExpandableSegment<cuda::CUDAStream> {
  void init(
      c10::DeviceIndex device,
      std::optional<StreamT> stream,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers) override {
    TORCH_INTERNAL_ASSERT(false, "expandable segment not supported");
  }

 private:
  size_t getReservedVirtualMemorySize(c10::DeviceIndex device) override {
    return 0;
  };

  void createVirtualMemoryAddress(void** ptr) override {};

  void releaseVirtualMemoryAddress(void* ptr) override {};

  void mapHandles(size_t begin, size_t end) override {};

  void unmapHandles(size_t begin, size_t end) override {};

  void setAccess(c10::DeviceIndex device, size_t begin, size_t end) override {};
}
#endif

// BlockState, BlockPoolState, and PrivatePoolState contain the information
// needed to reconstruct a private pool to a previous state. See note
// [Checkpointing PrivatePoolState]
struct BlockState {
  c10::DeviceIndex device = 0;
  cudaStream_t stream = nullptr;
  stream_set stream_uses = {};
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int64_t gc_count_base = 0;
  // maintain invariant that event_count == 0 ;
  // history will be left alone in checkpoint

  BlockState(Block* block);
};

struct SegmentState {
  std::vector<BlockState> blocks;
  bool is_small = false;

  SegmentState(Block* head);
};

struct PrivatePoolState : AllocatorState {
  // omitting use_count, and cudaMalloc_count as they remain the same
  MempoolId_t owner_id = {0, 0};

  std::vector<SegmentState> segments;

  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};

struct RestoreResult {
  std::vector<void*> allocations_freed;
  std::vector<Block*> allocations_created;
};

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>;
  // TODO: Explicit device count
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(c10::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](cudaEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<cudaEvent_t>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    auto new_ptr = std::make_unique<cudaEvent_t>();
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming));

    return Event(new_ptr.release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

BlockState::BlockState(Block* block)
    : device(block->device),
      stream(block->stream),
      stream_uses(block->stream_uses),
      size(block->size),
      ptr(block->ptr),
      allocated(block->allocated),
      gc_count_base(block->gc_count_base) {
  TORCH_CHECK(
      block->event_count == 0,
      "Events should have synchronized when checkpointing block");
}

SegmentState::SegmentState(Block* head) {
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
  is_small = head->pool->is_small;

  for (Block* curr = head; curr != nullptr; curr = curr->next) {
    blocks.emplace_back(curr);
  }
}

PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Block*>& private_pool_head_blocks)
    : owner_id(std::move(pool_id)) {
  for (Block* head : private_pool_head_blocks) {
    segments.emplace_back(head);
  }
}

cudaError_t allocPrimitive(void** ptr, size_t size, AllocParams& p) {
  if (p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator()) {
    *ptr = p.pool->owner_PrivatePool->allocator()->raw_alloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
  } else {
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(ptr, size));
  }
}

} // anonymous namespace
} // namespace Native

namespace Native {

struct CUDACachingDeviceAllocatorImpl : CachingDeviceAllocatorImpl<
                                            cuda::CUDAStream,
                                            EventPool::Event,
                                            Block,
                                            CUDAExpandableSegment> {
  void getMemoryInfo(
      c10::DeviceIndex device,
      size_t& free_bytes,
      size_t& total_bytes) override {
    c10::cuda::CUDAGuard device_guard(device);
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_buffer_max_entries,
      RecordContext when,
      bool clearHistory) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(when == RecordContext::NEVER || context_recorder);
    record_history = enabled;
    context_recorder_.store(record_history ? context_recorder : nullptr);
    cuda_alloc_buffer.setMaxEntries(alloc_buffer_max_entries);
    record_context_ = enabled ? when : RecordContext::NEVER;
    if (!enabled || clearHistory) {
      cuda_alloc_buffer.clear();
    }
  }

  bool isHistoryEnabled() {
    return record_history;
  }

 private:
  void deallocate_device_ptr(BlockT* block) override {
    C10_CUDA_CHECK(cudaFree((void*)block->ptr));
  }

  void allocate_device_ptr(void** ptr, AllocParamsT& p) {
    cudaError_t error = C10_CUDA_ERROR_HANDLED(cudaMalloc(ptr, p.alloc_size));
    if (error == cudaErrorMemoryAllocation) {
      p.status = AllocParamsT::OOM;
      return;
    }
    C10_CUDA_CHECK(error);
    p.status = AllocParamsT::Ok;
  }

  void cudaMallocMaybeCapturing(void** ptr, AllocParamsT& p) {
    if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
        at::cuda::CaptureStatus::None) {
      return allocPrimitive(ptr, p);
    } else {
      // It's ok to capture cudaMallocs, as long as we never cudaFree those
      // addresses before replay.
      // Capturing cudaMalloc behaves nicely: it gives the graph new VA,
      // but is ignored (won't leakily allocate new memory) in replays.
      at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
      return allocPrimitive(ptr, p);
    }
  }

  void mallocMaybeCapturingWithOptionalLock(
      void** ptr,
      AllocParamsT& p,
      std::unique_lock<std::recursive_mutex>& lock) override {
    if (CUDAAllocatorConfig::release_lock_on_cudamalloc()) {
      // At scope exit, acquire the lock again. This provides safety against
      // any potential exceptions in the cudaMallocMaybeCapturing function.
      auto sg = c10::make_scope_exit([&]() { lock.lock(); });
      lock.unlock();
      cudaMallocMaybeCapturing(&ptr, p);
    } else {
      cudaMallocMaybeCapturing(&ptr, p);
    }
    if (CUDAAllocatorConfig::release_lock_on_cudamalloc()) {
      TORCH_CHECK(lock.owns_lock(), "Failed to acquire lock after cudaMalloc");
    }
  }

#ifdef PYTORCH_C10_DRIVER_API_SUPPORTED
  std::string reportProcessMemoryInfo(c10::DeviceIndex device) override {
    void* nvml_handle = DriverAPI::get_nvml_handle();
    if (!nvml_handle) {
      return "";
    }
    static bool nvml_init [[maybe_unused]] = []() {
      TORCH_INTERNAL_ASSERT(NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_());
      return true;
    }();

    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // NOLINTNEXTLINE(*-c-arrays)
    char pci_id[80];
    snprintf(
        pci_id,
        sizeof(pci_id),
        NVML_DEVICE_PCI_BUS_ID_FMT,
        prop.pciDomainID,
        prop.pciBusID,
        prop.pciDeviceID);

    nvmlDevice_t nvml_device = nullptr;
    TORCH_INTERNAL_ASSERT(
        NVML_SUCCESS ==
        DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_(
            pci_id, &nvml_device));

    std::vector<nvmlProcessInfo_v1_t> procs(8);
    unsigned int size = procs.size();
    nvmlReturn_t r{};
    while ((r = DriverAPI::get()->nvmlDeviceGetComputeRunningProcesses_(
                nvml_device, &size, procs.data())) ==
           NVML_ERROR_INSUFFICIENT_SIZE) {
      procs.resize(size);
    }
    unsigned int self_pid = getpid();
    std::stringstream ss;
    TORCH_INTERNAL_ASSERT(NVML_SUCCESS == r);
    ss << "";
    for (auto i : c10::irange(size)) {
      auto& proc = procs[i];
      if (self_pid == proc.pid) {
        ss << "Including non-PyTorch memory, this process";
      } else {
        ss << "Process " << proc.pid;
      }
      ss << " has " << format_size(proc.usedGpuMemory) << " memory in use. ";
    }
    return ss.str();
  }
#endif

  EventT create_event_internal(c10::DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  EventT record_event_for_stream(StreamT stream) override {
    C10_CUDA_CHECK(c10::cuda::SetDevice(stream.device_index()));
    EventT event = create_event_internal(stream.device_index());
    C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));
  }

  bool query_event(const EventT& event) override {
    cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));
    if (err == cudaErrorNotReady) {
      // ignore and clear the error if not ready
      (void)cudaGetLastError();
      return false;
    }
    C10_CUDA_CHECK(err);
    return true;
  }

  void synchronize_event(const EventT& event) override {
    C10_CUDA_CHECK(cudaEventSynchronize(*event));
  }

#ifdef PYTORCH_C10_DRIVER_API_SUPPORTED
  bool is_expandable_segment_supported() const override {
    return true;
  }
#endif

  // Iterates over sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPoolT& pool, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  bool checkPoolLiveAllocations(
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    PrivatePool* pool = nullptr;
    auto pool_it = graph_pools.find(mempool_id);
    TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id");
    pool = pool_it->second.get();

    TORCH_INTERNAL_ASSERT(pool != nullptr);

    size_t allocated_pool_blocks = 0;

    for (Block* b : active_blocks) {
      TORCH_INTERNAL_ASSERT(b != nullptr);
      TORCH_INTERNAL_ASSERT(b->pool != nullptr);
      if (b->allocated && b->pool->owner_PrivatePool == pool) {
        if (!expected_live_allocations.count(b->ptr)) {
          return false;
        }

        allocated_pool_blocks += 1;
      }
    }

    return allocated_pool_blocks == expected_live_allocations.size();
  }

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(
        !block->expandable_segment_,
        "Tensors allocated with expandable_segments:True cannot be shared between processes. Consider using expandable_segments:False in data loading workers via torch.cuda.memory._set_allocator_settings('expandable_segments:False')");
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  ShareableHandle shareIpcHandle(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::ostringstream ss;
    ss.put(SHAREABLE_HANDLE_VERSION);
    ptrdiff_t offset = 0;
    if (!block->expandable_segment_) {
      ss.put(SHAREABLE_CUDA_MALLOC);
      Block* base_block = block;
      while (base_block->prev) {
        base_block = base_block->prev;
      }
      offset = (char*)block->ptr - (char*)base_block->ptr;
      cudaIpcMemHandle_t handle;
      C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, base_block->ptr));
      ss.write((char*)&handle, CUDA_IPC_HANDLE_SIZE);
    } else {
      ss.put(SHAREABLE_CUDA_EXPANDABLE_SEGMENT);
      auto full_range = block->expandable_segment_->share(
          SegmentRange(block->ptr, block->size), ss);
      offset = (char*)block->ptr - (char*)full_range.ptr;
    }
    return ShareableHandle{offset, ss.str()};
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes = 0;
      C10_CUDA_CHECK(cudaMemGetInfo(
          largest, // Use free memory as an optimistic initial guess of *largest
          &tmp_bytes));
    }
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_blocks, largest);
      cache_info_aux(gp.second->small_blocks, largest);
    }
  }

  /* Checkpoint the state of a private pool necessary to return it to its
   * current state */
  std::unique_ptr<PrivatePoolState> getCheckpointState(MempoolId_t id) {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    insert_events_deferred_until_no_capture(context);

    auto pool = graph_pools.find(id);
    if (pool != graph_pools.end()) {
      auto private_pool_head_blocks =
          get_private_pool_head_blocks(pool->second.get());
      return std::make_unique<PrivatePoolState>(id, private_pool_head_blocks);
    } else if (graph_pools_freeable.count(id)) {
      TORCH_CHECK(false, "Not expected to checkpoint freeable graph");
    } else {
      TORCH_CHECK(false, "Could not find pool of id");
    }
  }

  void freeBlocksAllocatedToPool(PrivatePool* private_pool, RestoreResult& rr) {
    auto pool_blocks = get_private_pool_head_blocks(private_pool);

    std::vector<Block*> head_blocks;
    for (Block* block : pool_blocks) {
      if (block->prev == nullptr) {
        head_blocks.push_back(block);
      }
    }

    for (Block* block : head_blocks) {
      Block* curr = block;

      while (curr) {
        // When we free a block, its pointer should never change
        // only its adjacent blocks, so free, then look at pointer
        if (curr->allocated) {
          TORCH_CHECK(
              curr->event_count == 0,
              "Events should have synchronized when setting checkpointed block");
          rr.allocations_freed.push_back(curr->ptr);
          free(curr);
          TORCH_CHECK(!curr->allocated)
        }
        curr = curr->next;
      }
    }

    for (Block* b : get_private_pool_head_blocks(private_pool)) {
      Block* curr = b;
      while (curr) {
        TORCH_CHECK(!curr->allocated);
        curr = curr->next;
      }
    }
  }

  // checkpoint the state of an allocation that may have been
  // split into multiple blocks
  void setSegmentStateToCheckpoint(
      Block* block,
      SegmentState& segment,
      const std::shared_ptr<GatheredContext>& context,
      RestoreResult& rr) {
    Block* curr_block = block;
    Block* last_block = block;

    TORCH_INTERNAL_ASSERT(block->pool);
    BlockPool& pool = *block->pool;
    const auto segment_len = segment.blocks.size();

    // allocate all blocks in the segment
    for (size_t i = 0; i < segment_len; ++i) {
      // The last block in every expandable segment is the remaining amount of
      // available unmapped virtual address space. We shouldn't change it but
      // instead check it is correctly formed then skip over allocating it.
      if (i == segment_len - 1 && curr_block->expandable_segment_) {
        TORCH_CHECK(curr_block->next == nullptr);
        TORCH_CHECK(!curr_block->mapped);
        TORCH_CHECK(curr_block->allocated == false);
        continue;
      }

      auto& block_state = segment.blocks.at(i);
      AllocParams params(
          block_state.device,
          block_state.size,
          block_state.stream,
          &pool,
          block_state.size);
      pool.blocks.erase(curr_block);
      params.block = curr_block;
      params.stat_types = get_stat_types_for_pool(pool);

      // splitting a block depends on `max_split_size`, which may have changed
      // between when checkpoint was taken and now, so we make sure to recreate
      // the behavior from the checkpoint. Keep splitting as long as there is
      // space left in the block because the block is already the size of how it
      // appears in the segment, so any leftover space belongs to the next
      // block.
      bool split = curr_block->size > block_state.size;

      // curr_block will become next pointer if it is split, so reassign with
      // the returned value
      curr_block = alloc_found_block(params, block_state.size, context, split);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->size == block_state.size);

      last_block = curr_block;
      curr_block = curr_block->next;

      TORCH_CHECK((curr_block != nullptr) == ((i + 1) < (segment_len)));
    }

    while (last_block->prev) {
      last_block = last_block->prev;
    }

    // free blocks that are not allocated in the checkpoint
    curr_block = last_block;

    for (size_t i = 0; i < segment_len; ++i, curr_block = curr_block->next) {
      if (i == segment_len - 1 && curr_block->expandable_segment_) {
        TORCH_CHECK(curr_block->next == nullptr);
        TORCH_CHECK(!curr_block->mapped);
        TORCH_CHECK(curr_block->allocated == false);
        continue;
      }

      auto& block_state = segment.blocks.at(i);
      TORCH_INTERNAL_ASSERT(curr_block != nullptr);

      if (block_state.allocated) {
        rr.allocations_created.push_back(curr_block);
        continue;
      }

      free(curr_block);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->allocated == block_state.allocated);
      TORCH_CHECK(curr_block->size == block_state.size);
    }
  }

  /**
   * Note [Checkpointing PrivatePoolState]
   *
   * Refer above to Note [Interaction with CUDA graph capture]. Allocations made
   * during graph capture are made from a separate private pool. During graph
   * capture allocations behave as usual. During graph replay the allocator
   * state does not change even as new tensors are created. The private pool
   * will not free its blocks to the main caching allocator until cuda graph use
   * is finished to prevent an allocation from eager clobbering the memory from
   * a live but unaccounted for tensor that was created during replay.
   *
   * `make_graphed_callables`, a series of separate callables chained in
   * successive cuda graphs, can share a memory pool because after a cuda graph
   * recording the allocations in the shared private pool exactly reflect the
   * tensors that are allocated.
   *
   * We would like to extend callable chaining to support a graphed callable
   * tree. In this scenario, we have a tree of callable chains which will be
   * captured with cuda graphs. In the diagram below, we have a tree with four
   * callables, A, B, C, and D. Suppose we have captured, and subsequently
   * replayed, A, B, and C. Then on a new invocation, we replay A and B, but
   * would now like to record D. At this point the private pool will not reflect
   * any of the live tensors created during graph replay. Allocations made
   * during a new recording with the pool could overwrite those live tensors.
   *
   * In order to record a new graph capture after replaying prior callables in
   * the tree, we need the allocator to reflect the state of the live tensors.
   * We checkpoint the state of the private pool after each recording, and then
   * reapply it when we are starting a new recording chain. Additionally, we
   * must free the allocations for any tensors that died between the end of our
   * previous graph replaying and our new recording. All of the allocated
   * segments that existed in the checkpointed state must still exist in the
   * pool. There may also exist new allocated blocks.
   * (TODO : link note [live tensors between iterations] when it exists). For
   * every block that is currently allocated but no allocated in the snapshot,
   * we will return a pointer to their block.
   *.
   *
   *
   *  ---------------> A ---------------> B ---------------> C
   *                                      |
   *                                      |
   *                                      |
   *                                      |
   *                                      ╰ ---------------> D
   */
  RestoreResult setCheckpointPoolState(PrivatePoolState& pps) {
    // To reset the caching allocator state we will
    // - Free all the blocks currently allocated to the pool (see [live tensors
    // between iterations])
    // - Allocate all the blocks in a checkpointed segment, whether they are
    // live or not
    // - Free the blocks in a checkpointed segment which are not live
    // This could be optimized, but it nicely reuses exiting apis, and this
    // is not on the hot path.

    // following `done outside the lock because we don't know what locks the
    // recorder needs to have...`

    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::STATE);

    std::lock_guard<std::recursive_mutex> lock(mutex);

    RestoreResult rr;

    TORCH_CHECK(
        !graph_pools_freeable.count(pps.owner_id),
        "Not expected to checkpoint freeable graph");

    auto pool = graph_pools.find(pps.owner_id);
    TORCH_CHECK(pool != graph_pools.end(), "Could not find private pool id");

    PrivatePool* private_pool = pool->second.get();

    freeBlocksAllocatedToPool(private_pool, rr);

    std::unordered_map<void*, Block*> ptrs_to_blocks;
    // at this point, all of the blocks should be free, so they will all be in
    // the block set
    for (Block* block : private_pool->small_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }
    for (Block* block : private_pool->large_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }

    for (auto& segment : pps.segments) {
      auto ptr = segment.blocks.at(0).ptr;
      TORCH_CHECK(ptrs_to_blocks.count(ptr), " could not find ", ptr)
      auto block = ptrs_to_blocks[ptr];

      setSegmentStateToCheckpoint(block, segment, context, rr);
    }
    return rr;
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<Block*> all_blocks;

    if (mempool_id.first != 0 || mempool_id.second != 0) {
      // If there is an active mempool, we find the corresponding PrivatePool
      // in graph_pools and only return the blocks from it.
      auto pool = graph_pools.find(mempool_id);
      if (pool != graph_pools.end()) {
        all_blocks = get_private_pool_head_blocks(pool->second.get());
      }
    } else {
      // When snapshot is called with non-default mempool_id, we return
      // all the blocks in the CUDACachingAllocator (as returned by
      // get_all_blocks).
      all_blocks = get_all_blocks();
    }

    size_t total_active = 0;
    std::vector<SegmentInfo> result;

    for (const Block* const head_block : all_blocks) {
      // For expandable segments, we report one segment for each contiguous
      // mapped range of memory
      if (head_block->prev && head_block->prev->mapped) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<size_t>(head_block->ptr);
      segment_info.stream = head_block->stream;
      segment_info.is_large = (!head_block->pool->is_small);
      segment_info.is_expandable = head_block->expandable_segment_;
      segment_info.context_when_allocated =
          head_block->context_when_segment_allocated;
      MempoolId_t id = head_block->pool->owner_MempoolId();
      if ((mempool_id.first == 0 && mempool_id.second == 0) ||
          id == mempool_id) {
        segment_info.owner_private_pool_id = id;
      }

      const Block* block = head_block;
      while (block != nullptr && block->mapped) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.requested_size = block->requested_size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0) ||
            !block->stream_uses.empty();

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
          segment_info.requested_size += block_info.requested_size;
        }
        block_info.context_when_allocated = block->context_when_allocated;
        block = block->next;
      }
      total_active += segment_info.active_size;
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });

    record_trace(
        TraceEntry::SNAPSHOT, 0, total_active, nullptr, 0, mempool_id, nullptr);
    return result;
  }

  std::vector<TraceEntry> trace(
      const std::function<time_t(approx_time_t)>& tsc_to_us) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    alloc_buffer.getEntries(result);

    // Convert all the timestamps from tsc to epoch time in microseconds.
    for (auto& te : result) {
      te.time_.t_ = tsc_to_us(te.time_.approx_t_);
    }
    return result;
  }

  bool hasAllocatedExpandableSegments() const {
    return !expandable_segments_.empty();
  }
};

// Returns whether to force all allocations to bypass the caching allocator and
// go straight to cudaMalloc.  This setting is useful when debugging GPU memory
// errors, since the caching allocator foils cuda-memcheck.
static bool forceUncachedAllocator() {
  // Allow either CUDA or HIP name for env var for maximum user comfort
  // the CUDA env var avoids being hipified in cuda_to_hip_mappings.py
  static auto has_cuda_env =
      c10::utils::check_env("PYTORCH_NO_CUDA_MEMORY_CACHING") == true;
  static auto has_rocm_env =
      c10::utils::check_env("PYTORCH_NO_HIP_MEMORY_CACHING") == true;
  static bool force_uncached = has_cuda_env || has_rocm_env;
  return force_uncached;
}

static void* uncached_allocate(size_t size) {
  void* devPtr = nullptr;
  // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
  // if someone tries to use forceUncachedAllocator while capturing.
  C10_CUDA_CHECK(cudaMalloc(&devPtr, size));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_allocation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(devPtr));
  }
  return devPtr;
}

static void uncached_delete(void* ptr) {
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_deallocation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(ptr));
  }
  C10_CUDA_CHECK(cudaFree(ptr));
}

static void local_raw_delete(void* ptr);
using CUDACachingDeviceAllocatorInterface = CachingDeviceAllocatorInterface<
    c10::kCUDA,
    local_raw_delete,
    CUDACachingDeviceAllocatorImpl,
    CUDAAllocator>;

class NativeCachingAllocator : public CUDACachingDeviceAllocatorInterface {
 private:
  // allows this allocator to be turned on and off programmatically
  bool enable_ = true;

  // Variables by memory snapshot
  c10::ApproximateClockToUnixTimeConverter clock_converter;
  bool record_history = false;
  RingBuffer<AnnotationEntry> annotation_buffer;

 public:
  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_buffer_max_entries,
      RecordContext when,
      bool clearHistory) override {
    record_history = enabled;
    annotation_buffer.setMaxEntries(alloc_buffer_max_entries);
    annotation_buffer.clear();
    for (auto& impl : impls_) {
      impl->recordHistory(
          enabled,
          context_recorder,
          alloc_buffer_max_entries,
          when,
          clearHistory);
    }
  }

  void recordAnnotation(
      const std::vector<std::pair<std::string, std::string>>& md) override {
    if (!record_history) {
      return;
    }
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    auto ae = AnnotationEntry(
        /*device=*/device,
        /*time=*/getApproximateTime());
    for (const auto& md_pair : md) {
      ae.recordUserMetadata(md_pair.first, md_pair.second);
    }
    annotation_buffer.insertEntries(ae);
  }

  void pushCompileContext(std::string& md) override {
    if (!record_history) {
      return;
    }
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    impls_[device]->pushCompileContext(md);
  }

  void popCompileContext() override {
    if (!record_history) {
      return;
    }
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    impls_[device]->popCompileContext();
  }

  bool isHistoryEnabled() override {
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    return impls_[device]->isHistoryEnabled();
  }

  bool checkPoolLiveAllocations(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) override {
    return impls_[device]->checkPoolLiveAllocations(
        mempool_id, expected_live_allocations);
  }

  void enable(bool value) override {
    enable_ = value;
  }

  bool isEnabled() const override {
    return enable_;
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) override {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return impls_[block->device]->getBaseAllocation(block, outSize);
  }

  ShareableHandle shareIpcHandle(void* ptr) override {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return impls_[block->device]->shareIpcHandle(block);
  }

  SnapshotInfo snapshot(MempoolId_t mempool_id) override {
    // Set-up converter to convert timestamps from tsc to microseconds.
    auto tsc_to_ns = clock_converter.makeConverter();
    auto tsc_to_us = [=](approx_time_t t_approx) {
      return tsc_to_ns(t_approx) / 1000;
    };

    SnapshotInfo result;

    // Get AnnotationEntry list and convert the timestamps.
    annotation_buffer.getEntries(result.external_annotations);
    for (auto& ae : result.external_annotations) {
      ae.time_.t_ = tsc_to_us(ae.time_.approx_t_);
    }

    // Get the device_traces' TraceEntry lists.
    for (auto& impl : impls_) {
      result.device_traces.emplace_back(impl->trace(tsc_to_us));
      auto snap = impl->snapshot(mempool_id);
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }

    auto& md = result.config_metadata;
    md.garbage_collection_threshold =
        AcceleratorAllocatorConfig::garbage_collection_threshold();
    md.max_split_size = AcceleratorAllocatorConfig::max_split_size();
    md.pinned_num_register_threads =
        CUDAAllocatorConfig::pinned_num_register_threads();
    md.expandable_segments = CUDAAllocatorConfig::expandable_segments();
    md.release_lock_on_malloc =
        CUDAAllocatorConfig::release_lock_on_cudamalloc();
    md.pinned_use_host_register =
        CUDAAllocatorConfig::pinned_use_cuda_host_register();
    md.last_allocator_settings =
        AcceleratorAllocatorConfig::last_allocator_settings();
    md.roundup_power2_divisions =
        AcceleratorAllocatorConfig::roundup_power2_divisions();

    return result;
  }

  std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) override {
    return impls_[device]->getCheckpointState(id);
  }

  /**
   * @brief Checkpoint the private pool state identified in `as` to its prior
   * state
   *
   * @param device - device of the pool to manipulate
   * @param as - allocator state
   * @param stale_live_storages - storages of tensors which are currently
   * allocated but which will be not be allocated after the checkpoint is set.
   * For these storages we will remove their deleter function.
   * @return CheckpointDelta - Freed Pointers and DataPtrs that contain deleter
   * functions for all allocated blocks in the new checkpoint state.
   */
  CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> as) override {
    std::shared_ptr<PrivatePoolState> pps =
        std::dynamic_pointer_cast<PrivatePoolState>(as);

    TORCH_CHECK(pps, "Expected PrivatePoolState");

    auto rr = impls_[device]->setCheckpointPoolState(*pps);

    CheckpointDelta cpd;
    for (void* ptr : rr.allocations_freed) {
      get_allocated_block(ptr, /*remove*/ true);
      cpd.ptrs_freed.push_back(ptr);
    }
    for (Block* block : rr.allocations_created) {
      add_allocated_block(block);
      cpd.dataptrs_allocd.emplace_back(
          block->ptr,
          block->ptr,
          &local_raw_delete,
          Device(DeviceType::CUDA, device));
    }

    return cpd;
  }

  DataPtr allocate(size_t size) override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    void* devPtr = nullptr;
    void (*deleteFunc)(void*) = &local_raw_delete;
    CUDAStream stream = cuda::getCurrentCUDAStream(device);

    if (forceUncachedAllocator() || !isEnabled()) {
      deleteFunc = &uncached_delete;
      devPtr = uncached_allocate(size);
    } else {
      if (size != 0) {
        this->malloc(&devPtr, device, size, stream);
      }
    }

    if (size && TORCH_SDT_IS_ENABLED(malloc)) {
      TORCH_SDT_WITH_SEMAPHORE(malloc, devPtr, device, size, stream.id());
    }

    return {devPtr, devPtr, deleteFunc, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator() || !isEnabled()) {
      return &uncached_delete;
    } else {
      return &local_raw_delete;
    }
  }
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {
    impls_[device]->cacheInfo(largestBlock);
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    void* r = nullptr;
    if (forceUncachedAllocator() || !isEnabled()) {
      r = uncached_allocate(nbytes);
    } else {
      c10::DeviceIndex device = 0;
      C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
      malloc(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
    }
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    void* r = nullptr;
    if (forceUncachedAllocator() || !isEnabled()) {
      r = uncached_allocate(nbytes);
    } else {
      c10::DeviceIndex device = 0;
      C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
      malloc(&r, device, nbytes, stream);
    }
    return r;
  }

  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override {
    c10::cuda::CUDAGuard device_guard(dev);
    cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
      // ignore and clear the error if access was already enabled
      (void)cudaGetLastError();
    } else {
      C10_CUDA_CHECK(err);
    }
    impls_[dev_to_access]->addPeerAccess(dev);
    std::lock_guard<std::mutex> lock(IpcMutex);
    for (auto& entry : ipcMemHandle_to_devptr) {
      if (entry.second.device_ == dev_to_access &&
          entry.second.expandable_segment_) {
        entry.second.expandable_segment_->addPeer(dev);
      }
    }
  }

  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override {
    if (p2p_enabled || // memcpy ok because memory is mapped in both devices
        srcDevice == dstDevice || // memcpy ok on a single device
        // memcpy ok because both dst and src must have come from cudaMalloc
        (!impls_[dstDevice]->hasAllocatedExpandableSegments() &&
         !impls_[srcDevice]->hasAllocatedExpandableSegments())) {
      return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }
    // when p2p is not enabled, only cudaMemcpyPeerAsync correctly handles
    // memory not allocated via cudaMalloc
    return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
  }

  void raw_delete(void* ptr) override {
    if (forceUncachedAllocator() || !isEnabled()) {
      uncached_delete(ptr);
    } else {
      this->free(ptr);
    }
  }

  // In CUDA IPC, sender sends a tensor to receiver via shareIPCHandle,
  // getIpcDevPtr is called by the receiving process to map the CUDA memory from
  // the sending process into its own address space.

  // When allocated with cudaMalloc we use the cudaIPCMemHandle_t APIs.
  // These APIs only allow sharing a big memory block associated with a
  // cudaIpcMemHandle_t and it can be opened only **once** per context per
  // process. There can be multiple types of storage in the same IPC mem block,
  // so we must cache the device ptr to construct typed storage as it comes.

  // When using cuMemCreate, via expandable segments, we use
  // cuMemExportToShareableHandle to create a file descriptor that can be sent
  // to the other process to sort the object. Then we recreate part of the
  // exandable segment necessary to load the allocation.

  // ipcMemHandle_to_devptr caches the mapping from shareable handle to
  // this process' memory mapping information for that share to ensure we do not
  // create it twice. When the shared_ptr is no longer in use we clean up the
  // cache.

  std::mutex IpcMutex;
  struct MemHandleCacheEntry {
    MemHandleCacheEntry(
        c10::DeviceIndex device,
        std::string& handle,
        const CUDACachingDeviceAllocatorImpl& allocator_impl)
        : device_(device) {
      int type = SHAREABLE_CUDA_MALLOC;
      std::istringstream ss(handle);
      if (handle.size() != CUDA_IPC_HANDLE_SIZE) {
        auto version = ss.get();
        TORCH_CHECK(
            version <= SHAREABLE_HANDLE_VERSION,
            "received sharable handle from a future version of torch that this version does not know how to handle")
        type = ss.get();
      } // otherwise this is coming from an old pytorch where it has to be a raw
        // SHARABLE_CUDA_MALLOC
      if (type == SHAREABLE_CUDA_MALLOC) {
        cudaIpcMemHandle_t cuda_handle;
        ss.read((char*)&cuda_handle, CUDA_IPC_HANDLE_SIZE);
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(
            &cuda_ipc_ptr_, cuda_handle, cudaIpcMemLazyEnablePeerAccess));
      } else if (type == SHAREABLE_CUDA_EXPANDABLE_SEGMENT) {
        expandable_segment_ =
            ExpandableSegmentT::fromShared(device, allocator_impl.peers(), ss)
                .release();
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "unexpected or illformed shareable handle type");
      }
    }
    // this struct expects that clear is explicitly called to
    // free resources, because we only want this code running when
    // the shared pointer to this entry is destructed, not during
    // deinitialization when cuda may already have been shutdown.
    // This replicates the previous behavior of this map when it
    // stored raw cuda_ipc_ptr_ handles.
    void clear() {
      if (cuda_ipc_ptr_) {
        cuda::CUDAGuard device_guard(device_);
        C10_CUDA_CHECK(cudaIpcCloseMemHandle(cuda_ipc_ptr_));
        cuda_ipc_ptr_ = nullptr;
      }
      if (expandable_segment_) {
        delete expandable_segment_;
        expandable_segment_ = nullptr;
      }
    }
    void* ptr() {
      if (cuda_ipc_ptr_) {
        return cuda_ipc_ptr_;
      } else {
        return expandable_segment_->ptr();
      }
    }
    c10::DeviceIndex device_;
    ExpandableSegmentT* expandable_segment_{nullptr};
    void* cuda_ipc_ptr_{nullptr}; // nullptr if expandable_segment_ is not null
    std::weak_ptr<void> wp_;
  };

  ska::flat_hash_map<std::string, MemHandleCacheEntry> ipcMemHandle_to_devptr;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    std::lock_guard<std::mutex> lock(IpcMutex);

    auto iter = ipcMemHandle_to_devptr.find(handle);
    if (iter != ipcMemHandle_to_devptr.end()) {
      auto devptr = iter->second.wp_.lock();
      // the weak_ptr should always be valid because we delete the entry from
      // the cache when the shared_ptr is destructed, so we should never get
      // here.
      TORCH_INTERNAL_ASSERT(devptr, "entry in cache has missing shared_ptr");
      return devptr;
    }
    c10::DeviceIndex curr_device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&curr_device));
    auto inserted = ipcMemHandle_to_devptr.insert(
        iter,
        {handle,
         MemHandleCacheEntry(curr_device, handle, *impls_[curr_device])});
    auto sp = std::shared_ptr<void>(
        inserted->second.ptr(), [handle, this](void* ptr) {
          std::unique_lock<std::mutex> deleter_lock(IpcMutex);

          auto it = ipcMemHandle_to_devptr.find(handle);
          TORCH_INTERNAL_ASSERT(it != ipcMemHandle_to_devptr.end());
          auto entry = std::move(it->second);
          ipcMemHandle_to_devptr.erase(it);

          // ExpandableSegment synchronizes on destruction in unmapHandles, so
          // we need to release the lock first to minimize the performance hit.
          deleter_lock.unlock();
          entry.clear();
        });
    inserted->second.wp_ = sp;
    return sp;
  }

  std::string name() override {
    return "native";
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    C10_CUDA_CHECK(
        cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
  }
};

static NativeCachingAllocator allocator;

void local_raw_delete(void* ptr) {
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  allocator.free(ptr);
}

} // namespace Native

namespace CudaMallocAsync {
// If this is put in its own header file, it gets incorrectly renamed in HIPify.
// NOLINTNEXTLINE(misc-use-internal-linkage)
CUDAAllocator* allocator();

} // namespace CudaMallocAsync

struct BackendStaticInitializer {
  CUDAAllocator* parseEnvForBackend() {
    // If the environment variable is set, we use the CudaMallocAsync allocator.
    if (CUDAAllocatorConfig::use_async_allocator()) {
      return CudaMallocAsync::allocator();
    }
    return &Native::allocator;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
// Register this HIP allocator as the CUDA allocator to allow it to work
// with both c10::GetAllocator(kCUDA) and c10::getDeviceAllocator(kCUDA)
// APIs. We don't perform this masquerading inside
// HIPAllocatorMasqueradingAsCUDA because it needs to happen during static
// initialization, and doing so there may introduce static initialization
// order (SIOF) issues.
#define HIP_MASQUERADING_AS_CUDA \
  "cud"                          \
  "a"
    at::SetAllocator(c10::Device(HIP_MASQUERADING_AS_CUDA).type(), r, 0);
    allocator.store(r);
#undef HIP_MASQUERADING_AS_CUDA
  }
};

std::atomic<CUDAAllocator*> allocator;
static BackendStaticInitializer backend_static_initializer;
} // namespace cuda::CUDACachingAllocator
} // namespace c10

namespace c10::cuda {

// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10::cuda::MemPool().
//
// uuid_ is incremented when CUDAGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or CUDAGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    CUDACachingAllocator::CUDAAllocator* allocator,
    bool is_user_created,
    bool use_on_oom,
    bool symmetric)
    : allocator_(allocator),
      is_user_created_(is_user_created),
      symmetric_(symmetric) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10::cuda::current_device();
  CUDACachingAllocator::createOrIncrefPool(device_, id_, allocator);
  if (use_on_oom) {
    CUDACachingAllocator::setUseOnOOM(device_, id_);
  }
}

MemPool::~MemPool() {
  TORCH_INTERNAL_ASSERT(use_count() == 1);
  CUDACachingAllocator::releasePool(device_, id_);
  c10::cuda::CUDACachingAllocator::emptyCache(id_);
}

MempoolId_t MemPool::id() {
  return id_;
}

bool MemPool::is_symmetric() {
  return symmetric_;
}

CUDACachingAllocator::CUDAAllocator* MemPool::allocator() {
  return allocator_;
}

int MemPool::use_count() {
  return CUDACachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device() {
  return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

} // namespace c10::cuda
