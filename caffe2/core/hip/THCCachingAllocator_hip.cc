#include "caffe2/core/hip/THCCachingAllocator_hip.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

#include <hip/hip_runtime_api.h>

#include "caffe2/core/hip/context_hip.h"

//
// Yet another caching allocator for HIP device allocations.
//
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to hipMalloc.
// - If the hipMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocation requests are handled separately. Large
//   allocation requests can be filled by a hipMalloc call of the exact size.
//   Small requests will allocate and split a 1MB buffer, if necessary.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// Thread Safety: the allocator is NOT thread safe. Calls to { Alloc, Free }
// must be synchronized by the programmer.
//

namespace {

const size_t kRoundSmall = 512; // round up small allocs to 512 bytes
const size_t kRoundLarge = 131072; // round up large allocs to 128 KiB
const size_t kSmallAlloc = 1048576; // largest "small" allocation is 1 MiB

struct Block {
  int device; // gpu
  hipStream_t stream; // allocation stream
  size_t size; // block size in bytes
  char* ptr; // memory address
  bool allocated; // in-use flag
  Block* prev; // prev block if split from a larger allocation
  Block* next; // next block if split from a larger allocation
  int event_count; // number of outstanding HIP events

  Block(int device, hipStream_t stream, size_t size, char* ptr = nullptr)
      : device(device),
        stream(stream),
        size(size),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        event_count(0) {}
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->device != b->device) {
    return a->device < b->device;
  }
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static size_t roundSize(size_t size) {
  if (size < kRoundSmall) {
    size = kRoundSmall;
  } else if (size < kSmallAlloc) {
    size += kRoundSmall - 1 - (size - 1) % kRoundSmall;
  } else {
    size += kRoundLarge - 1 - (size - 1) % kRoundLarge;
  }
  return size;
}

} // namespace

namespace caffe2 {

struct THCCachingAllocatorImpl {
  typedef bool (*Comparison)(const Block*, const Block*);
  typedef std::set<Block*, Comparison> FreeBlocks;

  // lock around all operations
  std::mutex mutex;

  // cached blocks larger than 1 MB
  FreeBlocks largeBlocks_;

  // cached blocks 1 MB or smaller
  FreeBlocks smallBlocks_;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocatedBlocks_;

  THCCachingAllocatorImpl()
      : largeBlocks_(BlockComparator), smallBlocks_(BlockComparator) {}

  ~THCCachingAllocatorImpl() {
    emptyCache();
  }

  /** allocates a block which is safe to use from the provided stream */
  hipError_t Alloc(void** devPtr, size_t size, hipStream_t stream) {
    int device;
    hipError_t err = hipGetDevice(&device);
    if (err != hipSuccess) {
      return err;
    }

    size = roundSize(size);
    bool small = size <= kSmallAlloc;

    Block search_key(device, stream, size);
    auto& free_blocks = small ? smallBlocks_ : largeBlocks_;

    Block* block = nullptr;
    Block* remaining = nullptr;

    auto it = free_blocks.lower_bound(&search_key);
    if (it != free_blocks.end() && (*it)->device == device &&
        (*it)->stream == stream) {
      block = *it;
      free_blocks.erase(it);
    } else {
      void* ptr;
      size_t alloc_size = small ? kSmallAlloc : size;
      err = hipMallocRetry(device, &ptr, alloc_size);
      if (err != hipSuccess) {
        return err;
      }
      block = new Block(device, stream, alloc_size, (char*)ptr);
    }

    if (block->size - size >= (small ? kRoundSmall : kSmallAlloc + 1)) {
      remaining = block;

      block = new Block(device, stream, size, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr += size;
      remaining->size -= size;
      free_blocks.insert(remaining);
    }

    block->allocated = true;
    allocatedBlocks_[block->ptr] = block;

    *devPtr = (void*)block->ptr;
    return hipSuccess;
  }

  hipError_t Free(void* ptr) {
    if (!ptr) {
      return hipSuccess;
    }

    auto it = allocatedBlocks_.find(ptr);
    if (it == allocatedBlocks_.end()) {
      return hipErrorInvalidDevicePointer;
    }

    Block* block = it->second;
    allocatedBlocks_.erase(it);
    block->allocated = false;

    freeBlock(block);
    return hipSuccess;
  }

  /** returns cached blocks to the system allocator */
  hipError_t emptyCache() {
    hipError_t err =
        freeBlocks(largeBlocks_, largeBlocks_.begin(), largeBlocks_.end());
    if (err != hipSuccess) {
      return err;
    }
    err = freeBlocks(smallBlocks_, smallBlocks_.begin(), smallBlocks_.end());
    if (err != hipSuccess) {
      return err;
    }
    return hipSuccess;
  }

  /** moves a block into the free block list */
  void freeBlock(Block* block) {
    CAFFE_ENFORCE(!block->allocated && block->event_count == 0);
    bool small = block->size <= kSmallAlloc;
    auto& free_blocks = small ? smallBlocks_ : largeBlocks_;
    tryMergeBlocks(block, block->prev, free_blocks);
    tryMergeBlocks(block, block->next, free_blocks);
    free_blocks.insert(block);
  }

  /** combine previously split blocks */
  void tryMergeBlocks(Block* dst, Block* src, FreeBlocks& free_blocks) {
    if (!src || src->allocated || src->event_count > 0) {
      return;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    dst->size += src->size;
    free_blocks.erase(src);
    delete src;
  }

  hipError_t hipMallocRetry(int device, void** devPtr, size_t size) {
    // Try hipMalloc. If hipMalloc fails, frees all non-split cached blocks
    // and retries.
    hipError_t err = hipMalloc(devPtr, size);
    if (err != hipSuccess) {
      hipGetLastError();
      err = freeCachedBlocks(device);
      if (err != hipSuccess) {
        return err;
      }
      err = hipMalloc(devPtr, size);
      if (err != hipSuccess) {
        return err;
      }
    }
    return hipSuccess;
  }

  hipError_t freeCachedBlocks(int device) {
    // Free all non-split cached blocks on device
    Block lower_bound(device, nullptr, 0);
    Block upper_bound(device + 1, nullptr, 0);

    hipError_t err = freeBlocks(
        largeBlocks_,
        largeBlocks_.lower_bound(&lower_bound),
        largeBlocks_.lower_bound(&upper_bound));
    if (err != hipSuccess) {
      return err;
    }
    err = freeBlocks(
        smallBlocks_,
        smallBlocks_.lower_bound(&lower_bound),
        smallBlocks_.lower_bound(&upper_bound));
    return err;
  }

  hipError_t freeBlocks(
      FreeBlocks& blocks,
      FreeBlocks::iterator it,
      FreeBlocks::iterator end) {
    // Frees all non-split blocks between `it` and `end`
    while (it != end) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        hipError_t err = hipFree((void*)block->ptr);
        if (err != hipSuccess) {
          return err;
        }
        auto cur = it;
        ++it;
        blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
    return hipSuccess;
  }
};

THCCachingAllocator::THCCachingAllocator()
    : _impl(new THCCachingAllocatorImpl()) {}

THCCachingAllocator::~THCCachingAllocator() {
  delete _impl;
}

hipError_t
THCCachingAllocator::Alloc(void** refPtr, size_t nbytes, hipStream_t stream) {
  return _impl->Alloc(refPtr, nbytes, stream);
}

hipError_t THCCachingAllocator::Free(void* ptr) {
  return _impl->Free(ptr);
}

} // namespace caffe2
