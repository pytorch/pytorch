#include "THCCachingHostAllocator.h"

#include <cuda_runtime_api.h>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <utility>


namespace {

typedef std::shared_ptr<THCStream> THCStreamPtr;

struct BlockSize
{
  size_t  size; // allocation size
  void*   ptr;  // host memory pointer

  BlockSize(size_t size, void* ptr=NULL) : size(size), ptr(ptr) {}
};

struct Block : public BlockSize
{
  bool  allocated;    // true if the block is currently allocated
  int   event_count;  // number of outstanding cuda events
  std::set<THCStreamPtr> streams;

  Block(size_t size, void* ptr, bool allocated) :
      BlockSize(size, ptr), allocated(allocated), event_count(0), streams() {}
};

static bool BlockComparator(const BlockSize& a, const BlockSize& b)
{
  // sort by size, break ties with pointer
  if (a.size != b.size) {
    return a.size < b.size;
  }
  return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
}

struct HostAllocator
{
  typedef bool (*Comparison)(const BlockSize&, const BlockSize&);

  // lock around all operations
  std::mutex mutex;

  // blocks by pointer
  std::unordered_map<void*, Block> blocks;

  // pointers that are ready to be allocated (event_count=0)
  std::set<BlockSize, Comparison> available;

  // outstanding cuda events
  std::deque<std::pair<cudaEvent_t, void*>> cuda_events;

  HostAllocator() : available(BlockComparator) {}

  cudaError_t malloc(void** ptr, size_t size)
  {
    std::lock_guard<std::mutex> lock(mutex);

    // process outstanding cuda events which may have occurred
    cudaError_t err = processEvents();
    if (err != cudaSuccess) {
      return err;
    }

    // search for the smallest block which can hold this allocation
    BlockSize search_key(size);
    auto it = available.lower_bound(search_key);
    if (it != available.end()) {
      Block& block = blocks.at(it->ptr);
      THAssert(!block.allocated && block.event_count == 0);
      block.allocated = true;
      *ptr = block.ptr;
      available.erase(it);
      return cudaSuccess;
    }

    // note that cudaHostAlloc may not touch pointer if size is 0
    *ptr = 0;

    // allocate a new block if no cached allocation is found
    err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
      return err;
    }

    blocks.insert({*ptr, Block(size, *ptr, true)});
    return cudaSuccess;
  }

  cudaError_t free(void* ptr)
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (!ptr) {
      return cudaSuccess;
    }

    // process outstanding cuda events which may have occurred
    cudaError_t err = processEvents();
    if (err != cudaSuccess) {
      return err;
    }

    auto it = blocks.find(ptr);
    THAssert(it != blocks.end());

    Block& block = it->second;
    THAssert(block.allocated);

    // free (on valid memory) shouldn't fail, so mark unallocated before
    // we process the streams.
    block.allocated = false;

    // insert CUDA events for each stream on which this block was used. This
    err = insertEvents(block);
    if (err != cudaSuccess) {
      return err;
    }

    if (block.event_count == 0) {
      // the block can be re-used if there are no outstanding cuda events
      available.insert(block);
    }
    return cudaSuccess;
  }

  cudaError_t recordEvent(void* ptr, THCStream *stream)
  {
    std::lock_guard<std::mutex> lock(mutex);
    cudaError_t err;

    auto it = blocks.find(ptr);
    if (it == blocks.end()) {
      // ignore events for untracked pointers
      return cudaSuccess;
    }

    Block& block = it->second;
    THAssert(block.allocated);

    THCStreamPtr stream_ptr(stream, &THCStream_free);
    THCStream_retain(stream);

    block.streams.insert(std::move(stream_ptr));
    return cudaSuccess;
  }

  cudaError_t processEvents()
  {
    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!cuda_events.empty()) {
      auto& e = cuda_events.front();
      cudaEvent_t event = e.first;

      cudaError_t err = cudaEventQuery(event);
      if (err == cudaErrorNotReady) {
        break;
      } else if (err != cudaSuccess) {
        return err;
      }
      err = cudaEventDestroy(event);
      if (err != cudaSuccess) {
        return err;
      }

      Block& block = blocks.at(e.second);
      block.event_count--;
      if (block.event_count == 0 && !block.allocated) {
        available.insert(block);
      }
      cuda_events.pop_front();
    }
    return cudaSuccess;
  }

  void emptyCache()
  {
    std::lock_guard<std::mutex> lock(mutex);

    // remove events for freed blocks
    for (auto it = cuda_events.begin(); it != cuda_events.end(); ++it) {
      cudaEvent_t event = it->first;
      Block& block = blocks.at(it->second);
      if (!block.allocated) {
        THCudaCheckWarn(cudaEventDestroy(event));
        block.event_count--;
      }
    }

    // all cuda_events have been processed
    cuda_events.clear();

    // clear list of available blocks
    available.clear();

    // free and erase non-allocated blocks
    for (auto it = blocks.begin(); it != blocks.end();) {
      Block& block = it->second;
      if (!block.allocated) {
        THCudaCheckWarn(cudaFreeHost(block.ptr));
        it = blocks.erase(it);
      } else {
        ++it;
      }
    }
  }

  cudaError_t insertEvents(Block& block)
  {
    cudaError_t err;

    int prev_device;
    err = cudaGetDevice(&prev_device);
    if (err != cudaSuccess) return err;

    std::set<THCStreamPtr> streams(std::move(block.streams));
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      auto& stream = *it;

      err = cudaSetDevice(stream->device);
      if (err != cudaSuccess) break;

      cudaEvent_t event;
      err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (err != cudaSuccess) break;

      err = cudaEventRecord(event, stream->stream);
      if (err != cudaSuccess) break;

      block.event_count++;
      cuda_events.emplace_back(event, block.ptr);
    }

    cudaSetDevice(prev_device);
    return err;
  }
};

}  // namespace

static HostAllocator allocator;

static void* THCCachingHostAllocator_malloc(void* ctx, ptrdiff_t size)
{
  THAssert(size >= 0);
  void *ptr;
  THCudaCheck(allocator.malloc(&ptr, size));
  return ptr;
}

static void THCCachingHostAllocator_free(void* ctx, void* ptr)
{
  allocator.free(ptr);
}

cudaError_t THCCachingHostAllocator_recordEvent(void *ptr, THCStream *stream)
{
  return allocator.recordEvent(ptr, stream);
}

void THCCachingHostAllocator_emptyCache()
{
  allocator.emptyCache();
}

THAllocator THCCachingHostAllocator = {
  &THCCachingHostAllocator_malloc,
  NULL,
  &THCCachingHostAllocator_free,
};
