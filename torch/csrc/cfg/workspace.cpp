#include <torch/csrc/cfg/workspace.h>

#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/util/Flags.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

// What alloc_cpu does to fresh memory on request, a kept block needs too.
C10_DECLARE_bool(caffe2_cpu_allocator_do_zero_fill);

namespace torch::cfg {

namespace {

// In front of every block this allocator hands out, so that freeing one
// needs nothing but its address. As large as the alignment of alloc_cpu,
// which the data keeps.
constexpr size_t kHeader = 64;
// Smaller blocks are not kept: malloc has them at hand.
constexpr size_t kSmallest = size_t{1} << 15;
// Blocks are cut out of the memory a workspace holds in multiples of a page.
constexpr size_t kGrain = size_t{1} << 12;
// Memory that no call took a block of in this many calls goes back to malloc.
constexpr uint64_t kUnusedCalls = 3;
// Bytes a workspace may hold whatever its calls needed so far.
constexpr size_t kAlways = size_t{1} << 26;
// The most it holds on top of what its tensors held at once.
constexpr size_t kSpare = size_t{1} << 28;

// A piece of memory from malloc that a workspace cuts blocks out of.
struct Chunk {
  char* memory = nullptr;
  size_t size = 0;
  // Bytes of it that tensors hold, and the last call one of them was cut in.
  size_t used = 0;
  uint64_t call = 0;
  // The ranges of it that nothing holds: where each starts, and how long it
  // is. Ranges that touch are joined, so no two do.
  std::map<size_t, size_t> free;
};

struct Header {
  // Null for a block that malloc made outside of any workspace.
  Workspace* owner;
  Chunk* chunk;
  // How much of the chunk the block takes, this header included.
  size_t length;
};

} // namespace

struct Workspace {
  std::mutex mutex;
  std::vector<std::unique_ptr<Chunk>> chunks;
  // Every free range of every chunk by its length, then chunk and offset:
  // the first one that is long enough is the one that wastes the least.
  std::set<std::tuple<size_t, Chunk*, size_t>> ranges;
  // Bytes of all chunks.
  size_t held = 0;
  // One for whoever made it, one per block that tensors hold.
  size_t references = 1;
  bool closed = false;
  WorkspaceStats stats = {};

  Workspace() = default;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&&) = delete;
  Workspace& operator=(Workspace&&) = delete;
  ~Workspace() {
    for (const auto& chunk : chunks) {
      c10::free_cpu(chunk->memory);
    }
  }

  // The most that is held: what the tensors held at once, and half as much
  // again, up to a fixed amount, for the calls that are not laid out like
  // the last one. A model that trains on gigabytes holds a few percent more
  // with a workspace than its tensors need without one.
  size_t most() const {
    const size_t spare = std::min<size_t>(stats.peak / 2, kSpare);
    return std::max<size_t>(stats.peak + spare, kAlways);
  }

  void addFree(Chunk* chunk, size_t offset, size_t length) {
    chunk->free.emplace(offset, length);
    ranges.emplace(length, chunk, offset);
  }

  size_t dropFree(Chunk* chunk, std::map<size_t, size_t>::iterator range) {
    const size_t length = range->second;
    ranges.erase({length, chunk, range->first});
    chunk->free.erase(range);
    return length;
  }

  // Takes a chunk that nothing is in out of the workspace. Its memory is
  // the caller's to free.
  void* drop(Chunk* chunk) {
    for (const auto& [offset, length] : chunk->free) {
      ranges.erase({length, chunk, offset});
    }
    held -= chunk->size;
    void* memory = chunk->memory;
    chunks.erase(
        std::find_if(chunks.begin(), chunks.end(), [&](const auto& other) {
          return other.get() == chunk;
        }));
    return memory;
  }

  // Lets go of chunks nothing is in, the longest unused first, until `more`
  // bytes fit under the limit with what stays, or there is none left.
  // `freed` gets their memory.
  void shed(size_t more, std::vector<void*>& freed) {
    while (held + more > most()) {
      Chunk* idle = nullptr;
      for (const auto& chunk : chunks) {
        if (chunk->used == 0 && (idle == nullptr || chunk->call < idle->call)) {
          idle = chunk.get();
        }
      }
      if (idle == nullptr) {
        return;
      }
      freed.push_back(drop(idle));
    }
  }

  Header* claim(Chunk* chunk, size_t offset, size_t length) {
    chunk->used += length;
    chunk->call = stats.calls;
    stats.live += length;
    stats.peak = std::max(stats.peak, stats.live);
    // Blocks start where the alignment of the chunk puts a header.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* header = reinterpret_cast<Header*>(chunk->memory + offset);
    header->owner = this;
    header->chunk = chunk;
    header->length = length;
    return header;
  }

  // A block of `need` bytes cut out of the free range that fits it closest,
  // or null if none holds it: then the caller gets `need` bytes from malloc
  // and has them adopted. Either way the caller holds a block of this
  // workspace from now on. `freed` gets the memory of chunks that go so
  // that a new one fits under the limit.
  Header* take(size_t need, std::vector<void*>& freed) {
    std::lock_guard<std::mutex> guard(mutex);
    ++references;
    auto found = ranges.lower_bound({need, nullptr, 0});
    if (found == ranges.end()) {
      ++stats.misses;
      shed(need, freed);
      return nullptr;
    }
    ++stats.hits;
    auto [length, chunk, offset] = *found;
    dropFree(chunk, chunk->free.find(offset));
    // What is left of the range stays free, unless it is too little to be
    // a block.
    if (length - need >= kSmallest) {
      addFree(chunk, offset + need, length - need);
      length = need;
    }
    return claim(chunk, offset, length);
  }

  // Makes `need` bytes from malloc a chunk that one block takes whole.
  Header* adopt(void* memory, size_t need) {
    std::lock_guard<std::mutex> guard(mutex);
    chunks.push_back(std::make_unique<Chunk>());
    Chunk* chunk = chunks.back().get();
    chunk->memory = static_cast<char*>(memory);
    chunk->size = need;
    held += need;
    return claim(chunk, 0, need);
  }

  // For a take that found nothing, when malloc had nothing either. Returns
  // whether the workspace itself is to be deleted.
  bool abandon() {
    std::lock_guard<std::mutex> guard(mutex);
    return --references == 0;
  }

  // Takes a block back: its range is free again, as one with the free
  // ranges next to it. Returns the memory of its chunk if that is the
  // caller's to free (the workspace is closed, and nothing is left in the
  // chunk), and whether the workspace itself is to be deleted.
  std::pair<void*, bool> give(Header* header) {
    std::lock_guard<std::mutex> guard(mutex);
    Chunk* chunk = header->chunk;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    size_t offset = reinterpret_cast<char*>(header) - chunk->memory;
    size_t length = header->length;
    stats.live -= length;
    chunk->used -= length;
    const bool last = --references == 0;
    if (closed && chunk->used == 0) {
      return {drop(chunk), last};
    }
    auto after = chunk->free.find(offset + length);
    if (after != chunk->free.end()) {
      length += dropFree(chunk, after);
    }
    auto before = chunk->free.lower_bound(offset);
    if (before != chunk->free.begin()) {
      --before;
      if (before->first + before->second == offset) {
        offset = before->first;
        length += dropFree(chunk, before);
      }
    }
    addFree(chunk, offset, length);
    return {nullptr, last};
  }

  // The memory of the chunks nothing is in that no recent call used, or of
  // all that nothing is in.
  std::vector<void*> unused(bool all) {
    std::vector<void*> freed;
    for (size_t i = chunks.size(); i-- > 0;) {
      Chunk* chunk = chunks[i].get();
      if (chunk->used == 0 &&
          (all || chunk->call + kUnusedCalls < stats.calls)) {
        freed.push_back(drop(chunk));
      }
    }
    return freed;
  }
};

namespace {

thread_local Workspace* current = nullptr;

void freeAll(const std::vector<void*>& memory) {
  for (void* chunk : memory) {
    c10::free_cpu(chunk);
  }
}

class WorkspaceAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t size) override {
    Workspace* workspace = size >= kSmallest ? current : nullptr;
    Header* header = nullptr;
    if (workspace == nullptr) {
      header = static_cast<Header*>(fresh(size + kHeader, size, nullptr));
      header->owner = nullptr;
    } else {
      const size_t need = (size + kHeader + kGrain - 1) / kGrain * kGrain;
      std::vector<void*> freed;
      header = workspace->take(need, freed);
      // Before more is asked of malloc: what goes may be what it gives.
      freeAll(freed);
      if (header == nullptr) {
        header = workspace->adopt(fresh(need, size, workspace), need);
      }
    }
    static_assert(sizeof(Header) <= kHeader);
    void* block = header;
    void* data = static_cast<char*>(block) + kHeader;
    if (workspace != nullptr && FLAGS_caffe2_cpu_allocator_do_zero_fill) {
      std::memset(data, 0, size);
    }
    c10::profiledCPUMemoryReporter().New(data, size);
    return {data, data, &free, c10::Device(c10::DeviceType::CPU)};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &free;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

 private:
  // `bytes` from malloc, for a tensor of `size` bytes; `workspace` is the
  // one that was asked first and holds a block for it.
  static void* fresh(size_t bytes, size_t size, Workspace* workspace) {
    try {
      return c10::alloc_cpu(bytes);
    } catch (c10::Error&) {
      if (workspace != nullptr && workspace->abandon()) {
        delete workspace;
      }
      c10::profiledCPUMemoryReporter().OutOfMemory(size);
      throw;
    }
  }

  static void free(void* data) {
    if (data == nullptr) {
      return;
    }
    c10::profiledCPUMemoryReporter().Delete(data);
    void* block = static_cast<char*>(data) - kHeader;
    auto* header = static_cast<Header*>(block);
    Workspace* workspace = header->owner;
    if (workspace == nullptr) {
      c10::free_cpu(block);
      return;
    }
    const auto [memory, last] = workspace->give(header);
    if (memory != nullptr) {
      c10::free_cpu(memory);
    }
    if (last) {
      delete workspace;
    }
  }
};

// Whether tensors get their memory from the allocator above. Decided once.
bool installed() {
  static const bool done = [] {
    static WorkspaceAllocator allocator;
    if (c10::GetCPUAllocator() != c10::GetDefaultCPUAllocator()) {
      return false;
    }
    c10::SetCPUAllocator(&allocator, /*priority=*/1);
    return c10::GetCPUAllocator() == &allocator;
  }();
  return done;
}

} // namespace

Workspace* workspaceNew() {
  return installed() ? new Workspace() : nullptr;
}

void workspaceFree(Workspace* workspace) {
  std::vector<void*> freed;
  bool last = false;
  {
    std::lock_guard<std::mutex> guard(workspace->mutex);
    workspace->closed = true;
    freed = workspace->unused(/*all=*/true);
    last = --workspace->references == 0;
  }
  freeAll(freed);
  if (last) {
    delete workspace;
  }
}

Workspace* workspaceEnter(Workspace* workspace) {
  Workspace* previous = current;
  current = workspace;
  return previous;
}

void workspaceLeave(Workspace* workspace, Workspace* previous) {
  current = previous;
  if (workspace == previous) {
    return;
  }
  std::vector<void*> freed;
  {
    std::lock_guard<std::mutex> guard(workspace->mutex);
    ++workspace->stats.calls;
    freed = workspace->unused(/*all=*/false);
  }
  freeAll(freed);
}

WorkspaceStats workspaceStats(Workspace* workspace) {
  std::lock_guard<std::mutex> guard(workspace->mutex);
  WorkspaceStats stats = workspace->stats;
  stats.kept = workspace->held - stats.live;
  return stats;
}

} // namespace torch::cfg
