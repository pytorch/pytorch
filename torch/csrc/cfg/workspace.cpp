#include <torch/csrc/cfg/workspace.h>

#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/util/Flags.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
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
// A block that no call asked for in this many calls goes back to malloc.
constexpr uint64_t kUnusedCalls = 3;
// Bytes a workspace may hold whatever its calls needed so far.
constexpr size_t kAlways = size_t{1} << 26;

struct Header {
  Workspace* owner;
  // How large the block is, which may be more than was asked for.
  size_t size;
};

struct Kept {
  void* block;
  uint64_t call;
};

} // namespace

struct Workspace {
  std::mutex mutex;
  // By size. A model asks for the same sizes again, and where it does not
  // (every layer of a DenseNet is a little wider than the last) a block
  // that is somewhat larger does as well: its pages are there already.
  std::map<size_t, std::vector<Kept>> kept;
  // One for whoever made it, one per block that tensors hold.
  size_t references = 1;
  bool closed = false;
  WorkspaceStats stats = {};

  // The most that is held for later: what tensors held at once, and half
  // as much again for blocks that did not fit what was asked next.
  size_t most() const {
    return std::max<size_t>(stats.peak + stats.peak / 2, kAlways);
  }

  // A block of at least `size` bytes that was kept, or null; `size` becomes
  // what the block holds. Either way the caller holds a block of this
  // workspace from now on.
  void* take(size_t& size) {
    std::lock_guard<std::mutex> guard(mutex);
    ++references;
    void* block = nullptr;
    auto found = kept.lower_bound(size);
    if (found != kept.end() && found->first <= size + size) {
      size = found->first;
      block = found->second.back().block;
      found->second.pop_back();
      if (found->second.empty()) {
        kept.erase(found);
      }
      stats.kept -= size;
      ++stats.hits;
    } else {
      ++stats.misses;
    }
    stats.live += size;
    stats.peak = std::max(stats.peak, stats.live);
    return block;
  }

  // Takes a block back. Returns the blocks that are the caller's to free:
  // this one if it is not kept, or smaller ones that made room for it,
  // which are the cheaper to make again. The second value says that the
  // workspace itself is to be deleted.
  std::pair<std::vector<void*>, bool> give(void* block, size_t size) {
    std::lock_guard<std::mutex> guard(mutex);
    std::vector<void*> blocks;
    stats.live -= size;
    const bool last = --references == 0;
    if (block == nullptr) {
      return {blocks, last};
    }
    if (closed) {
      blocks.push_back(block);
      return {blocks, last};
    }
    while (stats.kept + size > most() && !kept.empty() &&
           kept.begin()->first < size) {
      auto smallest = kept.begin();
      blocks.push_back(smallest->second.back().block);
      smallest->second.pop_back();
      stats.kept -= smallest->first;
      if (smallest->second.empty()) {
        kept.erase(smallest);
      }
    }
    if (stats.kept + size > most()) {
      blocks.push_back(block);
    } else {
      kept[size].push_back({block, stats.calls});
      stats.kept += size;
    }
    return {blocks, last};
  }

  // The blocks no recent call used, or all of them.
  std::vector<void*> unused(bool all) {
    std::vector<void*> blocks;
    for (auto entry = kept.begin(); entry != kept.end();) {
      auto& [size, entries] = *entry;
      auto stale = [&](const Kept& block) {
        return all || block.call + kUnusedCalls < stats.calls;
      };
      for (const Kept& block : entries) {
        if (stale(block)) {
          blocks.push_back(block.block);
          stats.kept -= size;
        }
      }
      entries.erase(
          std::remove_if(entries.begin(), entries.end(), stale), entries.end());
      entry = entries.empty() ? kept.erase(entry) : std::next(entry);
    }
    return blocks;
  }
};

namespace {

thread_local Workspace* current = nullptr;

void freeBlocks(const std::vector<void*>& blocks) {
  for (void* block : blocks) {
    c10::free_cpu(block);
  }
}

class WorkspaceAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t size) override {
    Workspace* workspace = size >= kSmallest ? current : nullptr;
    // What the block holds: more than `size` if a larger one was kept.
    size_t held = size;
    void* block = workspace != nullptr ? workspace->take(held) : nullptr;
    if (block == nullptr) {
      try {
        block = c10::alloc_cpu(size + kHeader);
      } catch (c10::Error&) {
        if (workspace != nullptr) {
          release(workspace, nullptr, held);
        }
        c10::profiledCPUMemoryReporter().OutOfMemory(size);
        throw;
      }
    } else if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
      std::memset(block, 0, size + kHeader);
    }
    auto* header = static_cast<Header*>(block);
    header->owner = workspace;
    header->size = held;
    void* data = static_cast<char*>(block) + kHeader;
    c10::profiledCPUMemoryReporter().New(data, size);
    return {data, data, &free, c10::Device(c10::DeviceType::CPU)};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &free;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

  static void release(Workspace* workspace, void* block, size_t size) {
    const auto [blocks, last] = workspace->give(block, size);
    for (void* unkept : blocks) {
      c10::free_cpu(unkept);
    }
    if (last) {
      delete workspace;
    }
  }

 private:
  static void free(void* data) {
    if (data == nullptr) {
      return;
    }
    c10::profiledCPUMemoryReporter().Delete(data);
    void* block = static_cast<char*>(data) - kHeader;
    const auto* header = static_cast<const Header*>(block);
    if (header->owner == nullptr) {
      c10::free_cpu(block);
    } else {
      release(header->owner, block, header->size);
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
  std::vector<void*> blocks;
  bool last = false;
  {
    std::lock_guard<std::mutex> guard(workspace->mutex);
    workspace->closed = true;
    blocks = workspace->unused(/*all=*/true);
    last = --workspace->references == 0;
  }
  freeBlocks(blocks);
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
  std::vector<void*> blocks;
  {
    std::lock_guard<std::mutex> guard(workspace->mutex);
    ++workspace->stats.calls;
    blocks = workspace->unused(/*all=*/false);
  }
  freeBlocks(blocks);
}

WorkspaceStats workspaceStats(Workspace* workspace) {
  std::lock_guard<std::mutex> guard(workspace->mutex);
  return workspace->stats;
}

} // namespace torch::cfg
