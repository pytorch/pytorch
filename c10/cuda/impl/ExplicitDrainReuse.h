#pragma once

#include <cstddef>
#include <stdexcept>

namespace c10::cuda::CUDACachingAllocator::detail {

// An intrusive index of CCA-owned Blocks, NOT another CUDA event/segment owner.
// All calls require the containing DeviceCachingAllocator's mutex.
template <typename Block>
struct ExplicitDrainReuse {
  bool enabled{false};
  bool disabled{false};
  bool publishing{false};
  Block* held_head{nullptr};
  std::size_t held_bytes{0};

  bool hold(Block* block) noexcept {
    if (!enabled || !block->explicit_drain_quarantined ||
        (publishing && !disabled)) {
      return false;
    }
    if (!block->explicit_drain_held) {
      block->explicit_drain_next = held_head;
      held_head = block;
      block->explicit_drain_held = true;
      held_bytes += block->size;
    }
    return true;
  }

  template <typename Wait, typename Publish>
  void drain(Wait&& wait, Publish&& publish) {
    if (!enabled || disabled || publishing) {
      throw std::logic_error("explicit-drain pool is unavailable");
    }
    try {
      wait(); // Complete the whole wait phase before allowing ANY publication.
      publishing = true;
      while (held_head) {
        Block* block = held_head;
        Block* next = block->explicit_drain_next;
        const auto bytes = block->size;
        publish(block);
        held_head = next;
        held_bytes -= bytes;
        block->explicit_drain_next = nullptr;
        block->explicit_drain_held = false;
        block->explicit_drain_quarantined = false;
      }
      publishing = false;
    } catch (...) {
      publishing = false;
      disabled = true;
      throw; // No rollback or cross-rank recovery promise.
    }
  }
};

} // namespace c10::cuda::CUDACachingAllocator::detail
