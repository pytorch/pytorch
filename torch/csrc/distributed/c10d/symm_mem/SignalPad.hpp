#pragma once

#include <c10/util/intrusive_ptr.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace c10d::symmetric_memory {

// The internal signal pad of one (process group, device): the synchronization
// state every built-in symmetric-memory operation of the group uses on that
// device, whichever allocation it runs on. Ops sharing a pad then share a
// group, so GroupStreamGuard's per-group ordering covers all of them, except
// the replay of a captured graph. get_signal_pad() does not return it; that is
// the allocation's pad, for kernels outside PyTorch.
//
// Immutable once built and shared by every handle of the group. `owner` keeps
// the mapped memory alive; how is the backend's business.
class SignalPad {
 public:
  SignalPad(
      c10::intrusive_ptr<c10::intrusive_ptr_target> owner,
      std::vector<void*> peers,
      void** peers_dev,
      void* multicast,
      size_t size,
      uint64_t peer_mapping_id)
      : owner_(std::move(owner)),
        peers_(std::move(peers)),
        peers_dev_(peers_dev),
        multicast_(multicast),
        size_(size),
        peer_mapping_id_(peer_mapping_id) {}

  // Each rank's pad, indexed by group rank.
  const std::vector<void*>& peers() const {
    return peers_;
  }
  // The same pointers in a device array of size world_size.
  void** peers_dev() const {
    return peers_dev_;
  }
  // This rank's view of the multicast mapping, or nullptr without multicast.
  void* multicast() const {
    return multicast_;
  }
  // Bytes of the channel area. Fixed when the pad is created;
  // set_signal_pad_size() only affects pads created later.
  size_t size() const {
    return size_;
  }
  // Identifies the device of every rank when the pad was created. The pad
  // pairs those devices only; a rendezvous with another mapping must not use
  // it.
  uint64_t peer_mapping_id() const {
    return peer_mapping_id_;
  }

 private:
  c10::intrusive_ptr<c10::intrusive_ptr_target> owner_;
  std::vector<void*> peers_;
  void** peers_dev_;
  void* multicast_;
  size_t size_;
  uint64_t peer_mapping_id_;
};

// Every word of synchronization state belongs to one protocol and one scope:
//
//   state                      scope       protocol            at rest
//   slot (channel, src)        group pad   0/1 compare-swap    0
//   multimem barrier counter   group pad   add, wait, subtract 0
//
// A group's pad is the channel area, `size` bytes of world_size slots per
// channel indexed world_size * channel + src, as an allocation's pad is; then
// one barrier counter per channel.
constexpr size_t signal_pad_barrier_state_offset(size_t size) {
  return (size + sizeof(uint32_t) - 1) / sizeof(uint32_t) * sizeof(uint32_t);
}

constexpr size_t signal_pad_alloc_size(size_t size, size_t world_size) {
  return signal_pad_barrier_state_offset(size) +
      size / (sizeof(uint32_t) * world_size) * sizeof(uint32_t);
}

} // namespace c10d::symmetric_memory
