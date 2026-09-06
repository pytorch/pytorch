#pragma once
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

namespace c10d {
namespace symmetric_memory {

class NCCLPeerAllocInfo;

// Host-side CFT (Compute Fabric Transport) logical-endpoint coordinates.
// `(le_id, le_offset)` is exactly the pair that the device-side `ncclCft`
// put/get/red family takes, so a custom kernel can address peer memory over
// CFT without building a `ncclDevComm`. `le_offset` is a plain byte offset:
// advancing into the buffer is just `le_offset + n`.
//
// A handle is only valid for the group its NCCLSymmetricMemory belongs to.
// Every process group registers its own window over the allocation and owns a
// separate set of logical endpoints, so rendezvousing one tensor with two
// groups yields two unrelated handles. Nothing checks this at use time --
// crossing them silently addresses the wrong memory.
struct NCCLCftHandle {
  uint32_t le_id;
  size_t le_offset;
};

// TORCH_API because torch_python downcasts to it for the CFT handle bindings.
class TORCH_API NCCLSymmetricMemory : public SymmetricMemory {
 public:
  NCCLSymmetricMemory(c10::intrusive_ptr<NCCLPeerAllocInfo> pai, size_t offset);

  ~NCCLSymmetricMemory() override = default;

  std::vector<void*> get_buffer_ptrs() override;

  std::vector<void*> get_signal_pad_ptrs() override;

  void** get_buffer_ptrs_dev() override;

  void** get_signal_pad_ptrs_dev() override;

  size_t get_buffer_size() override;

  std::string get_group_name();

  bool has_multicast_support() override;

  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;

  int get_world_size() override;

  c10::Device get_device() override;

  ncclWindow_t get_window();

  // CFT handle addressing `peer`'s copy of this buffer. Requires the group's
  // communicator to have been created with `hostCftMode` enabled and the peer
  // to be inside the flat CFT team.
  NCCLCftHandle get_peer_cft_handle(int peer);

  // CFT handle addressing the multicast (multimem) view of this buffer, for
  // the `putMultimem` / `redMultimem` device ops. Unlike the unicast query,
  // the first call is collective over the group unless the multicast endpoint
  // was already created at window-registration time (i.e. `hostCftMode` on).
  NCCLCftHandle get_multimem_cft_handle();

  size_t get_offset() override;

  // Byte offset of this handle's data within the NCCL window. The window
  // starts at the signal pad, so this is buffer_offset + get_offset().
  size_t get_window_offset();

 private:
  c10::intrusive_ptr<NCCLPeerAllocInfo> pai_;
  size_t offset_;
  int rank_;
  int world_size_;
  int device_idx_;
};

} // namespace symmetric_memory
} // namespace c10d
#endif // NCCL_HAS_SYMMEM_SUPPORT
