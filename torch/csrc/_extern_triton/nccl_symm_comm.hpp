// nccl_symm_comm.hpp
// Header for NCCLSymmComm class - NCCL symmetric memory communication
//
// This provides the declaration for NCCLSymmComm, which is a host-side class
// for managing NCCL symmetric memory communication. The implementation is
// in nccl_symm_comm.cpp.

#pragma once

#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

#include <torch/csrc/Export.h>
#include <torch/csrc/_extern_triton/symm_comm.cuh>

#include <memory>
#include <string>

// Forward declarations for NCCL types
struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace c10d {
namespace symmetric_memory {

/**
 * NCCLSymmComm - Host-side class for NCCL symmetric memory communication.
 *
 * This class directly manages all NCCL resources without relying on
 * NCCLDevCommManager. It handles:
 * - NCCL initialization
 * - Symmetric memory allocation using ncclMemAlloc
 * - Window registration using ncclCommWindowRegister
 * - Device communicator creation using ncclDevCommCreate
 * - Proper cleanup with ncclDevCommDestroy
 *
 * The class creates an NCCLSymmContext that can be passed to device kernels.
 *
 * Usage from Python:
 *   from torch._C._distributed_c10d import NCCLSymmComm
 *   comm = NCCLSymmComm(group_name, buffer_size, device_idx)
 *   ctx_ptr = comm.get_context_ptr()  # Pass to Triton kernel
 */
class TORCH_API NCCLSymmComm {
 public:
  /**
   * Constructor - Initialize NCCL symmetric memory communication.
   *
   * @param group_name Name of the process group to use
   * @param buffer_size Size of the symmetric buffer in bytes
   * @param device_idx CUDA device index
   */
  NCCLSymmComm(
      const std::string& group_name,
      size_t buffer_size,
      int device_idx);

  /**
   * Destructor - Clean up all NCCL resources.
   */
  ~NCCLSymmComm();

  // Disable copy
  NCCLSymmComm(const NCCLSymmComm&) = delete;
  NCCLSymmComm& operator=(const NCCLSymmComm&) = delete;

  // Disable move (due to device communicator ownership)
  NCCLSymmComm(NCCLSymmComm&& other) noexcept = delete;
  NCCLSymmComm& operator=(NCCLSymmComm&& other) noexcept = delete;

  /**
   * Get pointer to the device-side context.
   * This pointer can be passed to Triton kernels.
   *
   * @return Device pointer to NCCLSymmContext as int64
   */
  int64_t get_context_ptr() const;

  /**
   * Get pointer to the local buffer.
   *
   * @return Device pointer to the symmetric buffer as int64
   */
  int64_t get_buffer_ptr() const;

  /**
   * Get the buffer size in bytes.
   *
   * @return Buffer size
   */
  size_t get_buffer_size() const;

  /**
   * Get the rank of this process.
   *
   * @return Rank
   */
  int get_rank() const;

  /**
   * Get the world size (number of processes).
   *
   * @return World size
   */
  int get_world_size() const;

  /**
   * Get the device index.
   *
   * @return Device index
   */
  int get_device_idx() const;

 private:
  void initialize();
  void cleanup();

  // Configuration
  std::string group_name_;
  size_t buffer_size_;
  int device_idx_;
  int rank_;
  int world_size_;

  // NCCL resources (opaque pointers in header)
  void* buffer_ptr_;
  void* signal_pad_ptr_;
  void* buffer_window_; // ncclWindow_t
  void* signal_window_; // ncclWindow_t

  // Host communicator (borrowed from ProcessGroupNCCL)
  ncclComm_t comm_;

  // Device communicator (managed internally)
  void* dev_comm_storage_; // Storage for ncclDevComm
  bool owns_dev_comm_;

  // Context
  NCCLSymmContext context_host_;
  NCCLSymmContext* context_dev_;
};

} // namespace symmetric_memory
} // namespace c10d

#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
