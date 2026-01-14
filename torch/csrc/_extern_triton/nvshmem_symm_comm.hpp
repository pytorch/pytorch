// nvshmem_symm_comm.hpp
// Header for NVSHMEMSymmComm class - NVSHMEM symmetric memory communication
//
// This provides the declaration for NVSHMEMSymmComm, which is a host-side class
// for managing NVSHMEM symmetric memory communication. The implementation is
// in nvshmem_symm_comm.cpp.

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/_extern_triton/symm_comm.cuh>

#include <memory>
#include <string>

namespace c10d {
namespace symmetric_memory {

/**
 * NVSHMEMSymmComm - Host-side class for NVSHMEM symmetric memory communication.
 *
 * This class manages NVSHMEM resources for symmetric memory operations:
 * - NVSHMEM initialization via the symmetric memory backend
 * - Symmetric memory allocation using the NVSHMEM allocator
 * - Context creation for passing to Triton kernels
 *
 * The class creates an NVSHMEMSymmContext that can be passed to device kernels.
 *
 * Usage from Python:
 *   from torch._C._distributed_c10d import NVSHMEMSymmComm
 *   comm = NVSHMEMSymmComm(group_name, buffer_size, device_idx)
 *   ctx_ptr = comm.get_context_ptr()  # Pass to Triton kernel
 */
class TORCH_API NVSHMEMSymmComm {
 public:
  /**
   * Constructor - Initialize NVSHMEM symmetric memory communication.
   *
   * @param group_name Name of the process group to use
   * @param buffer_size Size of the symmetric buffer in bytes
   * @param device_idx CUDA device index
   */
  NVSHMEMSymmComm(
      const std::string& group_name,
      size_t buffer_size,
      int device_idx);

  /**
   * Destructor - Clean up all NVSHMEM resources.
   */
  ~NVSHMEMSymmComm();

  // Disable copy
  NVSHMEMSymmComm(const NVSHMEMSymmComm&) = delete;
  NVSHMEMSymmComm& operator=(const NVSHMEMSymmComm&) = delete;

  // Disable move (due to device context ownership)
  NVSHMEMSymmComm(NVSHMEMSymmComm&& other) noexcept = delete;
  NVSHMEMSymmComm& operator=(NVSHMEMSymmComm&& other) noexcept = delete;

  /**
   * Get pointer to the device-side context.
   * This pointer can be passed to Triton kernels.
   *
   * @return Device pointer to NVSHMEMSymmContext as int64
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

  // NVSHMEM resources
  void* buffer_ptr_; // Symmetric memory buffer allocated via nvshmem_malloc

  // Context
  NVSHMEMSymmContext context_host_;
  NVSHMEMSymmContext* context_dev_;
};

} // namespace symmetric_memory
} // namespace c10d
