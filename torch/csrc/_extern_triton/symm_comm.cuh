// symm_comm.cuh
// Unified header for symmetric memory communication contexts
//
// This file provides:
// 1. SymmContext - Abstract base class for symmetric communication
// 2. NCCLSymmContext - NCCL-specific context (uses NCCL LSA API)
// 3. NVSHMEMSymmContext - NVSHMEM-specific context (uses NVSHMEM API)
//
// The unified design allows kernels to work with either backend through
// runtime type dispatch.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// =============================================================================
// DEVICE-SIDE CHECK MACROS
// =============================================================================

/**
 * TORCH_SYMM_CHECK - Device-side check macro for symmetric memory operations.
 *
 * Uses __assert_fail which is available in CUDA device code. This will trigger
 * a device-side assertion failure that will be caught at runtime and cause
 * the kernel to abort.
 *
 * Note: __assert_fail is a CUDA intrinsic that triggers a trap instruction.
 * The message will appear in the CUDA error output when running with
 * cuda-memcheck or similar tools.
 */
#ifdef __CUDA_ARCH__
extern "C" __device__ void __assertfail(
    const char* message,
    const char* file,
    unsigned int line,
    const char* function,
    size_t charSize);

#define TORCH_SYMM_CHECK(condition, message)                             \
  do {                                                                   \
    if (!(condition)) {                                                  \
      __assertfail(message, __FILE__, __LINE__, __PRETTY_FUNCTION__, 1); \
    }                                                                    \
  } while (0)
#else
// Host-side fallback (should not be used in device code)
#include <cassert>
#define TORCH_SYMM_CHECK(condition, message) assert((condition) && (message))
#endif

// =============================================================================
// NCCL TYPE DEFINITIONS
// =============================================================================

// Include NCCL headers for proper type definitions when available
#if __has_include(<nccl.h>)
#include <nccl.h>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include <nccl_device.h>
#define NCCL_SYMM_TYPES_AVAILABLE
#endif
#endif

// Fallback forward declarations only if NCCL types are not available
#ifndef NCCL_SYMM_TYPES_AVAILABLE
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ncclComm* ncclComm_t;
typedef void* ncclWindow_t;
typedef struct ncclDevComm ncclDevComm;
#ifdef __cplusplus
}
#endif
#endif

// =============================================================================
// NVSHMEM DEVICE FUNCTION DECLARATIONS
// These are provided by libnvshmem_device.bc and linked at compile time
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Get pointer to symmetric memory on a remote PE
extern __device__ void* nvshmem_ptr(const void* dest, int pe);

// Barrier synchronization across all PEs (block-level)
extern __device__ void nvshmemx_barrier_all_block();

// Get the calling PE's number
extern __device__ int nvshmem_my_pe();

// Get the number of PEs
extern __device__ int nvshmem_n_pes();

// Float put (single element)
extern __device__ void nvshmem_float_p(float* dest, float value, int pe);

// Float get (single element)
extern __device__ float nvshmem_float_g(const float* source, int pe);

// Block-level put (for larger transfers)
extern __device__ void nvshmemx_float_put_block(
    float* dest,
    const float* source,
    size_t nelems,
    int pe);

// Block-level get (for larger transfers)
extern __device__ void nvshmemx_float_get_block(
    float* dest,
    const float* source,
    size_t nelems,
    int pe);

// Block-level memory put (generic, byte-oriented)
// dest: Symmetric address on the destination PE
// source: Local address of data to send
// nelems: Number of bytes to transfer
// pe: Destination PE number
extern __device__ void nvshmemx_putmem_block(
    void* dest,
    const void* source,
    size_t nelems,
    int pe);

// Non-blocking immediate memory put (generic, byte-oriented)
// dest: Symmetric address on the destination PE
// source: Local address of data to send
// nelems: Number of bytes to transfer
// pe: Destination PE number
// Returns immediately without waiting for completion. Use nvshmem_quiet() to
// ensure delivery.
extern __device__ void nvshmem_putmem_nbi(
    void* dest,
    const void* source,
    size_t nelems,
    int pe);

// Block-level memory put with signal (generic, byte-oriented)
// dest: Symmetric address on the destination PE
// source: Local address of data to send
// nelems: Number of bytes to transfer
// sig_addr: Symmetric address of the signal variable on the destination PE
// signal: Value to be used in the signal operation
// sig_op: Signal operation type (NVSHMEM_SIGNAL_SET=9 or NVSHMEM_SIGNAL_ADD=10)
// pe: Destination PE number
// After copying data, atomically updates the remote signal at sig_addr
extern __device__ void nvshmemx_putmem_signal_block(
    void* dest,
    const void* source,
    size_t nelems,
    uint64_t* sig_addr,
    uint64_t signal,
    int sig_op,
    int pe);

// Non-blocking immediate memory put with signal (generic, byte-oriented)
// dest: Symmetric address on the destination PE
// source: Local address of data to send
// nelems: Number of bytes to transfer
// sig_addr: Symmetric address of the signal variable on the destination PE
// signal: Value to be used in the signal operation
// sig_op: Signal operation type (NVSHMEM_SIGNAL_SET=9 or NVSHMEM_SIGNAL_ADD=10)
// pe: Destination PE number
// After copying data, atomically updates the remote signal at sig_addr.
// Returns immediately without waiting for completion. Use nvshmem_quiet() to
// ensure delivery.
extern __device__ void nvshmemx_putmem_signal_nbi(
    void* dest,
    const void* source,
    size_t nelems,
    uint64_t* sig_addr,
    uint64_t signal,
    int sig_op,
    int pe);

// Quiet - ensures all prior NVSHMEM operations are complete
extern __device__ void nvshmem_quiet();

// Fence - ensures ordering of memory operations
extern __device__ void nvshmem_fence();

// Team-scoped barrier - block-level synchronization within a team
// team: nvshmem_team_t (int32) - the team to synchronize with
// Use NVSHMEM_TEAM_SHARED (1) for LSA domain (same-node PEs)
// Use NVSHMEM_TEAM_WORLD (0) for all PEs
// Returns 0 on success
extern __device__ int nvshmemx_barrier_block(int32_t team);

// Multicast pointer - returns multicast address for broadcasting to all PEs
// Returns nullptr if multicast is not supported
// team: nvshmem_team_t (int32) - the team to use for multicast
// ptr: pointer to symmetric memory
extern __device__ void* nvshmemx_mc_ptr(int32_t team, const void* ptr);

// =============================================================================
// NVSHMEM ATOMIC OPERATIONS FOR SIGNALING
// These are used for point-to-point signaling without data transfer
// =============================================================================

// nvshmemx_signal_op - Perform an atomic signal operation on a remote PE
// sig_addr: Symmetric address of the signal variable (uint64_t*)
// signal: Value to be used in the signal operation
// sig_op: Signal operation type (NVSHMEM_SIGNAL_SET=9 or NVSHMEM_SIGNAL_ADD=10)
//         Note: These values are defined in nvshmem_common_transport.h
// pe: PE number of the remote PE
extern __device__ void nvshmemx_signal_op(
    uint64_t* sig_addr,
    uint64_t signal,
    int sig_op,
    int pe);

// Atomic set - sets a uint64 value at dest on pe
extern __device__ void nvshmem_uint64_atomic_set(
    uint64_t* dest,
    uint64_t value,
    int pe);

// Atomic add - adds value to dest on pe and returns old value
extern __device__ uint64_t
nvshmem_uint64_atomic_fetch_add(uint64_t* dest, uint64_t value, int pe);

// Atomic add (non-fetch version) - adds value to dest on pe
extern __device__ void nvshmem_uint64_atomic_add(
    uint64_t* dest,
    uint64_t value,
    int pe);

// =============================================================================
// NVSHMEM SIGNAL WAIT OPERATIONS
// =============================================================================

// Signal comparison condition constants for symm_signal_wait_until
// These are abstracted constants that map to NVSHMEM's comparison types.
// We use SIGNAL_CMP_* naming to avoid conflicts with NVSHMEM's NVSHMEM_CMP_*
// when both headers are included.
//
// Mapping to NVSHMEM's nvshmemi_cmp_type enum:
// SIGNAL_CMP_EQ (1) -> NVSHMEM_CMP_EQ (0)
// SIGNAL_CMP_NE (2) -> NVSHMEM_CMP_NE (1)
// SIGNAL_CMP_GT (3) -> NVSHMEM_CMP_GT (2)
// SIGNAL_CMP_GE (4) -> NVSHMEM_CMP_GE (3)
// SIGNAL_CMP_LT (5) -> NVSHMEM_CMP_LT (4)
// SIGNAL_CMP_LE (6) -> NVSHMEM_CMP_LE (5)
//
// Note: We use 1-based values to avoid conflicts and make 0 invalid.
#ifndef SIGNAL_CMP_EQ
#define SIGNAL_CMP_EQ 1
#define SIGNAL_CMP_NE 2
#define SIGNAL_CMP_GT 3
#define SIGNAL_CMP_GE 4
#define SIGNAL_CMP_LT 5
#define SIGNAL_CMP_LE 6
#endif

// nvshmem_signal_wait_until - Wait until a signal meets a condition
// sig_addr: Address of the signal variable to wait on (local)
// cmp: Comparison operation (NVSHMEM_CMP_EQ, NVSHMEM_CMP_GE, etc.)
//      Note: This takes NVSHMEM's native comparison constants, not SIGNAL_CMP_*
// cmp_value: Value to compare against
// Returns the signal value that satisfied the condition
extern __device__ uint64_t
nvshmem_signal_wait_until(uint64_t* sig_addr, int cmp, uint64_t cmp_value);

#ifdef __cplusplus
}
#endif

// =============================================================================
// NCCL DEVICE FUNCTION DECLARATIONS
// Note: NCCL does NOT provide a device bitcode library (unlike NVSHMEM)
// These declarations are here for completeness but will result in unresolved
// symbols when compiling to bitcode for Triton.
//
// NCCL LSA Barrier API:
// The proper NCCL LSA barrier uses ncclLsaBarrierSession<Coop>, which is a
// templated RAII class defined in nccl_device.h. It cannot be forward-declared
// here. When NCCL_SYMM_TYPES_AVAILABLE is defined (NCCL >= 2.28), the full
// nccl_device.h header is included which provides the barrier API.
// =============================================================================

extern "C" __device__ void* ncclGetLsaPointer(
    void* window,
    size_t offset,
    int peer);
extern "C" __device__ void* ncclGetLsaMultimemPointer(
    void* window,
    size_t offset,
    void* devComm);
// Note: ncclLsaBarrierSession is the correct barrier API (not ncclLsaBarrier)
// It is a RAII class from nccl_device.h with arrive()/wait()/sync() methods.
// See nccl_symm.cuh for usage via nccl_lsa_barrier_sync() helper.

// Mark structures as extern "C" to avoid C++ name mangling for use with Triton
extern "C" {

// =============================================================================
// SYMMETRIC CONTEXT BASE CLASS
// =============================================================================

/**
 * Abstract base class for symmetric communication context.
 * This allows kernels to work with different backends (NCCL, NVSHMEM, etc.)
 * without knowing the specific implementation details.
 */
struct SymmContext {
  // Type identifier for runtime type checking
  enum class Type : int32_t {
    NCCL = 0,
    NVSHMEM = 1,
  };

  Type type;
  int32_t rank;
  int32_t world_size;

  __host__ __device__ SymmContext(Type t, int32_t r, int32_t ws)
      : type(t), rank(r), world_size(ws) {}
};

// =============================================================================
// NCCL-SPECIFIC SYMMETRIC CONTEXT
// =============================================================================

// Maximum number of windows that can be registered in NCCLSymmContext
#ifndef NCCL_SYMM_MAX_WINDOWS
#define NCCL_SYMM_MAX_WINDOWS 8
#endif

/**
 * Entry in the NCCL window registry.
 * Maps a local memory region to its corresponding ncclWindow_t.
 */
struct NCCLWindowEntry {
  void* base_ptr; // Base address of the registered memory region
  size_t size; // Size of the region in bytes
  ncclWindow_t window; // The NCCL window handle for this region

  __host__ __device__ NCCLWindowEntry()
      : base_ptr(nullptr), size(0), window(nullptr) {}

  __host__ __device__ NCCLWindowEntry(void* ptr, size_t sz, ncclWindow_t win)
      : base_ptr(ptr), size(sz), window(win) {}

  /**
   * Check if a pointer falls within this window's memory region.
   */
  __host__ __device__ bool contains(const void* ptr) const {
    if (base_ptr == nullptr || size == 0) {
      return false;
    }
    const char* p = reinterpret_cast<const char*>(ptr);
    const char* base = reinterpret_cast<const char*>(base_ptr);
    return p >= base && p < (base + size);
  }

  /**
   * Calculate the byte offset of a pointer within this window.
   * Caller must ensure ptr is within this window (use contains() first).
   */
  __host__ __device__ size_t offset_of(const void* ptr) const {
    return reinterpret_cast<const char*>(ptr) -
        reinterpret_cast<const char*>(base_ptr);
  }
};

/**
 * Result of resolving a pointer to a window.
 */
struct NCCLWindowResolution {
  ncclWindow_t window; // The window containing the pointer
  size_t offset; // Byte offset within the window
  bool valid; // Whether the resolution was successful

  __host__ __device__ NCCLWindowResolution()
      : window(nullptr), offset(0), valid(false) {}

  __host__ __device__ NCCLWindowResolution(ncclWindow_t win, size_t off)
      : window(win), offset(off), valid(true) {}
};

/**
 * NCCL-specific symmetric communication context.
 *
 * This structure holds all NCCL-related data needed for LSA (Local Symmetric
 * Access) operations including:
 * - A registry of ncclWindow_t handles for registered memory regions
 * - Pointer to the device communicator
 *
 * The window registry allows resolving any local pointer to its corresponding
 * ncclWindow_t and offset, which is required for GIN put/get operations.
 *
 * LIMITATION: NCCL does not provide a device bitcode library, so this context
 * cannot be used with Triton extern_libs linking. It's included here for
 * completeness and potential future support.
 */
struct NCCLSymmContext : public SymmContext {
  // Window registry - maps local memory regions to ncclWindow_t handles
  NCCLWindowEntry windows[NCCL_SYMM_MAX_WINDOWS];
  int32_t num_windows;

  // Pointer to the NCCL device communicator (created via ncclDevCommCreate)
  ncclDevComm* dev_comm;

  // Legacy fields for backward compatibility
  // (buffer_window and local_buffer are now in windows[0])
  ncclWindow_t buffer_window; // Primary data buffer window
  ncclWindow_t signal_window; // Signal pad window
  void* local_buffer; // Base address of primary symmetric buffer
  size_t buffer_size; // Size of primary buffer in bytes

  // Signal pad pointers for each rank (device array of pointers)
  // signal_pad_ptrs[peer] gives the signal pad for that peer
  // Used for point-to-point signaling operations
  uint64_t** signal_pad_ptrs;

  // Device index where this context is valid
  int32_t device_idx;

  __host__ __device__ NCCLSymmContext()
      : SymmContext(Type::NCCL, 0, 0),
        num_windows(0),
        dev_comm(nullptr),
        buffer_window(nullptr),
        signal_window(nullptr),
        local_buffer(nullptr),
        buffer_size(0),
        signal_pad_ptrs(nullptr),
        device_idx(-1) {
    for (int i = 0; i < NCCL_SYMM_MAX_WINDOWS; i++) {
      windows[i] = NCCLWindowEntry();
    }
  }

  __host__ __device__ NCCLSymmContext(
      int32_t rank,
      int32_t world_size,
      ncclWindow_t buf_win,
      ncclWindow_t sig_win,
      ncclDevComm* dcomm,
      void* buf,
      size_t size,
      uint64_t** sig_pads,
      int32_t dev_idx)
      : SymmContext(Type::NCCL, rank, world_size),
        num_windows(0),
        dev_comm(dcomm),
        buffer_window(buf_win),
        signal_window(sig_win),
        local_buffer(buf),
        buffer_size(size),
        signal_pad_ptrs(sig_pads),
        device_idx(dev_idx) {
    for (int i = 0; i < NCCL_SYMM_MAX_WINDOWS; i++) {
      windows[i] = NCCLWindowEntry();
    }
    // Register the primary buffer window
    if (buf != nullptr && size > 0 && buf_win != nullptr) {
      register_window(buf, size, buf_win);
    }
  }

  /**
   * Register a memory region with its corresponding ncclWindow_t.
   * Returns true if successful, false if registry is full.
   */
  __host__ __device__ bool register_window(
      void* base_ptr,
      size_t size,
      ncclWindow_t window) {
    if (num_windows >= NCCL_SYMM_MAX_WINDOWS) {
      return false;
    }
    windows[num_windows] = NCCLWindowEntry(base_ptr, size, window);
    num_windows++;
    return true;
  }

  /**
   * Resolve a local pointer to its corresponding ncclWindow_t and offset.
   * Searches through all registered windows to find the one containing the
   * pointer.
   *
   * @param ptr Local pointer to resolve
   * @return NCCLWindowResolution with window, offset, and validity flag
   */
  __host__ __device__ NCCLWindowResolution
  resolve_window(const void* ptr) const {
    for (int i = 0; i < num_windows; i++) {
      if (windows[i].contains(ptr)) {
        return NCCLWindowResolution(
            windows[i].window, windows[i].offset_of(ptr));
      }
    }
    // Pointer not found in any registered window
    return NCCLWindowResolution();
  }
};

// =============================================================================
// NVSHMEM-SPECIFIC SYMMETRIC CONTEXT
// =============================================================================

/**
 * NVSHMEM-specific symmetric communication context.
 *
 * This structure holds all NVSHMEM-related data needed for symmetric memory
 * operations. Unlike NCCL, NVSHMEM uses a simpler programming model where
 * symmetric memory is accessed directly via nvshmem_ptr().
 *
 * IMPORTANT: NVSHMEM uses GLOBAL PE numbers (from nvshmem_my_pe()), not
 * group-local ranks. The kernel should use nvshmem_my_pe() and nvshmem_n_pes()
 * for PE numbering, or use the global_rank/global_world_size stored here.
 *
 * Key differences from NCCL:
 * - No window registration needed (NVSHMEM handles this internally)
 * - nvshmem_ptr() returns direct pointers to peer memory
 * - Barrier operations use nvshmemx_barrier_all_block()
 * - PE numbers are GLOBAL, not group-local
 *
 * Signal Pads:
 * NVSHMEM maintains two separate signal pads for different communication
 * domains:
 * - lsa_signal_pad: For LSA (Local Symmetric Access) domain operations
 *   Used for local/direct memory access signaling via nvshmem_ptr()
 * - gin_signal_pad: For GIN (GPU-Initiated Networking) domain operations
 *   Used for remote signaling via nvshmemx_signal_op()
 *
 * The symm_signal primitive uses gin_signal_pad (for remote atomic signals),
 * while symm_lsa_signal_ptr returns pointers to lsa_signal_pad (for P2P
 * access).
 *
 * This context is created on the host and passed to device kernels.
 * NVSHMEM provides libnvshmem_device.bc which can be linked with Triton
 * kernels, making this implementation fully functional.
 */
struct NVSHMEMSymmContext : public SymmContext {
  // Local buffer pointer (the base address of this PE's symmetric memory)
  // This is the symmetric address that can be used with nvshmem_ptr()
  void* local_buffer;

  // Size of the buffer in bytes
  size_t buffer_size;

  // LSA signal pad pointer (symmetric address for P2P load/store signaling)
  // This is accessible via nvshmem_ptr() for direct memory access
  // Used by symm_lsa_signal_ptr to get peer's signal pad address
  uint64_t* lsa_signal_pad;

  // GIN signal pad pointer (symmetric address for remote atomic signaling)
  // This is used with nvshmemx_signal_op() for GPU-initiated networking
  // Used by symm_signal for atomic signal operations
  uint64_t* gin_signal_pad;

  // Device index where this context is valid
  int32_t device_idx;

  // Offset within the allocation (for sub-tensors)
  size_t offset;

  // Global PE number (from nvshmem_my_pe(), not group-local rank)
  // Use this with nvshmem_ptr() for remote access
  int32_t global_rank;

  // Total number of PEs in the NVSHMEM job (from nvshmem_n_pes())
  int32_t global_world_size;

  __host__ __device__ NVSHMEMSymmContext()
      : SymmContext(Type::NVSHMEM, 0, 0),
        local_buffer(nullptr),
        buffer_size(0),
        lsa_signal_pad(nullptr),
        gin_signal_pad(nullptr),
        device_idx(-1),
        offset(0),
        global_rank(-1),
        global_world_size(0) {}

  __host__ __device__ NVSHMEMSymmContext(
      int32_t rank,
      int32_t world_size,
      void* local_buf,
      size_t size,
      uint64_t* lsa_sig_pad,
      uint64_t* gin_sig_pad,
      int32_t dev_idx,
      size_t off,
      int32_t g_rank,
      int32_t g_world_size)
      : SymmContext(Type::NVSHMEM, rank, world_size),
        local_buffer(local_buf),
        buffer_size(size),
        lsa_signal_pad(lsa_sig_pad),
        gin_signal_pad(gin_sig_pad),
        device_idx(dev_idx),
        offset(off),
        global_rank(g_rank),
        global_world_size(g_world_size) {}
};

} // extern "C"

// =============================================================================
// SYMMETRIC TEAM BASE CLASS (TOPOLOGY MANAGER)
// =============================================================================

extern "C" {

/**
 * SymmTeam represents a group of communicating peers (Topology Manager).
 *
 * It decouples synchronization scope from memory resources and provides
 * information about:
 * - Team membership (size, rank within team)
 * - LSA (Local Symmetric Access) domain - peers with direct load/store access
 *
 * LSA Domain: A subset of peers that can directly access each other's memory
 * via load/store operations (e.g., NVLink-connected GPUs on the same node).
 * Peers outside the LSA domain require explicit communication primitives.
 *
 * Common team types:
 * - WORLD: All ranks in the job
 * - NODE_LOCAL: Ranks on the same physical node
 * - Custom: User-defined subgroups
 *
 * This is the base class. Use NCCLSymmTeam or NVSHMEMSymmTeam for
 * backend-specific implementations with embedded team handles.
 */
struct SymmTeam {
  // Type identifier for runtime type checking (matches SymmContext::Type)
  enum class Type : int32_t {
    NCCL = 0,
    NVSHMEM = 1,
  };

  Type type;

  // Team membership information
  int32_t team_size; // Total number of ranks in this team
  int32_t team_rank; // This process's rank within the team (0..team_size-1)

  // LSA (Local Symmetric Access) domain information
  // Peers in the same LSA domain can directly access each other's memory
  int32_t lsa_size; // Number of ranks in caller's LSA domain
  int32_t lsa_base_rank; // First rank in the LSA domain (for range-based check)

  // Bitmask for LSA membership (for small teams, up to 64 ranks)
  // If team_size > 64, use lsa_base_rank + lsa_size for range check
  uint64_t lsa_mask;

  __host__ __device__ SymmTeam()
      : type(Type::NVSHMEM),
        team_size(0),
        team_rank(-1),
        lsa_size(0),
        lsa_base_rank(0),
        lsa_mask(0) {}

  __host__ __device__ SymmTeam(
      Type t,
      int32_t size,
      int32_t rank,
      int32_t lsa_sz,
      int32_t lsa_base,
      uint64_t mask)
      : type(t),
        team_size(size),
        team_rank(rank),
        lsa_size(lsa_sz),
        lsa_base_rank(lsa_base),
        lsa_mask(mask) {}

  /**
   * Check if a peer rank is in the same LSA domain as the caller.
   * Returns true if peer can be accessed via direct load/store.
   */
  __host__ __device__ bool is_lsa_peer(int32_t peer) const {
    if (peer < 0 || peer >= team_size) {
      return false;
    }
    // Use bitmask for small teams
    if (team_size <= 64) {
      return (lsa_mask & (1ULL << peer)) != 0;
    }
    // Use range check for larger teams
    return peer >= lsa_base_rank && peer < (lsa_base_rank + lsa_size);
  }
};

// =============================================================================
// NCCL-SPECIFIC SYMMETRIC TEAM
// =============================================================================

/**
 * NCCL-specific symmetric team.
 *
 * Extends SymmTeam with NCCL-specific team handle. In NCCL, teams are
 * represented by the ncclTeam type which is obtained from the device
 * communicator. For LSA operations, NCCL uses ncclTeamLsa to get the
 * local symmetric access team.
 *
 * LIMITATION: NCCL does not provide a device bitcode library, so this team
 * cannot be used with Triton extern_libs linking.
 */
struct NCCLSymmTeam : public SymmTeam {
  // NCCL device communicator reference (needed to get ncclTeamLsa)
  ncclDevComm* dev_comm;

  // Barrier handle for team-scoped barriers
  int32_t barrier_id;

  __host__ __device__ NCCLSymmTeam()
      : SymmTeam(Type::NCCL, 0, -1, 0, 0, 0),
        dev_comm(nullptr),
        barrier_id(0) {}

  __host__ __device__ NCCLSymmTeam(
      int32_t size,
      int32_t rank,
      int32_t lsa_sz,
      int32_t lsa_base,
      uint64_t mask,
      ncclDevComm* dcomm,
      int32_t barrier)
      : SymmTeam(Type::NCCL, size, rank, lsa_sz, lsa_base, mask),
        dev_comm(dcomm),
        barrier_id(barrier) {}
};

// =============================================================================
// NVSHMEM-SPECIFIC SYMMETRIC TEAM
// =============================================================================

// NVSHMEM team type (nvshmem_team_t is int32_t)
typedef int32_t nvshmem_team_t;

// Well-known NVSHMEM team values (matching NVSHMEM's nvshmem_team_id_t enum)
// These are provided as constexpr values instead of macros to avoid
// conflicts with NVSHMEM header definitions.
// Use these values directly when NVSHMEM headers are not included.
static constexpr nvshmem_team_t SYMM_TEAM_WORLD = 0;
static constexpr nvshmem_team_t SYMM_TEAM_SHARED = 1;
static constexpr nvshmem_team_t SYMM_TEAM_NODE = 2;

/**
 * NVSHMEM-specific symmetric team.
 *
 * Extends SymmTeam with the NVSHMEM team handle (nvshmem_team_t).
 * The team handle is used for team-scoped operations like:
 * - nvshmemx_mc_ptr(team, ptr) - Get multicast pointer for team
 * - nvshmem_team_sync(team) - Team-scoped barrier
 * - nvshmem_*_reduce(team, ...) - Team-scoped reductions
 *
 * Common team handles:
 * - NVSHMEM_TEAM_WORLD (0): All PEs in the job
 * - NVSHMEM_TEAM_SHARED (1): PEs that share memory (same node)
 * - NVSHMEM_TEAM_NODE (2): PEs on the same node
 *
 * The team handle can be obtained via:
 * - nvshmem_team_split_strided() - Create custom team
 * - nvshmem_team_split_2d() - Create 2D team
 * - Predefined teams (NVSHMEM_TEAM_WORLD, etc.)
 */
struct NVSHMEMSymmTeam : public SymmTeam {
  // NVSHMEM team handle (nvshmem_team_t = int32_t)
  // This is passed to nvshmemx_mc_ptr() and other team-scoped operations
  nvshmem_team_t nvshmem_team;

  __host__ __device__ NVSHMEMSymmTeam()
      : SymmTeam(Type::NVSHMEM, 0, -1, 0, 0, 0),
        nvshmem_team(SYMM_TEAM_WORLD) {}

  __host__ __device__ NVSHMEMSymmTeam(
      int32_t size,
      int32_t rank,
      int32_t lsa_sz,
      int32_t lsa_base,
      uint64_t mask,
      nvshmem_team_t team)
      : SymmTeam(Type::NVSHMEM, size, rank, lsa_sz, lsa_base, mask),
        nvshmem_team(team) {}

  /**
   * Get the NVSHMEM team handle for use with NVSHMEM APIs.
   */
  __host__ __device__ nvshmem_team_t get_team_handle() const {
    return nvshmem_team;
  }
};

} // extern "C"
