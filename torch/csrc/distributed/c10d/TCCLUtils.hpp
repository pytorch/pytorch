#pragma once

#ifdef USE_C10D_TCCL

// TCCL transport-layer types and free-function helpers used by ProcessGroupTCCL.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

struct ibv_context;
struct ibv_pd;
struct ibv_cq;
struct ibv_qp;
struct ibv_mr;
struct ibv_device;
struct ibv_qp_init_attr;
struct ibv_qp_attr;
struct ibv_port_attr;
struct ibv_comp_channel;
struct ibv_sge;
struct ibv_wc;
union ibv_gid;

namespace c10d {

class Store;

class TCCLRegisteredMPSTensor;

// POD routing info exchanged between peers during QP setup. Serialized by
// reinterpret_cast to bytes.
//
// 32 bytes on macOS arm64: 3 * int (12 bytes, 4-byte aligned) + ibv_gid
// (16 bytes union). The static_assert in TCCLUtils.cpp catches any platform
// where this assumption breaks before we send corrupt bytes over the wire.
struct TCCLDestination {
  int32_t lid;
  int32_t qp_num;
  int32_t psn;
  int32_t _pad;       // align gid to 16 bytes; ibv_gid is a 16-byte union
  uint8_t gid[16];    // raw GID bytes

};

// =============================================================================
// IBV wrapper - singleton dlopen of librdma.dylib
// =============================================================================

// Process-wide singleton owning the dlopen handle and dlsym'd function
// pointers for all librdma.dylib control-path symbols. Data-path symbols
// (ibv_post_send, ibv_post_recv, ibv_poll_cq) are NOT dlsym'd - they are
// inline in <infiniband/verbs.h> and call through ctx->ops.

class TORCH_API TCCLIBVWrapper {
 public:
  static TCCLIBVWrapper& instance();

  // Disable copy and move - singleton.
  TCCLIBVWrapper(const TCCLIBVWrapper&) = delete;
  TCCLIBVWrapper& operator=(const TCCLIBVWrapper&) = delete;
  TCCLIBVWrapper(TCCLIBVWrapper&&) = delete;
  TCCLIBVWrapper& operator=(TCCLIBVWrapper&&) = delete;

  // Control-path verbs entry points, resolved via dlsym in the constructor.
  // Signatures mirror <infiniband/verbs.h> using forward-declared struct
  // types so this header does not depend on librdma headers.
  ibv_device** (*get_device_list)(int*) = nullptr;
  const char*  (*get_device_name)(ibv_device*) = nullptr;
  ibv_context* (*open_device)(ibv_device*) = nullptr;
  void         (*free_device_list)(ibv_device**) = nullptr;
  int          (*close_device)(ibv_context*) = nullptr;
  ibv_pd*      (*alloc_pd)(ibv_context*) = nullptr;
  ibv_cq*      (*create_cq)(
      ibv_context*,
      int cqe,
      void* cq_context,
      ibv_comp_channel*,
      int comp_vector) = nullptr;
  ibv_qp*      (*create_qp)(ibv_pd*, ibv_qp_init_attr*) = nullptr;
  int          (*destroy_qp)(ibv_qp*) = nullptr;
  int          (*destroy_cq)(ibv_cq*) = nullptr;
  int          (*dealloc_pd)(ibv_pd*) = nullptr;
  int          (*query_port)(ibv_context*, uint8_t, ibv_port_attr*) = nullptr;
  int          (*query_gid)(ibv_context*, uint8_t, int, ibv_gid*) = nullptr;
  int          (*modify_qp)(ibv_qp*, ibv_qp_attr*, int) = nullptr;
  ibv_mr*      (*reg_mr)(ibv_pd*, void*, size_t, int) = nullptr;
  int          (*dereg_mr)(ibv_mr*) = nullptr;

 private:
  // Constructor performs dlopen + 16 dlsym in one shot.
  TCCLIBVWrapper();
  ~TCCLIBVWrapper() = default;

  void* handle_ = nullptr;
};

// =============================================================================
// Connection - owns one QP plus its supporting context/PD/CQ
// =============================================================================

// One Connection corresponds to one UC queue pair to one peer-wire. Each
// instance opens its own ibv_context, allocates its own PD/CQ, and creates
// one QP.

// Lifecycle:
//   ctor              opens device, allocates PD/CQ, creates QP, transitions
//                     QP to INIT, populates local_destination_.
//   transitionToRTR   feeds the peer's TCCLDestination to ibv_modify_qp(RTR).
//   transitionToRTS   transitions to RTS using local PSN.
//   dtor              tears down QP -> CQ -> PD -> ctx in that order.

class TORCH_API TCCLConnection {
 public:
  // Opens device_name (e.g. "rdma_en2") and runs through INIT-state
  // initialization. After construction, localDestination() returns valid
  // QP routing info ready to publish via Store.
  explicit TCCLConnection(const std::string& device_name);

  ~TCCLConnection();

  TCCLConnection(const TCCLConnection&) = delete;
  TCCLConnection& operator=(const TCCLConnection&) = delete;

  TCCLConnection(TCCLConnection&& other) noexcept;
  TCCLConnection& operator=(TCCLConnection&& other) noexcept;

  // Transition our QP to RTR using the peer's published routing info.
  // Throws DistBackendError on ibv_modify_qp failure.
  void transitionToRTR(const TCCLDestination& remote);

  // Transition our QP to RTS using our locally chosen PSN.
  void transitionToRTS();

  // ===== Data-path methods =====
  // Post a send WR referencing `length` bytes of `buf`'s storage. Caller is
  // responsible for buf having been registered to this Connection's PD via
  // TCCLSharedBuffer::registerToPD(protectionDomain()) at init time.
  // wr_id is opaque to verbs - the algorithm layer encodes (type, slot, peer)
  // into it for completion-event demultiplexing (see TCCLEngine).
  // Throws DistBackendError if ibv_post_send returns non-zero.
  void postSend(
      const class TCCLSharedBuffer& buf,
      uint64_t length,
      uint64_t wr_id);

  // Post a recv WR referencing `length` bytes of `buf`'s storage. Same
  // registration precondition as postSend. UC requires the recv to be
  // posted BEFORE the matching send arrives (Apple TN3205 §12.3: credit-based
  // flow control stalls the sender otherwise).
  void postRecv(
      class TCCLSharedBuffer& buf,
      uint64_t length,
      uint64_t wr_id);

  // Non-blocking. Polls up to `max_completions` work completion events
  // from this Connection's CQ into `wcs`. Returns the number actually
  // dequeued (0 if CQ empty, never negative - verbs errors throw).
  // Caller is the algorithm layer which busy-polls until all expected
  // completions arrive.
  int pollCq(int max_completions, ibv_wc* wcs);

  // Local routing info this rank publishes to peers via Store.
  const TCCLDestination& localDestination() const noexcept {
    return local_destination_;
  }

  // Accessors for the algorithm/buffer layer. Returned pointers are owned by
  // this Connection - do not free.
  ibv_context* context() const noexcept { return ctx_; }
  ibv_pd* protectionDomain() const noexcept { return pd_; }
  ibv_cq* completionQueue() const noexcept { return cq_; }
  ibv_qp* queuePair() const noexcept { return qp_; }

 private:
  void swap(TCCLConnection& other) noexcept;
  void destroy() noexcept;

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_qp* qp_ = nullptr;
  TCCLDestination local_destination_{};
};

// =============================================================================
// SharedBuffer - page-aligned RDMA-registerable storage
// =============================================================================

// A fixed-size, page-aligned chunk of host memory that can be registered to
// one or more protection domains via ibv_reg_mr.
//
// One of these is used per (peer, direction): every rank holds (size_-1) send
// buffers and (size_-1) recv buffers, each registered to the corresponding
// peer's PD.
//
// Move-only because the underlying allocation and the MR map cannot be
// safely copied. Default-constructible as an empty buffer (data_=nullptr,
// size_=0) so vectors of SharedBuffers can be sized first and assigned later.
class TORCH_API TCCLSharedBuffer {
 public:
  // Empty buffer. Useful as a placeholder for the self-slot in
  // peer-indexed vectors.
  TCCLSharedBuffer() noexcept = default;

  // Allocate `num_bytes` of page-aligned storage via posix_memalign and
  // zero-initialize. Throws DistBackendError on allocation failure.
  explicit TCCLSharedBuffer(size_t num_bytes);

  ~TCCLSharedBuffer();

  TCCLSharedBuffer(const TCCLSharedBuffer&) = delete;
  TCCLSharedBuffer& operator=(const TCCLSharedBuffer&) = delete;

  TCCLSharedBuffer(TCCLSharedBuffer&& other) noexcept;
  TCCLSharedBuffer& operator=(TCCLSharedBuffer&& other) noexcept;

  // Register the underlying storage with a protection domain. The MR is
  // owned by this object and freed in the destructor. Throws
  // DistBackendError if `pd` is already registered - buffers register to each
  // peer's PD exactly once at init.
  void registerToPD(ibv_pd* pd);

  // Build an ibv_sge describing `length` bytes at offset 0 of this buffer's
  // storage, using the lkey from the MR registered to `pd`. `pd` must have
  // been passed to registerToPD; otherwise throws DistBackendError.
  // `length` must be <= size().
  ibv_sge toSge(ibv_pd* pd, uint64_t length) const;

  // Raw access for memcpy in/out of the buffer. Pointer is page-aligned.
  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }

  size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return data_ == nullptr; }

 private:
  void destroy() noexcept;
  void swap(TCCLSharedBuffer& other) noexcept;

  void* data_ = nullptr;
  size_t size_ = 0;
  std::unordered_map<ibv_pd*, ibv_mr*> mrs_;
};

// =============================================================================
// Free helpers
// =============================================================================

// Lists the names of RDMA devices visible to librdma on this host.
// Loads the singleton TCCLIBVWrapper if not already loaded. Returns an empty
// vector if the library loads but the device list is empty (e.g.
// Thunderbolt is bridged - user must run `sudo tbtrdmactl unbridge`).
// Throws DistBackendError if librdma cannot be loaded.
TORCH_API std::vector<std::string> listRdmaDevices();

// Resolve which RDMA device the backend should bind to.
//
// Precedence:
//   1. `explicit_name` if non-empty (passed via Options::device_name).
//   2. `TCCL_DEVICE` environment variable.
//   3. Auto-detect via listRdmaDevices() - succeeds iff exactly one
//      device is visible (the common case on a single-Thunderbolt-port Mac).
//
// Throws DistBackendError with an actionable message when no device is
// found OR when multiple devices are found and no explicit selection was
// provided.
TORCH_API std::string resolveTcclDeviceName(const std::string& explicit_name);

// Resolve the per-peer RDMA device list for this rank. Returns a vector of
// size `size`; entry [peer] is the device this rank uses to reach `peer`, with
// the self-slot ([rank]) left empty. In a full Thunderbolt mesh each peer is on
// a different physical port (different rdma_enX), so a single device cannot
// drive the group.
//
// Precedence:
//   1. TCCL_PEER_DEVICES env - comma-separated list of exactly `size` device
//      names, self-slot empty (e.g. "rdma_en2,,rdma_en3,rdma_en4" for rank 1 of
//      4). This is the rank's row of the auto-discovered device matrix, set by
//      the launcher from the hostfile. Enables the full-mesh topology.
//   2. Fallback - a single device (resolveTcclDeviceName(explicit_name)) for
//      ALL peers (the single-Thunderbolt-link case).
//
// Throws DistBackendError if TCCL_PEER_DEVICES is set but malformed (wrong
// count, non-empty self-slot, or empty peer slot). Under ring_topology the row
// is sparse: only the two neighbor slots ((rank±1)%size) may be non-empty and
// all other slots (incl. self) must be empty - the inverse is enforced too.
TORCH_API std::vector<std::string> resolveTcclPeerDevices(
    int rank,
    int size,
    const std::string& explicit_name,
    bool ring_topology = false);

// Verify the underlying BSD interface for the requested RDMA device has a
// non-link-local static IPv4 address. Throws DistBackendError with the exact
// sudo commands to run if the prerequisite isn't met.
TORCH_API void checkLinkLayer(const std::string& rdma_device);

// Reserve a unique init sequence number that ALL ranks of this PG instance
// agree on. Used as a key prefix for destination exchange so PG re-init under
// the same group_name cannot accidentally read stale destinations from a
// prior incarnation.
//
// Only rank 0 reserves the number (Store::add - a server-side atomic on
// TCPStore, fresh on every (re)creation) and broadcasts it via the Store;
// every other rank reads rank 0's value. A per-rank
// Store::add would hand each rank a different post-increment value, so the
// ranks would build different key prefixes and never find each other. Mirrors
// NCCL's unique-id broadcast. Returns the agreed counter value (1-indexed).
TORCH_API int64_t tcclInitSequence(
    Store& store,
    int rank,
    std::chrono::milliseconds timeout);

// All-to-all destination exchange. Each rank publishes its
// num_wires-sized vector of TCCLDestinations under keyPrefix + std::to_string(rank),
// then blocks on Store::multiGet until every peer has done the same.
//
// On length mismatch (e.g. peer compiled with different num_wires), throws
// DistBackendError with the offending peer's rank and byte counts.
TORCH_API void allgatherDestinationsViaStore(
    Store& store,
    int rank,
    int size,
    const std::vector<TCCLDestination>& local,
    std::vector<std::vector<TCCLDestination>>& remote,
    const std::string& keyPrefix,
    std::chrono::milliseconds timeout);

// Global RTS barrier - every rank reaches this point after locally
// transitioning all of its QPs to RTS, before any rank is allowed to post
// a send. Otherwise UC drops on the receiver side if its QP isn't yet RTS.
//
// Wraps Store::barrier. Optimized server-side increment+wait - single round
// trip per rank.
TORCH_API void tcclRtsBarrier(
    Store& store,
    int size,
    const std::string& key,
    std::chrono::milliseconds timeout);

} // namespace c10d

#endif // USE_C10D_TCCL

