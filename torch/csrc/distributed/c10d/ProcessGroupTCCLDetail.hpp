#pragma once

#ifdef USE_C10D_TCCL

// TCCL collective-algorithm engine and reduction-op templates

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace c10d {

class TCCLConnection;
class TCCLSharedBuffer;

// ---- Reduction op templates ------------------------------------------------



template <typename T>
struct TCCLSumOp {
  void operator()(const T* input, T* output, std::size_t n) const;
};

template <typename T>
struct TCCLMaxOp {
  void operator()(const T* input, T* output, std::size_t n) const;
};

template <typename T>
struct TCCLMinOp {
  void operator()(const T* input, T* output, std::size_t n) const;
};

template <typename T>
struct TCCLProdOp {
  void operator()(const T* input, T* output, std::size_t n) const;
};

// Float32 SUM
template <>
inline void TCCLSumOp<float>::operator()(
    const float* input,
    float* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] += input[i];
  }
}

// float16 / bfloat16 SUM — accumulate in float32 then round back, matching
// PyTorch's reduction semantics for low-precision floats (the dtypes DDP/TP
// reduce most).
template <>
inline void TCCLSumOp<c10::Half>::operator()(
    const c10::Half* input,
    c10::Half* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = static_cast<c10::Half>(
        static_cast<float>(output[i]) + static_cast<float>(input[i]));
  }
}

template <>
inline void TCCLSumOp<c10::BFloat16>::operator()(
    const c10::BFloat16* input,
    c10::BFloat16* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = static_cast<c10::BFloat16>(
        static_cast<float>(output[i]) + static_cast<float>(input[i]));
  }
}

// Integer SUM
#define TCCL_DEFINE_INT_SUM(TYPE)                       \
  template <>                                           \
  inline void TCCLSumOp<TYPE>::operator()(              \
      const TYPE* input, TYPE* output, std::size_t n)   \
      const {                                           \
    for (std::size_t i = 0; i < n; ++i) {               \
      output[i] += input[i];                            \
    }                                                   \
  }
TCCL_DEFINE_INT_SUM(int8_t)
TCCL_DEFINE_INT_SUM(int16_t)
TCCL_DEFINE_INT_SUM(int32_t)
TCCL_DEFINE_INT_SUM(int64_t)
TCCL_DEFINE_INT_SUM(uint8_t)
#undef TCCL_DEFINE_INT_SUM

// bool SUM = logical OR
template <>
inline void TCCLSumOp<bool>::operator()(
    const bool* input,
    bool* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = output[i] || input[i];
  }
}

// PRODUCT  mirroring SUM: float32-accumulate for
// fp16/bf16, native in-dtype for the integer dtypes (wraparound), logical AND
// for bool.
template <>
inline void TCCLProdOp<float>::operator()(
    const float* input, float* output, std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] *= input[i];
  }
}
template <>
inline void TCCLProdOp<c10::Half>::operator()(
    const c10::Half* input, c10::Half* output, std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = static_cast<c10::Half>(
        static_cast<float>(output[i]) * static_cast<float>(input[i]));
  }
}
template <>
inline void TCCLProdOp<c10::BFloat16>::operator()(
    const c10::BFloat16* input, c10::BFloat16* output, std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = static_cast<c10::BFloat16>(
        static_cast<float>(output[i]) * static_cast<float>(input[i]));
  }
}
#define TCCL_DEFINE_INT_PROD(TYPE)                              \
  template <>                                                   \
  inline void TCCLProdOp<TYPE>::operator()(                     \
      const TYPE* input, TYPE* output, std::size_t n) const {   \
    for (std::size_t i = 0; i < n; ++i)                         \
      output[i] = static_cast<TYPE>(output[i] * input[i]);      \
  }
TCCL_DEFINE_INT_PROD(int8_t)
TCCL_DEFINE_INT_PROD(int16_t)
TCCL_DEFINE_INT_PROD(int32_t)
TCCL_DEFINE_INT_PROD(int64_t)
TCCL_DEFINE_INT_PROD(uint8_t)
#undef TCCL_DEFINE_INT_PROD
template <>
inline void TCCLProdOp<bool>::operator()(
    const bool* input, bool* output, std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = output[i] && input[i];
  }
}

// MIN/MAX
#define TCCL_DEFINE_MINMAX(TYPE)                                            \
  template <>                                                               \
  inline void TCCLMaxOp<TYPE>::operator()(                                  \
      const TYPE* input, TYPE* output, std::size_t n) const {              \
    for (std::size_t i = 0; i < n; ++i)                                     \
      output[i] = std::max(output[i], input[i]);                            \
  }                                                                         \
  template <>                                                               \
  inline void TCCLMinOp<TYPE>::operator()(                                  \
      const TYPE* input, TYPE* output, std::size_t n) const {              \
    for (std::size_t i = 0; i < n; ++i)                                     \
      output[i] = std::min(output[i], input[i]);                            \
  }
TCCL_DEFINE_MINMAX(float)
TCCL_DEFINE_MINMAX(int8_t)
TCCL_DEFINE_MINMAX(int16_t)
TCCL_DEFINE_MINMAX(int32_t)
TCCL_DEFINE_MINMAX(int64_t)
TCCL_DEFINE_MINMAX(uint8_t)
TCCL_DEFINE_MINMAX(bool)
#undef TCCL_DEFINE_MINMAX

#define TCCL_DEFINE_MINMAX_VIA_FLOAT(TYPE)                                  \
  template <>                                                               \
  inline void TCCLMaxOp<TYPE>::operator()(                                  \
      const TYPE* input, TYPE* output, std::size_t n) const {              \
    for (std::size_t i = 0; i < n; ++i)                                     \
      output[i] = static_cast<TYPE>(std::max(                               \
          static_cast<float>(output[i]), static_cast<float>(input[i])));    \
  }                                                                         \
  template <>                                                               \
  inline void TCCLMinOp<TYPE>::operator()(                                  \
      const TYPE* input, TYPE* output, std::size_t n) const {              \
    for (std::size_t i = 0; i < n; ++i)                                     \
      output[i] = static_cast<TYPE>(std::min(                               \
          static_cast<float>(output[i]), static_cast<float>(input[i])));    \
  }
TCCL_DEFINE_MINMAX_VIA_FLOAT(c10::Half)
TCCL_DEFINE_MINMAX_VIA_FLOAT(c10::BFloat16)
#undef TCCL_DEFINE_MINMAX_VIA_FLOAT


// ---- TCCLEngine ----------------------------------------------------------

class TCCLEngine {
 public:
  // Size of each per-peer staging buffer (hence its registered MR) and the
  // max bytes per ibv_post_send; larger messages split into a chunk loop and
  // reduce incrementally. Capped at 512 KB.
  static constexpr size_t kChunkSize = 512 * 1024;

  TCCLEngine(
      int rank,
      int size,
      std::vector<std::unique_ptr<TCCLConnection>>& connections,
      std::vector<TCCLSharedBuffer>& send_buffers,
      std::vector<TCCLSharedBuffer>& recv_buffers,
      const std::chrono::milliseconds& timeout,
      bool ring_topology = false)
      : rank_(rank),
        size_(size),
        connections_(connections),
        send_buffers_(send_buffers),
        recv_buffers_(recv_buffers),
        timeout_(timeout),
        ring_topology_(ring_topology) {}

  // True when this PG was brought up as a ring (sparse connections). Dispatch
  // reads it to force the ring path for every collective (see tcclUseRing).
  bool ringTopology() const {
    return ring_topology_;
  }

  template <typename T, typename ReduceOp>
  void all_reduce(T* data, std::size_t count, ReduceOp reduce_op);

  // In-place RING allreduce: reduce-scatter (size_-1 steps) + all-gather
  // (size_-1 steps), each step sending to the right neighbor and receiving from
  // the left.
  template <typename T, typename ReduceOp>
  void ring_all_reduce(T* data, std::size_t count, ReduceOp reduce_op);

  // RING reduce_scatter: in_chunks[p] points at this rank's contribution
  // destined for peer p (count_per_rank elements); on exit `out` holds the
  // element-wise reduction across ranks of chunk number rank_.
  template <typename T, typename ReduceOp>
  void ring_reduce_scatter(
      const std::vector<const T*>& in_chunks,
      T* out,
      std::size_t count_per_rank,
      ReduceOp reduce_op);

  // RING all_gather: `in` is this rank's shard (per_rank_bytes); on exit
  // out_ptrs[r] holds rank r's shard. N-1 ring steps (send right / recv left)
  // circulating shards; pure byte movement. out_ptrs.size() must equal size_.
  // Requires size_ > 2.
  void ring_all_gather(
      const void* in,
      const std::vector<void*>& out_ptrs,
      std::size_t per_rank_bytes);

  // In-place broadcast of `total_bytes` bytes at `data` from rank `root` to
  // every rank.
  void broadcast(void* data, std::size_t total_bytes, int root);

  // RING broadcast: pipelined store-and-forward of total_bytes from `root` along
  // the ring (root -> right -> ... -> root's left neighbour). Enables broadcast
  // on a ring-cabled cluster; on a full mesh the dispatch defaults to mesh broadcast.
  void ring_broadcast(void* data, std::size_t total_bytes, int root);

  // Mesh all-gather. `in` is this rank's contiguous shard of `per_rank_bytes`
  // bytes; on exit out_ptrs[r] receives rank r's shard (per_rank_bytes bytes).
  void all_gather(
      const void* in,
      const std::vector<void*>& out_ptrs,
      std::size_t per_rank_bytes);

  // Mesh reduce-scatter. in_chunks[p] points at this rank's contribution
  // destined for peer p (count_per_rank elements); on exit `out` holds the
  // element-wise reduction across ranks of chunk number rank_. Reduces straight
  // into `out` (no scratch).
  template <typename T, typename ReduceOp>
  void reduce_scatter(
      const std::vector<const T*>& in_chunks,
      T* out,
      std::size_t count_per_rank,
      ReduceOp reduce_op);

  // Point-to-point send: stream `nbytes` from `in` to peer `dst` over its UC QP,
  // chunked at kChunkSize through the per-peer send buffer (PIPELINE=1, one chunk
  // in flight).
  void p2p_send(int dst, const char* in, std::size_t nbytes);

  // Point-to-point recv: receive `nbytes` from peer `src` into `out`, chunked
  // through the per-peer recv buffer (PIPELINE=1). Posts the recv before polling.
  void p2p_recv(int src, char* out, std::size_t nbytes);

  // Mesh (direct) all-to-all. This rank's slab destined for peer p is
  // [send_base + send_off[p], +send_bytes[p]); the slab received from peer p lands
  // at [recv_base + recv_off[p], +recv_bytes[p]). Handles equal AND uneven
  // (alltoallv) splits — sizes are per-peer. The self slot (p == rank_) is a local
  // memcpy. Every peer is served simultaneously (recv+send in flight per peer,
  // chunked at kChunkSize, PIPELINE=1), polling each peer's CQ; recv posted before
  // send (UC discipline).
  void all_to_all(
      const char* send_base,
      char* recv_base,
      const std::vector<std::size_t>& send_off,
      const std::vector<std::size_t>& send_bytes,
      const std::vector<std::size_t>& recv_off,
      const std::vector<std::size_t>& recv_bytes);

  // RING all-to-all (neighbours only, store-and-forward), EQUAL splits: send_base
  // holds size_ segments of seg_bytes each (segment j destined for rank j); on exit
  // recv_base holds size_ segments (segment i received from rank i).
  void ring_all_to_all(
      const char* send_base, char* recv_base, std::size_t seg_bytes);

 private:
  // Core mesh primitive shared by all collectives above. Exchanges up to
  // `total_bytes` bytes with every non-self peer in kChunkSize chunks. Per
  // chunk [off, off+chunk_bytes):
  //   - for each peer with want_recv(peer) == true: pre-post a recv (UC needs
  //     the recv posted before the matching send arrives, Apple TN3205 §12.3);
  //   - for each peer where send_src(peer, off) != nullptr: memcpy that source
  //     region into the peer's send buffer and post a send;
  //   - busy-poll (PG-timeout watchdog) until every posted WR completes,
  //     calling on_recv(peer, recv_buf, off, chunk_bytes) once per recv
  //     completion.
  // The per-peer send_src lets reduce_scatter hand each peer a different slab;
  // want_recv + a nullable send_src let broadcast post an asymmetric
  // (root-sends / leaf-receives) WR set.
  template <typename SendSrc, typename WantRecv, typename OnRecv>
  void mesh_exchange(
      std::size_t total_bytes,
      SendSrc send_src,
      WantRecv want_recv,
      OnRecv on_recv);

  // One ring step shared by reduce-scatter and all-gather: send `send_bytes`
  // from `send_base` to the right neighbor `send_peer` AND receive `recv_bytes`
  // from the left neighbor `recv_peer`, both in kChunkSize pieces (PIPELINE=1,
  // one piece in flight per direction). recv posted before send (UC discipline);
  // per-completion wc.status check + a timeout_ watchdog. on_recv(recv_offset,
  // src, n_bytes) reduces/copies each received piece into the destination chunk.
  // The two directions carry independent byte counts (the ragged last chunk
  // differs per direction).
  template <typename OnRecv>
  void ring_step(
      int send_peer,
      const char* send_base,
      std::size_t send_bytes,
      int recv_peer,
      std::size_t recv_bytes,
      OnRecv on_recv);

  int rank_;
  int size_;
  std::vector<std::unique_ptr<TCCLConnection>>& connections_;
  std::vector<TCCLSharedBuffer>& send_buffers_;
  std::vector<TCCLSharedBuffer>& recv_buffers_;
  const std::chrono::milliseconds& timeout_;
  const bool ring_topology_{false};  // true => mesh_exchange must never be reached
};

} // namespace c10d

// Template implementations. Separated for readability; included at the end
// of this header so consumers get the full definitions without a separate
// include.
#include <torch/csrc/distributed/c10d/ProcessGroupTCCLDetail.ipp>

#endif // USE_C10D_TCCL
