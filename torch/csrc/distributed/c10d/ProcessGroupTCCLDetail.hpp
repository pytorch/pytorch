#pragma once

#ifdef USE_C10D_TCCL

// TCCL collective-algorithm engine and reduction-op templates

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <vector>

#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace c10d {

class TCCLConnection;
class TCCLSharedBuffer;

// Reduction op templates

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

// float16 / bfloat16 SUM - accumulate in float32 then round back, matching
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

// bfloat16 SUM. On FEAT_BF16 hardware (all Apple Silicon with bf16) this
// reduces with native __bf16 arithmetic (armv8.6-a); set TCCL_NATIVE_BF16=0 to
// force the float32-accumulate fallback. On Apple aarch64 there is no scalar
// bf16 FADD, so
// __bf16 addition lowers to fcvt -> fp32 add -> fcvt: a correctly-rounded
// single add, bitwise-identical to the fp32-accumulate path (verified), so this
// is a pure speedup with no change to reduction semantics. The native path
// avoids the explicit per-element convert loop, which is the bf16 all-reduce
// cost vs JACCL.

#if defined(__aarch64__) && defined(__APPLE__)

inline bool tcclNativeBf16Enabled() {
  static bool enabled = []() {
    int value = 0;
    std::size_t value_size = sizeof(value);
    bool hw =
        sysctlbyname(
            "hw.optional.arm.FEAT_BF16", &value, &value_size, nullptr, 0) ==
            0 &&
        value != 0;
    if (!hw) {
      return false;
    }
    const char* e = std::getenv("TCCL_NATIVE_BF16");
    return e == nullptr || std::atoi(e) != 0;
  }();
  return enabled;
}

__attribute__((target("arch=armv8.6-a"))) inline void tcclNativeBf16Sum(
    const c10::BFloat16* input,
    c10::BFloat16* output,
    std::size_t n) {
  auto in = reinterpret_cast<const __bf16*>(input);
  auto out = reinterpret_cast<__bf16*>(output);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = out[i] + in[i];
  }
}

#endif // defined(__aarch64__) && defined(__APPLE__)

template <>
inline void TCCLSumOp<c10::BFloat16>::operator()(
    const c10::BFloat16* input,
    c10::BFloat16* output,
    std::size_t n) const {
#if defined(__aarch64__) && defined(__APPLE__)
  if (tcclNativeBf16Enabled()) {
    tcclNativeBf16Sum(input, output, n);
    return;
  }
#endif
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = static_cast<c10::BFloat16>(
        static_cast<float>(output[i]) + static_cast<float>(input[i]));
  }
}

// Integer SUM
#define TCCL_DEFINE_INT_SUM(TYPE)                             \
  template <>                                                 \
  inline void TCCLSumOp<TYPE>::operator()(                    \
      const TYPE* input, TYPE* output, std::size_t n) const { \
    for (std::size_t i = 0; i < n; ++i) {                     \
      output[i] += input[i];                                  \
    }                                                         \
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
    const float* input,
    float* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] *= input[i];
  }
}
template <>
inline void TCCLProdOp<c10::Half>::operator()(
    const c10::Half* input,
    c10::Half* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = static_cast<c10::Half>(
        static_cast<float>(output[i]) * static_cast<float>(input[i]));
  }
}
template <>
inline void TCCLProdOp<c10::BFloat16>::operator()(
    const c10::BFloat16* input,
    c10::BFloat16* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = static_cast<c10::BFloat16>(
        static_cast<float>(output[i]) * static_cast<float>(input[i]));
  }
}
#define TCCL_DEFINE_INT_PROD(TYPE)                            \
  template <>                                                 \
  inline void TCCLProdOp<TYPE>::operator()(                   \
      const TYPE* input, TYPE* output, std::size_t n) const { \
    for (std::size_t i = 0; i < n; ++i)                       \
      output[i] = static_cast<TYPE>(output[i] * input[i]);    \
  }
TCCL_DEFINE_INT_PROD(int8_t)
TCCL_DEFINE_INT_PROD(int16_t)
TCCL_DEFINE_INT_PROD(int32_t)
TCCL_DEFINE_INT_PROD(int64_t)
TCCL_DEFINE_INT_PROD(uint8_t)
#undef TCCL_DEFINE_INT_PROD
template <>
inline void TCCLProdOp<bool>::operator()(
    const bool* input,
    bool* output,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = output[i] && input[i];
  }
}

// MIN/MAX
#define TCCL_DEFINE_MINMAX(TYPE)                              \
  template <>                                                 \
  inline void TCCLMaxOp<TYPE>::operator()(                    \
      const TYPE* input, TYPE* output, std::size_t n) const { \
    for (std::size_t i = 0; i < n; ++i)                       \
      output[i] = std::max(output[i], input[i]);              \
  }                                                           \
  template <>                                                 \
  inline void TCCLMinOp<TYPE>::operator()(                    \
      const TYPE* input, TYPE* output, std::size_t n) const { \
    for (std::size_t i = 0; i < n; ++i)                       \
      output[i] = std::min(output[i], input[i]);              \
  }
TCCL_DEFINE_MINMAX(int8_t)
TCCL_DEFINE_MINMAX(int16_t)
TCCL_DEFINE_MINMAX(int32_t)
TCCL_DEFINE_MINMAX(int64_t)
TCCL_DEFINE_MINMAX(uint8_t)
TCCL_DEFINE_MINMAX(bool)
#undef TCCL_DEFINE_MINMAX

// Floating MIN/MAX follow IEEE 754-2019 minimum/maximum: a NaN wins and
// -0 < +0. std::min/std::max keep or drop a NaN, and pick a signed zero, by
// argument order, and on a 2-rank mesh each rank reduces its own value first,
// so the ranks would end with different results.
inline float tcclMaxFloat(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (a == b) {
    return std::signbit(a) ? b : a;
  }
  return a > b ? a : b;
}
inline float tcclMinFloat(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (a == b) {
    return std::signbit(a) ? a : b;
  }
  return a < b ? a : b;
}

#define TCCL_DEFINE_MINMAX_FLOAT(TYPE)                                   \
  template <>                                                            \
  inline void TCCLMaxOp<TYPE>::operator()(                               \
      const TYPE* input, TYPE* output, std::size_t n) const {            \
    for (std::size_t i = 0; i < n; ++i)                                  \
      output[i] = static_cast<TYPE>(tcclMaxFloat(                        \
          static_cast<float>(output[i]), static_cast<float>(input[i]))); \
  }                                                                      \
  template <>                                                            \
  inline void TCCLMinOp<TYPE>::operator()(                               \
      const TYPE* input, TYPE* output, std::size_t n) const {            \
    for (std::size_t i = 0; i < n; ++i)                                  \
      output[i] = static_cast<TYPE>(tcclMinFloat(                        \
          static_cast<float>(output[i]), static_cast<float>(input[i]))); \
  }
TCCL_DEFINE_MINMAX_FLOAT(float)
TCCL_DEFINE_MINMAX_FLOAT(c10::Half)
TCCL_DEFINE_MINMAX_FLOAT(c10::BFloat16)
#undef TCCL_DEFINE_MINMAX_FLOAT

// TCCLEngine

class TCCLEngine {
 public:
  // Per-peer staging buffer size (hence its MR) and max bytes per
  // ibv_post_send. Larger messages split into a chunk loop and reduce
  // incrementally.
  static constexpr size_t kChunkSize = 512 * 1024;

  // Pipeline depth for the bidirectional ring: pieces in flight per stream,
  // with an equal number of double-buffered staging slots. 2 matches JACCL's
  // PIPELINE.
  static constexpr int kPipelineDepth = 2;

  TCCLEngine(
      int rank,
      int size,
      std::vector<std::unique_ptr<TCCLConnection>>& connections,
      std::vector<TCCLSharedBuffer>& send_buffers,
      std::vector<TCCLSharedBuffer>& recv_buffers,
      std::vector<TCCLSharedBuffer>& pipe_send_buffers,
      std::vector<TCCLSharedBuffer>& pipe_recv_buffers,
      const std::chrono::milliseconds& timeout,
      bool ring_topology = false)
      : rank_(rank),
        size_(size),
        connections_(connections),
        send_buffers_(send_buffers),
        recv_buffers_(recv_buffers),
        pipe_send_buffers_(pipe_send_buffers),
        pipe_recv_buffers_(pipe_recv_buffers),
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

  // RING broadcast: pipelined store-and-forward of total_bytes from `root`
  // along the ring (root -> right -> ... -> root's left neighbour). Enables
  // broadcast on a ring-cabled cluster; on a full mesh the dispatch defaults to
  // mesh broadcast.
  void ring_broadcast(void* data, std::size_t total_bytes, int root);

  // Mesh all-gather. `in` is this rank's contiguous shard of `per_rank_bytes`
  // bytes; on exit out_ptrs[r] receives rank r's shard (per_rank_bytes bytes).
  void all_gather(
      const void* in,
      const std::vector<void*>& out_ptrs,
      std::size_t per_rank_bytes);

  // Mesh gather (rooted): the root collects rank r's `per_rank_bytes` shard
  // into out_ptrs[r]; every non-root rank sends `in`. out_ptrs is root-only
  // (size_ entries); pass empty elsewhere.
  void gather(
      const void* in,
      const std::vector<void*>& out_ptrs,
      std::size_t per_rank_bytes,
      int root);

  // Mesh scatter (rooted): the root sends in_ptrs[r] (`per_rank_bytes`) to rank
  // r; every non-root rank receives its slice into `out`. in_ptrs is root-only
  // (size_ entries).
  void scatter(
      const std::vector<const void*>& in_ptrs,
      void* out,
      std::size_t per_rank_bytes,
      int root);

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

  // Point-to-point streams, advanced together: every (peer, direction) is an
  // ordered list of segments, cut into kChunkSize chunks with up to
  // kPipelineDepth in flight on double-buffered staging, so the copy of one
  // chunk overlaps the wire of the next. Segments can be added while it runs;
  // the i-th segment of a stream pairs with the peer's i-th segment in the
  // other direction. Zero-byte segments complete without traffic.
  class P2pStreams {
   public:
    explicit P2pStreams(TCCLEngine& engine)
        : e_(engine),
          send_(static_cast<std::size_t>(engine.size_)),
          recv_(static_cast<std::size_t>(engine.size_)) {}
    void addSend(int peer, const char* ptr, std::size_t bytes, std::size_t tag);
    void addRecv(int peer, char* ptr, std::size_t bytes, std::size_t tag);
    // Posts what it can, polls every busy peer once, and appends the tags of
    // the segments that completed to `done`. Returns whether a chunk
    // completed. Throws DistBackendError on a failed work completion.
    bool poll(std::vector<std::size_t>& done, const char* what);
    // True once every segment added so far has completed.
    bool empty() const;

   private:
    struct Seg {
      const char* src; // send
      char* dst; // recv
      std::size_t bytes;
      std::size_t tag;
    };
    struct Chunk {
      std::size_t seg; // absolute segment index
      std::size_t off;
      std::size_t len;
      int slot;
    };
    struct Stream {
      std::deque<Seg> segs; // segs[i] has absolute index base + i
      std::size_t base = 0;
      std::size_t next = 0; // absolute index of the segment being posted
      std::size_t next_off = 0; // bytes of it posted
      std::size_t front_done = 0; // bytes of segs.front() completed
      std::deque<Chunk> inflight;
      std::size_t chunks = 0; // chunks posted; chunk k uses slot k % depth
    };
    void post(int peer, bool isSend);
    static void retire(Stream& s, std::vector<std::size_t>& done);

    TCCLEngine& e_;
    std::vector<Stream> send_;
    std::vector<Stream> recv_;
  };

  // Mesh (direct) all-to-all. Slab destined for peer p is [send_base +
  // send_off[p], +send_bytes[p]); slab from peer p lands at [recv_base +
  // recv_off[p], +recv_bytes[p]). Handles equal AND uneven (alltoallv) splits.
  // Self slot is a local memcpy. Every peer served simultaneously through
  // P2pStreams, recv before send (UC discipline).
  void all_to_all(
      const char* send_base,
      char* recv_base,
      const std::vector<std::size_t>& send_off,
      const std::vector<std::size_t>& send_bytes,
      const std::vector<std::size_t>& recv_off,
      const std::vector<std::size_t>& recv_bytes);

  // One contiguous run of bytes to send to, or receive from, one peer.
  struct SendSeg {
    const char* ptr;
    std::size_t bytes;
  };
  struct RecvSeg {
    char* ptr;
    std::size_t bytes;
  };

  // Per-peer exchange, the core of all_to_all: sends
  // the segments in sends[p] to, and receives into the segments in recvs[p]
  // from, every peer at once, each list in order. Every (peer, direction)
  // advances on its own, so none waits on another. Zero-byte segments are
  // skipped; the self slot is ignored.
  void exchange(
      const std::vector<std::vector<SendSeg>>& sends,
      const std::vector<std::vector<RecvSeg>>& recvs,
      const char* what);

  // RING all-to-all (neighbours only, store-and-forward), EQUAL splits:
  // send_base holds size_ segments of seg_bytes each (segment j destined for
  // rank j); on exit recv_base holds size_ segments (segment i received from
  // rank i).
  void ring_all_to_all(
      const char* send_base,
      char* recv_base,
      std::size_t seg_bytes);

 private:
  // Core mesh primitive shared by the mesh collectives. Exchanges up to
  // `total_bytes` with every non-self peer in kChunkSize chunks. Per chunk:
  //   - want_recv(peer): post a recv (at 2 ranks only after this rank's own
  //     send of that chunk, since all_reduce reduces into the send buffer; a
  //     send that arrives first stalls until it is posted, Apple TN3205
  //     sec. 12.3)
  //   - send_src(peer, off) != nullptr: memcpy into the send buffer, post a
  //   send
  //   - busy-poll (timeout watchdog) until all WRs complete, firing
  //     on_recv(peer, recv_buf, off, chunk_bytes) per recv
  // Per-peer send_src lets reduce_scatter send each peer a different slab;
  // want_recv + nullable send_src let broadcast post an asymmetric WR set.
  // Pipeline=true drives a depth-kPipelineDepth per-peer pipeline at size_>=3;
  // safe ONLY when on_recv is cheap (a byte copy, e.g. all_gather) - a slow
  // on_recv (reduce) held over a recv buffer starves the all-to-all under UC
  // drop and deadlocks, so reduce collectives keep the default depth-1 path.
  template <
      bool Pipeline = false,
      typename SendSrc,
      typename WantRecv,
      typename OnRecv>
  void mesh_exchange(
      std::size_t total_bytes,
      SendSrc send_src,
      WantRecv want_recv,
      OnRecv on_recv);

  // One ring step shared by reduce-scatter and all-gather: send `send_bytes`
  // from `send_base` to right neighbor `send_peer` AND receive `recv_bytes`
  // from left neighbor `recv_peer`, in kChunkSize pieces (PIPELINE=1, one in
  // flight per direction). Recv before send (UC discipline), per-completion
  // wc.status check + timeout_ watchdog. on_recv reduces/copies each received
  // piece. The two directions carry independent byte counts (ragged last chunk
  // differs).
  template <typename OnRecv>
  void ring_step(
      int send_peer,
      const char* send_base,
      std::size_t send_bytes,
      int recv_peer,
      std::size_t recv_bytes,
      OnRecv on_recv);

  // Bidirectional ring step: two counter-rotating rings concurrently over the
  // two neighbor links, so both links carry traffic in both directions (~2x the
  // unidirectional ring bandwidth; JACCL all_reduce<MAX_DIR=2> equivalent).
  //   ring A (clockwise): send sendA_bytes from sendA_base to `right`, recv
  //     recvA_bytes from `left`, firing on_recv_A per received piece.
  //   ring B (counter-clockwise): send sendB_bytes from sendB_base to `left`,
  //   recv
  //     recvB_bytes from `right`, firing on_recv_B per received piece.
  // Four independent streams (A_send/A_recv/B_send/B_recv), kChunkSize pieces,
  // PIPELINE=1 per stream. Completions demuxed by the (isSend, peer) wr_id: a
  // connection's CQ carries one send stream and one recv stream (different
  // rings). Recv posted before send (UC discipline), per-WR wc.status check,
  // timeout_ watchdog.
  template <typename OnRecvA, typename OnRecvB>
  void bidir_ring_step(
      int right,
      int left,
      const char* sendA_base,
      std::size_t sendA_bytes,
      std::size_t recvA_bytes,
      OnRecvA on_recv_A,
      const char* sendB_base,
      std::size_t sendB_bytes,
      std::size_t recvB_bytes,
      OnRecvB on_recv_B);

  int rank_;
  int size_;
  std::vector<std::unique_ptr<TCCLConnection>>& connections_;
  std::vector<TCCLSharedBuffer>& send_buffers_;
  std::vector<TCCLSharedBuffer>& recv_buffers_;
  // Depth-indexed staging pools for the pipelined bidirectional ring, flat
  // [peer * kPipelineDepth + slot]. Separate from send_buffers_/recv_buffers_
  // (which stay one-per-peer for the PIPELINE=1 mesh/p2p/broadcast paths).
  std::vector<TCCLSharedBuffer>& pipe_send_buffers_;
  std::vector<TCCLSharedBuffer>& pipe_recv_buffers_;
  const std::chrono::milliseconds& timeout_;
  // True => mesh_exchange must never be reached
  const bool ring_topology_{false};
};

} // namespace c10d

// Template implementations. Separated for readability; included at the end
// of this header so consumers get the full definitions without a separate
// include.
#include <torch/csrc/distributed/c10d/ProcessGroupTCCLDetail.ipp>

#endif // USE_C10D_TCCL
