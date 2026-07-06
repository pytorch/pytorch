#pragma once

// Template implementations for ProcessGroupTCCLDetail.hpp. Pulled in via
// the trailing include in that header — do not include directly.

#ifdef USE_C10D_TCCL

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include <c10/util/Exception.h>

#include <torch/csrc/distributed/c10d/TCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/exception.h>

#include <infiniband/verbs.h>  // for ibv_wc, IBV_WC_SUCCESS

namespace c10d {

// Work-request id encoding for completion-event demultiplexing.
//   bit 63          : 1 = send, 0 = recv
//   bits 62..32     : reserved
//   bits 31..0      : peer rank
//
// Every collective posts WRs through mesh_exchange using this scheme so
// completion polling can identify which (peer, direction) a completion is
// for. At PIPELINE=1 there is one in-flight chunk per (peer, direction), so
// (direction, peer) fully identifies a completion; the reserved bits leave room
// to encode extra fields if that changes.
namespace tccl_wr {

constexpr uint64_t kSendBit = 1ULL << 63;

inline uint64_t makeSend(int peer) {
  return kSendBit | static_cast<uint64_t>(peer);
}

inline uint64_t makeRecv(int peer) {
  return static_cast<uint64_t>(peer);
}

inline bool isSend(uint64_t wr_id) {
  return (wr_id & kSendBit) != 0;
}

inline int peer(uint64_t wr_id) {
  return static_cast<int>(wr_id & 0xFFFFFFFFULL);
}

} // namespace tccl_wr

// ---- mesh_exchange ---------------------------------------------------------
//
// The one place the UC-ordering discipline (recv posted before send) and the
// WC-status / duplicate-completion / PG-timeout-watchdog checks live. All four
// collectives below are thin adapters over this. See the declaration in
// ProcessGroupTCCLDetail.hpp for the contract.
template <typename SendSrc, typename WantRecv, typename OnRecv>
void TCCLEngine::mesh_exchange(
    std::size_t total_bytes,
    SendSrc send_src,
    WantRecv want_recv,
    OnRecv on_recv) {
  // Ring topology has null connection slots for all non-neighbor peers;
  // mesh_exchange indexes every peer, so reaching it here would null-deref.
  // Dispatch must force the ring path in ring mode — fail loud if it didn't.
  TORCH_CHECK(
      !ring_topology_,
      "TCCL: mesh_exchange invoked under ring topology — a collective's "
      "dispatch failed to select its ring variant (internal error).");
  if (size_ <= 1) {
    return;
  }

  // Pre-allocated scratch for completion events; a few peers' worth per poll.
  std::vector<ibv_wc> wcs(8);

  std::size_t offset = 0;
  while (offset < total_bytes) {
    const std::size_t chunk_bytes =
        std::min(kChunkSize, total_bytes - offset);

    // Per-peer participation for this chunk.
    std::vector<char> want_send(size_, 0);
    std::vector<char> will_recv(size_, 0);
    int expected_completions = 0;

    // Step 1: pre-post recvs FIRST. UC requires the recv posted before the
    // matching send arrives (Apple TN3205 §12.3 credit-based flow control);
    // post every recv before any send.
    for (int peer = 0; peer < size_; ++peer) {
      if (peer == rank_) continue;
      if (want_recv(peer)) {
        connections_[peer]->postRecv(
            recv_buffers_[peer], chunk_bytes, tccl_wr::makeRecv(peer));
        will_recv[peer] = 1;
        ++expected_completions;
      }
    }

    // Step 2: for every peer with a non-null source, copy this chunk into the
    // peer's send buffer and post a send. send_src is per-peer so
    // reduce_scatter can hand each peer a different slab of `in`; a null
    // source means "no send to this peer" (broadcast on a non-root rank).
    for (int peer = 0; peer < size_; ++peer) {
      if (peer == rank_) continue;
      const void* src = send_src(peer, offset);
      if (src == nullptr) continue;
      std::memcpy(send_buffers_[peer].data(), src, chunk_bytes);
      connections_[peer]->postSend(
          send_buffers_[peer], chunk_bytes, tccl_wr::makeSend(peer));
      want_send[peer] = 1;
      ++expected_completions;
    }

    // Step 3: busy-poll each participating peer's CQ until every posted WR
    // completes. on_recv fires once per recv completion (the recv buffer is
    // safe to read at that point).
    std::vector<char> send_done(size_, 0);
    std::vector<char> recv_done(size_, 0);
    int completions_seen = 0;

    auto deadline =
        std::chrono::steady_clock::now() + timeout_;

    while (completions_seen < expected_completions) {
      bool made_progress = false;
      for (int peer = 0; peer < size_; ++peer) {
        if (peer == rank_) continue;
        const bool send_pending = want_send[peer] && !send_done[peer];
        const bool recv_pending = will_recv[peer] && !recv_done[peer];
        if (!send_pending && !recv_pending) continue;

        int n = connections_[peer]->pollCq(
            static_cast<int>(wcs.size()), wcs.data());
        if (n == 0) continue;
        made_progress = true;

        for (int i = 0; i < n; ++i) {
          const ibv_wc& wc = wcs[i];
          TORCH_CHECK_WITH(
              DistBackendError,
              wc.status == IBV_WC_SUCCESS,
              "TCCL mesh: WC failed status=",
              static_cast<int>(wc.status),
              " (peer=",
              peer,
              ", wr_id=",
              wc.wr_id,
              ")");

          const int wc_peer = tccl_wr::peer(wc.wr_id);
          TORCH_INTERNAL_ASSERT(
              wc_peer == peer,
              "TCCL mesh: completion peer mismatch ",
              wc_peer,
              " vs ",
              peer);

          if (tccl_wr::isSend(wc.wr_id)) {
            TORCH_CHECK_WITH(
                DistBackendError,
                want_send[peer] && !send_done[peer],
                "TCCL mesh: unexpected or duplicate send completion for peer ",
                peer);
            send_done[peer] = 1;
          } else {
            TORCH_CHECK_WITH(
                DistBackendError,
                will_recv[peer] && !recv_done[peer],
                "TCCL mesh: unexpected or duplicate recv completion for peer ",
                peer);
            recv_done[peer] = 1;
            // Recv complete — the peer's chunk is in recv_buffers_[peer].
            on_recv(peer, recv_buffers_[peer].data(), offset, chunk_bytes);
          }
          ++completions_seen;
        }
      }

      if (!made_progress &&
          std::chrono::steady_clock::now() > deadline) {
        std::string outstanding;
        for (int p = 0; p < size_; ++p) {
          if (p == rank_) continue;
          if (want_send[p] && !send_done[p])
            outstanding += " send->" + std::to_string(p);
          if (will_recv[p] && !recv_done[p])
            outstanding += " recv<-" + std::to_string(p);
        }
        TORCH_CHECK_WITH(
            DistBackendError,
            false,
            "TCCL mesh: timed out after ",
            timeout_.count(),
            " ms waiting for completions at offset ",
            offset,
            " (total_bytes=",
            total_bytes,
            ", chunk_bytes=",
            chunk_bytes,
            "). Saw ",
            completions_seen,
            "/",
            expected_completions,
            " completions; outstanding WRs:",
            outstanding,
            ". The poll bound is the process-group timeout "
            "(init_process_group(timeout=) / set_timeout); a persistent stall "
            "here indicates a dropped UC packet (no hardware retransmission).");
      }
    }

    offset += chunk_bytes;
  }
}

// ---- all_reduce ------------------------------------------------------------
template <typename T, typename ReduceOp>
void TCCLEngine::all_reduce(
    T* data,
    std::size_t count,
    ReduceOp reduce_op) {
  TORCH_CHECK_WITH(
      DistBackendError,
      data != nullptr,
      "TCCL mesh: all_reduce called with null data pointer.");
  if (size_ <= 1) {
    return;
  }

  char* base = reinterpret_cast<char*>(data);
  mesh_exchange(
      count * sizeof(T),
      // Send every peer this rank's chunk at `offset`.
      [&](int /*peer*/, std::size_t offset) -> const void* {
        return base + offset;
      },
      // Receive from every peer.
      [](int /*peer*/) { return true; },
      // Reduce the peer's chunk into our output in place.
      [&](int /*peer*/,
          const void* recv_ptr,
          std::size_t offset,
          std::size_t chunk_bytes) {
        reduce_op(
            static_cast<const T*>(recv_ptr),
            reinterpret_cast<T*>(base + offset),
            chunk_bytes / sizeof(T));
      });
}

// ---- ring_step -------------------------------------------------------------
//
// One step of a ring collective: stream `send_bytes` from `send_base` to the
// right neighbor while streaming `recv_bytes` from the left neighbor, both in
// kChunkSize pieces with one piece in flight per direction (PIPELINE=1). The
// recv is posted before the send (UC credit-flow discipline), every completion
// is status-checked, and a timeout_ watchdog bounds the wait. on_recv handles
// each received piece (reduce or copy).
template <typename OnRecv>
void TCCLEngine::ring_step(
    int send_peer,
    const char* send_base,
    std::size_t send_bytes,
    int recv_peer,
    std::size_t recv_bytes,
    OnRecv on_recv) {
  std::vector<ibv_wc> wcs(8);
  std::size_t send_off = 0;       // bytes acknowledged sent
  std::size_t recv_off = 0;       // bytes received and handled
  std::size_t send_inflight = 0;  // size of the in-flight send piece (0 = none)
  std::size_t recv_inflight = 0;  // size of the in-flight recv piece (0 = none)

  auto post_recv_piece = [&]() {
    if (recv_inflight == 0 && recv_off < recv_bytes) {
      recv_inflight = std::min(kChunkSize, recv_bytes - recv_off);
      connections_[recv_peer]->postRecv(
          recv_buffers_[recv_peer], recv_inflight, tccl_wr::makeRecv(recv_peer));
    }
  };
  auto post_send_piece = [&]() {
    if (send_inflight == 0 && send_off < send_bytes) {
      send_inflight = std::min(kChunkSize, send_bytes - send_off);
      std::memcpy(
          send_buffers_[send_peer].data(), send_base + send_off, send_inflight);
      connections_[send_peer]->postSend(
          send_buffers_[send_peer], send_inflight, tccl_wr::makeSend(send_peer));
    }
  };

  // UC discipline: recv posted before the matching send can arrive.
  post_recv_piece();
  post_send_piece();

  auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (recv_off < recv_bytes || send_off < send_bytes) {
    bool made_progress = false;
    // For a ring (size_ > 2) recv_peer != send_peer, so each peer's CQ carries
    // exactly one direction; poll the one(s) with work outstanding.
    for (int peer : {recv_peer, send_peer}) {
      const bool pending =
          (peer == recv_peer ? recv_inflight : send_inflight) != 0;
      if (!pending) continue;
      int n = connections_[peer]->pollCq(static_cast<int>(wcs.size()), wcs.data());
      if (n == 0) continue;
      made_progress = true;
      for (int i = 0; i < n; ++i) {
        const ibv_wc& wc = wcs[i];
        TORCH_CHECK_WITH(
            DistBackendError,
            wc.status == IBV_WC_SUCCESS,
            "TCCL ring: WC failed status=",
            static_cast<int>(wc.status),
            " (peer=",
            peer,
            ", wr_id=",
            wc.wr_id,
            ")");
        if (tccl_wr::isSend(wc.wr_id)) {
          send_off += send_inflight;
          send_inflight = 0;
          post_send_piece();
        } else {
          on_recv(recv_off, recv_buffers_[recv_peer].data(), recv_inflight);
          recv_off += recv_inflight;
          recv_inflight = 0;
          post_recv_piece();
        }
      }
    }
    if (!made_progress && std::chrono::steady_clock::now() > deadline) {
      TORCH_CHECK_WITH(
          DistBackendError,
          false,
          "TCCL ring: timed out after ",
          timeout_.count(),
          " ms. send ",
          send_off,
          "/",
          send_bytes,
          " to peer ",
          send_peer,
          ", recv ",
          recv_off,
          "/",
          recv_bytes,
          " from peer ",
          recv_peer,
          ". The poll bound is the process-group timeout "
          "(init_process_group(timeout=) / set_timeout); a persistent stall "
          "here indicates a dropped UC packet (no hardware retransmission).");
    }
  }
}

// ---- ring_all_reduce -------------------------------------------------------
//
// Bandwidth-optimal ring: reduce-scatter (size_-1 steps) then all-gather
// (size_-1 steps). At step s the rank sends chunk (rank-s) to the right and
// receives chunk (rank-s-1) from the left — the chunk it just reduced — so each
// step depends on the previous (synchronous, no cross-step pipelining yet).
// Traffic is bounded to the two ring neighbors (the avoidance strategy for
// large/bf16 messages).
template <typename T, typename ReduceOp>
void TCCLEngine::ring_all_reduce(
    T* data,
    std::size_t count,
    ReduceOp reduce_op) {
  TORCH_CHECK_WITH(
      DistBackendError,
      data != nullptr,
      "TCCL ring: all_reduce called with null data pointer.");
  if (size_ <= 1) {
    return;
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      size_ > 2,
      "TCCL ring_all_reduce requires size_ > 2 (dispatch uses mesh for <=2).");

  const int N = size_;
  const int right = (rank_ + 1) % N;
  const int left = (rank_ + N - 1) % N;
  char* base = reinterpret_cast<char*>(data);

  // Split `count` elements into N contiguous chunks of `chunk` elements (the
  // last chunk may be short or empty). Chunk c is elements [c*chunk, end_c).
  const std::size_t chunk = (count + static_cast<std::size_t>(N) - 1) / N;
  auto lo = [&](int c) {
    return std::min<std::size_t>(static_cast<std::size_t>(c) * chunk, count);
  };
  auto hi = [&](int c) {
    return std::min<std::size_t>((static_cast<std::size_t>(c) + 1) * chunk, count);
  };
  auto chunk_nbytes = [&](int c) { return (hi(c) - lo(c)) * sizeof(T); };
  auto chunk_ptr = [&](int c) { return base + lo(c) * sizeof(T); };

  // Reduce-scatter: send chunk (rank-s) right, recv+reduce chunk (rank-s-1) left.
  for (int s = 0; s < N - 1; ++s) {
    const int send_c = ((rank_ - s) % N + N) % N;
    const int recv_c = (send_c - 1 + N) % N;
    char* recv_dst = chunk_ptr(recv_c);
    ring_step(
        right,
        chunk_ptr(send_c),
        chunk_nbytes(send_c),
        left,
        chunk_nbytes(recv_c),
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          reduce_op(
              static_cast<const T*>(src),
              reinterpret_cast<T*>(recv_dst + off),
              nbytes / sizeof(T));
        });
  }

  // All-gather: send chunk (rank-s+1) right, recv+overwrite chunk (rank-s) left.
  for (int s = 0; s < N - 1; ++s) {
    const int send_c = ((rank_ - s + 1) % N + N) % N;
    const int recv_c = (send_c - 1 + N) % N;
    char* recv_dst = chunk_ptr(recv_c);
    ring_step(
        right,
        chunk_ptr(send_c),
        chunk_nbytes(send_c),
        left,
        chunk_nbytes(recv_c),
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          std::memcpy(recv_dst + off, src, nbytes);
        });
  }
}

// ---- ring_all_gather -------------------------------------------------------
inline void TCCLEngine::ring_all_gather(
    const void* in,
    const std::vector<void*>& out_ptrs,
    std::size_t per_rank_bytes) {
  TORCH_CHECK_WITH(
      DistBackendError, in != nullptr,
      "TCCL ring: all_gather called with null input pointer.");
  TORCH_CHECK_WITH(
      DistBackendError,
      static_cast<int>(out_ptrs.size()) == size_,
      "TCCL ring all_gather expects ",
      size_,
      " output pointers, got ",
      out_ptrs.size());
  // Self-placement: our own shard goes into our slot.
  std::memcpy(out_ptrs[rank_], in, per_rank_bytes);
  if (size_ <= 1) {
    return;
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      size_ > 2,
      "TCCL ring_all_gather requires size_ > 2 (dispatch uses mesh for <=2).");

  const int N = size_;
  const int right = (rank_ + 1) % N;
  const int left = (rank_ + N - 1) % N;
  // Step k: send shard (rank-k) to the right, receive shard (rank-k-1) from the
  // left (the shard we just received / our own at k=0), copy it into its slot.
  for (int k = 0; k < N - 1; ++k) {
    const int send_c = ((rank_ - k) % N + N) % N;
    const int recv_c = (send_c - 1 + N) % N;
    char* recv_dst = reinterpret_cast<char*>(out_ptrs[recv_c]);
    ring_step(
        right,
        reinterpret_cast<const char*>(out_ptrs[send_c]),
        per_rank_bytes,
        left,
        per_rank_bytes,
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          std::memcpy(recv_dst + off, src, nbytes);
        });
  }
}

// ---- ring_reduce_scatter ---------------------------------------------------
template <typename T, typename ReduceOp>
void TCCLEngine::ring_reduce_scatter(
    const std::vector<const T*>& in_chunks,
    T* out,
    std::size_t count_per_rank,
    ReduceOp reduce_op) {
  TORCH_CHECK_WITH(
      DistBackendError,
      static_cast<int>(in_chunks.size()) == size_ && out != nullptr,
      "TCCL ring: reduce_scatter expects size_ input-chunk pointers and a "
      "non-null output.");
  const int N = size_;
  const std::size_t chunk_bytes = count_per_rank * sizeof(T);
  if (N <= 1) {
    std::memcpy(out, in_chunks[0], chunk_bytes);
    return;
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      N > 2,
      "TCCL ring_reduce_scatter requires size_ > 2 (dispatch uses mesh for <=2).");

  // Ring reduce-scatter accumulates partials in place, so it needs a mutable
  // copy of all N input chunks, gathered here from the per-rank pointers (mesh
  // reduce_scatter reduces straight into `out` and needs no scratch — a fair
  // cost difference to keep in mind for the bench).
  std::vector<char> work(static_cast<std::size_t>(N) * chunk_bytes);
  for (int c = 0; c < N; ++c) {
    std::memcpy(
        work.data() + static_cast<std::size_t>(c) * chunk_bytes,
        in_chunks[c],
        chunk_bytes);
  }
  const int right = (rank_ + 1) % N;
  const int left = (rank_ + N - 1) % N;
  auto chunk = [&](int c) {
    return work.data() + static_cast<std::size_t>(c) * chunk_bytes;
  };
  // Step s: send chunk (rank-s-1) right, recv+reduce chunk (rank-s-2) from left
  // (carry-forward: what we send at s is what we reduced at s-1). This lands the
  // full reduction of chunk rank_ at work[rank_] after N-1 steps. (Indexing is
  // validated against mesh reduce_scatter by the correctness test.)
  for (int s = 0; s < N - 1; ++s) {
    const int send_c = ((rank_ - s - 1) % N + N) % N;
    const int recv_c = ((send_c - 1) % N + N) % N;
    char* recv_chunk = chunk(recv_c);
    ring_step(
        right,
        chunk(send_c),
        chunk_bytes,
        left,
        chunk_bytes,
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          reduce_op(
              static_cast<const T*>(src),
              reinterpret_cast<T*>(recv_chunk + off),
              nbytes / sizeof(T));
        });
  }
  std::memcpy(out, chunk(rank_), chunk_bytes);
}

// ---- ring_broadcast --------------------------------------------------------
//
// Store-and-forward along the ring: chunk by chunk, the root sends to its right;
// each non-root rank receives a chunk from its left, writes it to the output, and
// (unless it is the chain tail = root's left neighbour) forwards it to its right.
// RDMA does not relay, so forwarding is explicit. Per-chunk synchronous, with the
// recv posted before the matching send (UC credit-flow), a wc.status check, and
// the PG-timeout watchdog.
inline void TCCLEngine::ring_broadcast(
    void* data, std::size_t total_bytes, int root) {
  if (size_ <= 1) {
    return;
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      size_ > 2,
      "TCCL ring_broadcast requires size_ > 2 (dispatch uses mesh for <=2).");
  const int N = size_;
  const int pos = ((rank_ - root) % N + N) % N;  // 0 = root ... N-1 = chain tail
  const bool is_root = (pos == 0);
  const bool is_tail = (pos == N - 1);
  const int right = (rank_ + 1) % N;
  const int left = (rank_ + N - 1) % N;
  char* base = reinterpret_cast<char*>(data);
  std::vector<ibv_wc> wcs(8);

  auto poll_one = [&](int peer, bool want_send) {
    auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (true) {
      int n = connections_[peer]->pollCq(static_cast<int>(wcs.size()), wcs.data());
      for (int i = 0; i < n; ++i) {
        TORCH_CHECK_WITH(
            DistBackendError,
            wcs[i].status == IBV_WC_SUCCESS,
            "TCCL ring_broadcast: WC failed status=",
            static_cast<int>(wcs[i].status),
            " (peer=", peer, ")");
        if (tccl_wr::isSend(wcs[i].wr_id) == want_send) {
          return;
        }
      }
      if (n == 0 && std::chrono::steady_clock::now() > deadline) {
        TORCH_CHECK_WITH(
            DistBackendError, false,
            "TCCL ring_broadcast: timed out after ", timeout_.count(),
            " ms (root=", root, ", pos=", pos, "). The poll bound is the process-"
            "group timeout (init_process_group(timeout=) / set_timeout).");
      }
    }
  };

  std::size_t off = 0;
  while (off < total_bytes) {
    const std::size_t n = std::min(kChunkSize, total_bytes - off);
    if (!is_root) {  // receive this chunk from the left, write to output
      connections_[left]->postRecv(recv_buffers_[left], n, tccl_wr::makeRecv(left));
      poll_one(left, /*want_send=*/false);
      std::memcpy(base + off, recv_buffers_[left].data(), n);
    }
    if (!is_tail) {  // forward (root: from data; others: the just-received chunk)
      std::memcpy(send_buffers_[right].data(), base + off, n);
      connections_[right]->postSend(send_buffers_[right], n, tccl_wr::makeSend(right));
      poll_one(right, /*want_send=*/true);
    }
    off += n;
  }
}

// ---- broadcast -------------------------------------------------------------
inline void TCCLEngine::broadcast(
    void* data,
    std::size_t total_bytes,
    int root) {
  TORCH_CHECK_WITH(
      DistBackendError,
      data != nullptr,
      "TCCL mesh: broadcast called with null data pointer.");
  TORCH_CHECK_WITH(
      DistBackendError,
      root >= 0 && root < size_,
      "TCCL mesh: broadcast root ",
      root,
      " out of range [0, ",
      size_,
      ").");
  if (size_ <= 1) {
    return;
  }

  const bool is_root = (rank_ == root);
  char* base = reinterpret_cast<char*>(data);
  mesh_exchange(
      total_bytes,
      // Root sends its data to every peer; non-root sends nothing.
      [&](int /*peer*/, std::size_t offset) -> const void* {
        return is_root ? (base + offset) : nullptr;
      },
      // Only a non-root rank receives, and only from the root.
      [&](int peer) { return !is_root && peer == root; },
      // Copy the received chunk into our buffer (non-root path only).
      [&](int /*peer*/,
          const void* recv_ptr,
          std::size_t offset,
          std::size_t chunk_bytes) {
        std::memcpy(base + offset, recv_ptr, chunk_bytes);
      });
}

// ---- all_gather ------------------------------------------------------------
inline void TCCLEngine::all_gather(
    const void* in,
    const std::vector<void*>& out_ptrs,
    std::size_t per_rank_bytes) {
  TORCH_CHECK_WITH(
      DistBackendError,
      in != nullptr,
      "TCCL mesh: all_gather called with null input pointer.");
  TORCH_CHECK_WITH(
      DistBackendError,
      static_cast<int>(out_ptrs.size()) == size_,
      "TCCL mesh: all_gather expects ",
      size_,
      " output pointers, got ",
      out_ptrs.size());

  const char* in_base = reinterpret_cast<const char*>(in);

  // Self-placement: our own shard goes into our slot.
  std::memcpy(out_ptrs[rank_], in_base, per_rank_bytes);
  if (size_ <= 1) {
    return;
  }

  mesh_exchange(
      per_rank_bytes,
      // Send every peer our whole shard at `offset`.
      [&](int /*peer*/, std::size_t offset) -> const void* {
        return in_base + offset;
      },
      // Receive from every peer.
      [](int /*peer*/) { return true; },
      // Place the peer's shard at its destination slot.
      [&](int peer,
          const void* recv_ptr,
          std::size_t offset,
          std::size_t chunk_bytes) {
        std::memcpy(
            reinterpret_cast<char*>(out_ptrs[peer]) + offset,
            recv_ptr,
            chunk_bytes);
      });
}

// ---- reduce_scatter --------------------------------------------------------
template <typename T, typename ReduceOp>
void TCCLEngine::reduce_scatter(
    const std::vector<const T*>& in_chunks,
    T* out,
    std::size_t count_per_rank,
    ReduceOp reduce_op) {
  TORCH_CHECK_WITH(
      DistBackendError,
      static_cast<int>(in_chunks.size()) == size_ && out != nullptr,
      "TCCL mesh: reduce_scatter expects size_ input-chunk pointers and a "
      "non-null output.");

  // Seed our output with our own contribution — our chunk destined for rank_.
  // (Out-of-place, so we cannot rely on `out` already holding it the way an
  // in-place ring reduce-scatter would.)
  std::memcpy(
      out, in_chunks[rank_], count_per_rank * sizeof(T));
  if (size_ <= 1) {
    return;
  }

  const std::size_t per_rank_bytes = count_per_rank * sizeof(T);

  mesh_exchange(
      per_rank_bytes,
      // Send peer p the chunk of our input destined for it: in_chunks[p].
      [&](int peer, std::size_t offset) -> const void* {
        return reinterpret_cast<const char*>(in_chunks[peer]) + offset;
      },
      // Receive from every peer.
      [](int /*peer*/) { return true; },
      // Reduce the peer's contribution (their chunk destined for rank_) into
      // our output in place.
      [&](int /*peer*/,
          const void* recv_ptr,
          std::size_t offset,
          std::size_t chunk_bytes) {
        reduce_op(
            static_cast<const T*>(recv_ptr),
            reinterpret_cast<T*>(reinterpret_cast<char*>(out) + offset),
            chunk_bytes / sizeof(T));
      });
}

// ---- point-to-point send / recv -------------------------------------------
inline void TCCLEngine::p2p_send(
    int dst, const char* in, std::size_t nbytes) {
  TORCH_CHECK_WITH(
      DistBackendError,
      dst >= 0 && dst < size_ && dst != rank_,
      "TCCL p2p_send: invalid dst ", dst, " (rank ", rank_, ", size ", size_, ").");
  std::vector<ibv_wc> wcs(8);
  std::size_t off = 0;        // bytes acknowledged sent
  std::size_t inflight = 0;   // size of the in-flight chunk (0 = none)
  auto post = [&]() {
    if (inflight == 0 && off < nbytes) {
      inflight = std::min(kChunkSize, nbytes - off);
      std::memcpy(send_buffers_[dst].data(), in + off, inflight);
      connections_[dst]->postSend(
          send_buffers_[dst], inflight, tccl_wr::makeSend(dst));
    }
  };
  post();
  auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (off < nbytes) {
    int n = connections_[dst]->pollCq(static_cast<int>(wcs.size()), wcs.data());
    if (n == 0) {
      TORCH_CHECK_WITH(
          DistBackendError,
          std::chrono::steady_clock::now() <= deadline,
          "TCCL p2p_send: timed out after ", timeout_.count(), " ms sending ",
          off, "/", nbytes, " bytes to peer ", dst,
          ". The poll bound is the process-group timeout; a persistent stall here "
          "indicates a dropped UC packet (no hardware retransmission).");
      continue;
    }
    for (int i = 0; i < n; ++i) {
      TORCH_CHECK_WITH(
          DistBackendError,
          wcs[i].status == IBV_WC_SUCCESS,
          "TCCL p2p_send: WC failed status=", static_cast<int>(wcs[i].status),
          " (peer=", dst, ").");
      off += inflight;
      inflight = 0;
      post();
    }
    deadline = std::chrono::steady_clock::now() + timeout_;  // progress -> reset
  }
}

inline void TCCLEngine::p2p_recv(
    int src, char* out, std::size_t nbytes) {
  TORCH_CHECK_WITH(
      DistBackendError,
      src >= 0 && src < size_ && src != rank_,
      "TCCL p2p_recv: invalid src ", src, " (rank ", rank_, ", size ", size_, ").");
  std::vector<ibv_wc> wcs(8);
  std::size_t off = 0;        // bytes received and handled
  std::size_t inflight = 0;   // size of the in-flight chunk (0 = none)
  auto post = [&]() {
    if (inflight == 0 && off < nbytes) {
      inflight = std::min(kChunkSize, nbytes - off);
      connections_[src]->postRecv(
          recv_buffers_[src], inflight, tccl_wr::makeRecv(src));
    }
  };
  post();
  auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (off < nbytes) {
    int n = connections_[src]->pollCq(static_cast<int>(wcs.size()), wcs.data());
    if (n == 0) {
      TORCH_CHECK_WITH(
          DistBackendError,
          std::chrono::steady_clock::now() <= deadline,
          "TCCL p2p_recv: timed out after ", timeout_.count(), " ms receiving ",
          off, "/", nbytes, " bytes from peer ", src,
          ". The poll bound is the process-group timeout; a persistent stall here "
          "indicates a dropped UC packet (no hardware retransmission).");
      continue;
    }
    for (int i = 0; i < n; ++i) {
      TORCH_CHECK_WITH(
          DistBackendError,
          wcs[i].status == IBV_WC_SUCCESS,
          "TCCL p2p_recv: WC failed status=", static_cast<int>(wcs[i].status),
          " (peer=", src, ").");
      std::memcpy(out + off, recv_buffers_[src].data(), inflight);
      off += inflight;
      inflight = 0;
      post();
    }
    deadline = std::chrono::steady_clock::now() + timeout_;  // progress -> reset
  }
}

// ---- all-to-all (mesh / direct) -------------------------------------------
inline void TCCLEngine::all_to_all(
    const char* send_base,
    char* recv_base,
    const std::vector<std::size_t>& send_off,
    const std::vector<std::size_t>& send_bytes,
    const std::vector<std::size_t>& recv_off,
    const std::vector<std::size_t>& recv_bytes) {
  // Self slot: local copy (no wire).
  if (send_bytes[rank_] > 0) {
    std::memcpy(
        recv_base + recv_off[rank_], send_base + send_off[rank_],
        send_bytes[rank_]);
  }
  if (size_ <= 1) {
    return;
  }

  std::vector<std::size_t> s_done(size_, 0), r_done(size_, 0);
  std::vector<std::size_t> s_inflight(size_, 0), r_inflight(size_, 0);
  std::vector<ibv_wc> wcs(8);

  auto post_recv = [&](int p) {
    if (p != rank_ && r_inflight[p] == 0 && r_done[p] < recv_bytes[p]) {
      r_inflight[p] = std::min(kChunkSize, recv_bytes[p] - r_done[p]);
      connections_[p]->postRecv(
          recv_buffers_[p], r_inflight[p], tccl_wr::makeRecv(p));
    }
  };
  auto post_send = [&](int p) {
    if (p != rank_ && s_inflight[p] == 0 && s_done[p] < send_bytes[p]) {
      s_inflight[p] = std::min(kChunkSize, send_bytes[p] - s_done[p]);
      std::memcpy(
          send_buffers_[p].data(), send_base + send_off[p] + s_done[p],
          s_inflight[p]);
      connections_[p]->postSend(
          send_buffers_[p], s_inflight[p], tccl_wr::makeSend(p));
    }
  };

  // UC discipline: recv posted before the matching send, for every peer.
  for (int p = 0; p < size_; ++p) {
    if (p == rank_) continue;
    post_recv(p);
    post_send(p);
  }

  auto remaining = [&]() {
    for (int p = 0; p < size_; ++p) {
      if (p == rank_) continue;
      if (s_done[p] < send_bytes[p] || r_done[p] < recv_bytes[p]) return true;
    }
    return false;
  };

  auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (remaining()) {
    bool progress = false;
    for (int p = 0; p < size_; ++p) {
      if (p == rank_) continue;
      if (s_inflight[p] == 0 && r_inflight[p] == 0) continue;
      int n = connections_[p]->pollCq(static_cast<int>(wcs.size()), wcs.data());
      if (n == 0) continue;
      progress = true;
      for (int i = 0; i < n; ++i) {
        TORCH_CHECK_WITH(
            DistBackendError,
            wcs[i].status == IBV_WC_SUCCESS,
            "TCCL alltoall(mesh): WC failed status=",
            static_cast<int>(wcs[i].status), " (peer=", p, ").");
        if (tccl_wr::isSend(wcs[i].wr_id)) {
          s_done[p] += s_inflight[p];
          s_inflight[p] = 0;
          post_send(p);
        } else {
          std::memcpy(
              recv_base + recv_off[p] + r_done[p],
              recv_buffers_[p].data(), r_inflight[p]);
          r_done[p] += r_inflight[p];
          r_inflight[p] = 0;
          post_recv(p);
        }
      }
    }
    if (progress) {
      deadline = std::chrono::steady_clock::now() + timeout_;
    } else {
      TORCH_CHECK_WITH(
          DistBackendError,
          std::chrono::steady_clock::now() <= deadline,
          "TCCL alltoall(mesh): timed out after ", timeout_.count(),
          " ms. The poll bound is the process-group timeout; a persistent stall "
          "indicates a dropped UC packet (no hardware retransmission).");
    }
  }
}

// ---- all-to-all (ring / store-and-forward, equal splits) ------------------
inline void TCCLEngine::ring_all_to_all(
    const char* send_base, char* recv_base, std::size_t seg_bytes) {
  const int N = size_;
  // Self segment: local copy.
  std::memcpy(
      recv_base + static_cast<std::size_t>(rank_) * seg_bytes,
      send_base + static_cast<std::size_t>(rank_) * seg_bytes, seg_bytes);
  if (N <= 1) {
    return;
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      N > 2,
      "TCCL ring_all_to_all requires size_ > 2 (dispatch uses mesh for <=2).");

  const int right = (rank_ + 1) % N;
  const int left = (rank_ - 1 + N) % N;

  // Forward block, furthest-destination first: segment for (rank+N-1), (rank+N-2),
  // ..., (rank+1). The receiver peels the LAST segment (its immediate-dst hop) and
  // forwards the front; ordering stays furthest-first at each hop.
  std::vector<char> work(static_cast<std::size_t>(N - 1) * seg_bytes);
  for (int h = N - 1; h >= 1; --h) {
    const int dst = (rank_ + h) % N;
    std::memcpy(
        work.data() + static_cast<std::size_t>(N - 1 - h) * seg_bytes,
        send_base + static_cast<std::size_t>(dst) * seg_bytes, seg_bytes);
  }
  std::vector<char> recv_work(static_cast<std::size_t>(N - 1) * seg_bytes);

  for (int k = 1; k <= N - 1; ++k) {
    const std::size_t block = static_cast<std::size_t>(N - k) * seg_bytes;
    char* rw = recv_work.data();
    ring_step(
        right, work.data(), block, left, block,
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          std::memcpy(rw + off, src, nbytes);
        });
    // Peel the last segment of the received block — destined for this rank,
    // originated (k hops left) by rank (rank_ - k).
    const int origin = (rank_ - k + N) % N;
    std::memcpy(
        recv_base + static_cast<std::size_t>(origin) * seg_bytes,
        recv_work.data() + static_cast<std::size_t>(N - k - 1) * seg_bytes,
        seg_bytes);
    // Forward the front (N-k-1) segments next step.
    if (N - k - 1 > 0) {
      std::memcpy(
          work.data(), recv_work.data(),
          static_cast<std::size_t>(N - k - 1) * seg_bytes);
    }
  }
}

} // namespace c10d

#endif // USE_C10D_TCCL
