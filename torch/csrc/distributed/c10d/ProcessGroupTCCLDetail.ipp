#pragma once

// Template implementations for ProcessGroupTCCLDetail.hpp.

#ifdef USE_C10D_TCCL

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <c10/util/Exception.h>

#include <torch/csrc/distributed/c10d/TCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/exception.h>

// For ibv_wc, IBV_WC_SUCCESS
#include <infiniband/verbs.h>

namespace c10d {

// Work-request id encoding for completion-event demultiplexing.
//   bit 63          : 1 = send, 0 = recv
//   bits 40..33     : pipeline slot (0..kPipelineDepth-1)
//   bits 62..41, 32 : reserved
//   bits 31..0      : peer rank
//
// PIPELINE=1 callers omit the slot (defaults 0), so (direction, peer) still
// fully identifies their single in-flight chunk. The pipelined bidirectional
// ring keeps up to kPipelineDepth pieces in flight per (peer, direction) and
// uses the slot field to tell which staging buffer a completion refers to.
namespace tccl_wr {

constexpr uint64_t kSendBit = 1ULL << 63;
constexpr uint64_t kSlotShift = 33;
constexpr uint64_t kSlotMask = 0xFFULL;

inline uint64_t makeSend(int peer, int slot = 0) {
  return kSendBit | (static_cast<uint64_t>(slot) << kSlotShift) |
      static_cast<uint64_t>(peer);
}

inline uint64_t makeRecv(int peer, int slot = 0) {
  return (static_cast<uint64_t>(slot) << kSlotShift) |
      static_cast<uint64_t>(peer);
}

inline bool isSend(uint64_t wr_id) {
  return (wr_id & kSendBit) != 0;
}

inline int peer(uint64_t wr_id) {
  return static_cast<int>(wr_id & 0xFFFFFFFFULL);
}

inline int slot(uint64_t wr_id) {
  return static_cast<int>((wr_id >> kSlotShift) & kSlotMask);
}

} // namespace tccl_wr

// mesh_exchange
//
// The one place the UC-ordering discipline (recv before send) and the WC-status
// / duplicate-completion / timeout-watchdog checks live. The mesh collectives
// are thin adapters over this; contract in the header.
template <bool Pipeline, typename SendSrc, typename WantRecv, typename OnRecv>
void TCCLEngine::mesh_exchange(
    std::size_t total_bytes,
    SendSrc send_src,
    WantRecv want_recv,
    OnRecv on_recv) {
  // Ring topology has null connection slots for all non-neighbor peers;
  // mesh_exchange indexes every peer, so reaching it here would null-deref.
  // Dispatch must force the ring path in ring mode - fail loud if it didn't.
  TORCH_CHECK(
      !ring_topology_,
      "TCCL: mesh_exchange invoked under ring topology - a collective's "
      "dispatch failed to select its ring variant (internal error).");
  if (size_ <= 1) {
    return;
  }

  // 2-node fast path: depth-kPipelineDepth pipeline over the single peer, so the
  // per-piece memcpy/reduce overlaps the wire (the PIPELINE=1 general path below
  // ran them serially - worst at N=2 where mesh is the only option and the reduce
  // sat on the critical path). Restricted to size_==2: with one peer there is no
  // round-robin, so a recv buffer briefly held by the reduce cannot starve other
  // peers' recvs. At size_>=3 that starvation deadlocks under UC drop semantics,
  // so the general path keeps its synchronized one-chunk-per-round discipline.
  if (size_ == 2) {
    const int peer = 1 - rank_;
    const bool do_send = (send_src(peer, 0) != nullptr);
    const bool do_recv = want_recv(peer);
    constexpr int D = kPipelineDepth;
    std::vector<ibv_wc> wcs(2 * D + 2);
    std::size_t send_off = 0, recv_off = 0, send_done = 0, recv_done = 0;
    int send_if = 0, recv_if = 0;
    std::size_t slen[D] = {0}, roff[D] = {0}, rlen[D] = {0};
    auto fill_recv = [&]() {
      if (!do_recv) return;
      for (int s = 0; s < D; ++s) {
        const std::size_t idx = static_cast<std::size_t>(peer) * D + s;
        if (rlen[s] == 0 && recv_off < total_bytes) {
          const std::size_t L = std::min(kChunkSize, total_bytes - recv_off);
          roff[s] = recv_off;
          rlen[s] = L;
          connections_[peer]->postRecv(
              pipe_recv_buffers_[idx], L, tccl_wr::makeRecv(peer, s));
          recv_off += L;
          ++recv_if;
        }
      }
    };
    auto fill_send = [&]() {
      if (!do_send) return;
      for (int s = 0; s < D; ++s) {
        const std::size_t idx = static_cast<std::size_t>(peer) * D + s;
        if (slen[s] == 0 && send_off < total_bytes) {
          const std::size_t L = std::min(kChunkSize, total_bytes - send_off);
          const void* src = send_src(peer, send_off);
          std::memcpy(pipe_send_buffers_[idx].data(), src, L);
          slen[s] = L;
          connections_[peer]->postSend(
              pipe_send_buffers_[idx], L, tccl_wr::makeSend(peer, s));
          send_off += L;
          ++send_if;
        }
      }
    };
    fill_recv();
    fill_send();
    auto deadline = std::chrono::steady_clock::now() + timeout_;
    while ((do_send && send_done < total_bytes) ||
           (do_recv && recv_done < total_bytes)) {
      int n = connections_[peer]->pollCq(
          static_cast<int>(wcs.size()), wcs.data());
      if (n == 0) {
        TORCH_CHECK_WITH(
            DistBackendError,
            std::chrono::steady_clock::now() <= deadline,
            "TCCL mesh(2-node): timed out after ", timeout_.count(),
            " ms (send ", send_done, "/", do_send ? total_bytes : 0,
            ", recv ", recv_done, "/", do_recv ? total_bytes : 0, " peer ", peer,
            "). The poll bound is the process-group timeout; a persistent stall "
            "here indicates a dropped UC packet (no hardware retransmission).");
        continue;
      }
      for (int i = 0; i < n; ++i) {
        TORCH_CHECK_WITH(
            DistBackendError,
            wcs[i].status == IBV_WC_SUCCESS,
            "TCCL mesh(2-node): WC failed status=",
            static_cast<int>(wcs[i].status), " (peer=", peer, ").");
        const int s = tccl_wr::slot(wcs[i].wr_id);
        const std::size_t idx = static_cast<std::size_t>(peer) * D + s;
        if (tccl_wr::isSend(wcs[i].wr_id)) {
          send_done += slen[s];
          slen[s] = 0;
          --send_if;
          fill_send();
        } else {
          on_recv(peer, pipe_recv_buffers_[idx].data(), roff[s], rlen[s]);
          recv_done += rlen[s];
          rlen[s] = 0;
          --recv_if;
          fill_recv();
        }
      }
      deadline = std::chrono::steady_clock::now() + timeout_;
    }
    return;
  }

  // Pipelined multi-peer path (Pipeline=true, size_>=3): depth-kPipelineDepth per
  // peer over the full mesh, so all N-1 links stay busy and the per-piece copy
  // overlaps the wire. Only enabled for copy-based collectives (all_gather): the
  // on_recv is a fast memcpy that frees the recv buffer promptly, so a peer cannot
  // starve while another is serviced. A slow on_recv (reduce) would deadlock here,
  // which is why reduce collectives use the depth-1 path below.
  if (Pipeline && size_ > 2) {
    constexpr int D = kPipelineDepth;
    std::vector<ibv_wc> wcs(2 * D + 2);
    std::vector<char> sends_to(size_, 0), recvs_from(size_, 0);
    for (int p = 0; p < size_; ++p) {
      if (p == rank_) continue;
      sends_to[p] = (send_src(p, 0) != nullptr) ? 1 : 0;
      recvs_from[p] = want_recv(p) ? 1 : 0;
    }
    std::vector<std::size_t> send_off(size_, 0), recv_off(size_, 0);
    std::vector<std::size_t> send_done(size_, 0), recv_done(size_, 0);
    std::vector<int> send_if(size_, 0), recv_if(size_, 0);
    const std::size_t NS = static_cast<std::size_t>(size_) * D;
    std::vector<std::size_t> slen(NS, 0), roff(NS, 0), rlen(NS, 0);
    auto fill_recv = [&](int p) {
      if (!recvs_from[p]) return;
      for (int s = 0; s < D; ++s) {
        const std::size_t idx = static_cast<std::size_t>(p) * D + s;
        if (rlen[idx] == 0 && recv_off[p] < total_bytes) {
          const std::size_t L = std::min(kChunkSize, total_bytes - recv_off[p]);
          roff[idx] = recv_off[p];
          rlen[idx] = L;
          connections_[p]->postRecv(
              pipe_recv_buffers_[idx], L, tccl_wr::makeRecv(p, s));
          recv_off[p] += L;
          ++recv_if[p];
        }
      }
    };
    auto fill_send = [&](int p) {
      if (!sends_to[p]) return;
      for (int s = 0; s < D; ++s) {
        const std::size_t idx = static_cast<std::size_t>(p) * D + s;
        if (slen[idx] == 0 && send_off[p] < total_bytes) {
          const std::size_t L = std::min(kChunkSize, total_bytes - send_off[p]);
          const void* src = send_src(p, send_off[p]);
          std::memcpy(pipe_send_buffers_[idx].data(), src, L);
          slen[idx] = L;
          connections_[p]->postSend(
              pipe_send_buffers_[idx], L, tccl_wr::makeSend(p, s));
          send_off[p] += L;
          ++send_if[p];
        }
      }
    };
    for (int p = 0; p < size_; ++p)
      if (p != rank_) fill_recv(p);
    for (int p = 0; p < size_; ++p)
      if (p != rank_) fill_send(p);
    auto all_done = [&]() {
      for (int p = 0; p < size_; ++p) {
        if (p == rank_) continue;
        if (recvs_from[p] && recv_done[p] < total_bytes) return false;
        if (sends_to[p] && send_done[p] < total_bytes) return false;
      }
      return true;
    };
    auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (!all_done()) {
      bool made_progress = false;
      for (int p = 0; p < size_; ++p) {
        if (p == rank_) continue;
        if (send_if[p] == 0 && recv_if[p] == 0) continue;
        int n = connections_[p]->pollCq(static_cast<int>(wcs.size()), wcs.data());
        if (n == 0) continue;
        made_progress = true;
        for (int i = 0; i < n; ++i) {
          const ibv_wc& wc = wcs[i];
          TORCH_CHECK_WITH(
              DistBackendError, wc.status == IBV_WC_SUCCESS,
              "TCCL mesh(pipe): WC failed status=", static_cast<int>(wc.status),
              " (peer=", p, ", wr_id=", wc.wr_id, ")");
          const int s = tccl_wr::slot(wc.wr_id);
          const std::size_t idx = static_cast<std::size_t>(p) * D + s;
          if (tccl_wr::isSend(wc.wr_id)) {
            send_done[p] += slen[idx];
            slen[idx] = 0;
            --send_if[p];
            fill_send(p);
          } else {
            on_recv(p, pipe_recv_buffers_[idx].data(), roff[idx], rlen[idx]);
            recv_done[p] += rlen[idx];
            rlen[idx] = 0;
            --recv_if[p];
            fill_recv(p);
          }
        }
      }
      if (made_progress) {
        deadline = std::chrono::steady_clock::now() + timeout_;
      } else if (std::chrono::steady_clock::now() > deadline) {
        std::string outstanding;
        for (int p = 0; p < size_; ++p) {
          if (p == rank_) continue;
          if (sends_to[p] && send_done[p] < total_bytes)
            outstanding += " send->" + std::to_string(p);
          if (recvs_from[p] && recv_done[p] < total_bytes)
            outstanding += " recv<-" + std::to_string(p);
        }
        TORCH_CHECK_WITH(
            DistBackendError, false,
            "TCCL mesh(pipe): timed out after ", timeout_.count(),
            " ms (total_bytes=", total_bytes, "). outstanding:", outstanding,
            ". A persistent stall here indicates a dropped UC packet.");
      }
    }
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
    // matching send arrives (Apple TN3205 sec. 12.3 credit-based flow control);
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
            // Recv complete - the peer's chunk is in recv_buffers_[peer].
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

// all_reduce
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

// ring_step
//
// One ring step: stream `send_bytes` to the right neighbor while streaming
// `recv_bytes` from the left, in kChunkSize pieces, one in flight per direction
// (PIPELINE=1). Recv before send (UC credit-flow), every completion
// status-checked, timeout_ watchdog. on_recv handles each piece (reduce or copy).
template <typename OnRecv>
void TCCLEngine::ring_step(
    int send_peer,
    const char* send_base,
    std::size_t send_bytes,
    int recv_peer,
    std::size_t recv_bytes,
    OnRecv on_recv) {
  std::vector<ibv_wc> wcs(8);
  // Bytes acknowledged sent
  std::size_t send_off = 0;
  // Bytes received and handled
  std::size_t recv_off = 0;
  // In-flight send piece size - 0 = none
  std::size_t send_inflight = 0;
  // In-flight recv piece size - 0 = none
  std::size_t recv_inflight = 0;

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
    if (!made_progress) {
      if (std::chrono::steady_clock::now() > deadline) {
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
}

// bidir_ring_step (contract in the header)
//
// Four independent piece-streams over the two neighbor connections. conn[right]
// carries A_send (send completions) + B_recv (recv completions); conn[left]
// carries B_send + A_recv. Each stream keeps up to kPipelineDepth pieces in
// flight with an equal number of double-buffered staging slots, so a piece's
// memcpy/reduce overlaps the wire transfer of another piece (JACCL PIPELINE=2).
// Completions are demuxed by (isSend, slot) from the wr_id. Recv posted before
// send (UC), per-WR wc.status check, timeout_ watchdog.
template <typename OnRecvA, typename OnRecvB>
void TCCLEngine::bidir_ring_step(
    int right,
    int left,
    const char* sendA_base,
    std::size_t sendA_bytes,
    std::size_t recvA_bytes,
    OnRecvA on_recv_A,
    const char* sendB_base,
    std::size_t sendB_bytes,
    std::size_t recvB_bytes,
    OnRecvB on_recv_B) {
  constexpr int D = kPipelineDepth;
  std::vector<ibv_wc> wcs(2 * D + 2);

  // Per-stream state: next byte to post, bytes completed, in-flight slot count,
  // and per-slot (offset, length) so a completion knows which piece it was
  // (length 0 = slot free). Send buffers indexed by destination peer, recv by
  // source peer, both in the depth pool [peer * D + slot].
  std::size_t asend_next = 0, asend_done = 0;
  std::size_t bsend_next = 0, bsend_done = 0;
  std::size_t arecv_next = 0, arecv_done = 0;
  std::size_t brecv_next = 0, brecv_done = 0;
  int asend_if = 0, bsend_if = 0, arecv_if = 0, brecv_if = 0;
  std::size_t asoff[D] = {0}, aslen[D] = {0}; // A send -> right
  std::size_t bsoff[D] = {0}, bslen[D] = {0}; // B send -> left
  std::size_t aroff[D] = {0}, arlen[D] = {0}; // A recv <- left
  std::size_t broff[D] = {0}, brlen[D] = {0}; // B recv <- right

  auto fill_send = [&](int peer,
                       const char* base,
                       std::size_t total,
                       std::size_t& next_off,
                       int& inflight,
                       std::size_t* soff,
                       std::size_t* slen) {
    for (int s = 0; s < D; ++s) {
      if (slen[s] == 0 && next_off < total) {
        const std::size_t L = std::min(kChunkSize, total - next_off);
        auto& buf =
            pipe_send_buffers_[static_cast<std::size_t>(peer) * D + s];
        std::memcpy(buf.data(), base + next_off, L);
        soff[s] = next_off;
        slen[s] = L;
        connections_[peer]->postSend(buf, L, tccl_wr::makeSend(peer, s));
        next_off += L;
        ++inflight;
      }
    }
  };
  auto fill_recv = [&](int peer,
                       std::size_t total,
                       std::size_t& next_off,
                       int& inflight,
                       std::size_t* soff,
                       std::size_t* slen) {
    for (int s = 0; s < D; ++s) {
      if (slen[s] == 0 && next_off < total) {
        const std::size_t L = std::min(kChunkSize, total - next_off);
        auto& buf =
            pipe_recv_buffers_[static_cast<std::size_t>(peer) * D + s];
        soff[s] = next_off;
        slen[s] = L;
        connections_[peer]->postRecv(buf, L, tccl_wr::makeRecv(peer, s));
        next_off += L;
        ++inflight;
      }
    }
  };

  // UC discipline: post recvs before sends.
  fill_recv(left, recvA_bytes, arecv_next, arecv_if, aroff, arlen);
  fill_recv(right, recvB_bytes, brecv_next, brecv_if, broff, brlen);
  fill_send(right, sendA_base, sendA_bytes, asend_next, asend_if, asoff, aslen);
  fill_send(left, sendB_base, sendB_bytes, bsend_next, bsend_if, bsoff, bslen);

  auto done = [&]() {
    return asend_done >= sendA_bytes && arecv_done >= recvA_bytes &&
        bsend_done >= sendB_bytes && brecv_done >= recvB_bytes;
  };

  auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (!done()) {
    bool made_progress = false;

    if (asend_if != 0 || brecv_if != 0) {
      int n = connections_[right]->pollCq(
          static_cast<int>(wcs.size()), wcs.data());
      for (int i = 0; i < n; ++i) {
        const ibv_wc& wc = wcs[i];
        TORCH_CHECK_WITH(
            DistBackendError,
            wc.status == IBV_WC_SUCCESS,
            "TCCL bidir-ring: WC failed status=",
            static_cast<int>(wc.status),
            " (peer=",
            right,
            ", wr_id=",
            wc.wr_id,
            ")");
        made_progress = true;
        const int s = tccl_wr::slot(wc.wr_id);
        if (tccl_wr::isSend(wc.wr_id)) {
          asend_done += aslen[s];
          aslen[s] = 0;
          --asend_if;
          fill_send(
              right, sendA_base, sendA_bytes, asend_next, asend_if, asoff, aslen);
        } else {
          on_recv_B(
              broff[s],
              pipe_recv_buffers_[static_cast<std::size_t>(right) * D + s].data(),
              brlen[s]);
          brecv_done += brlen[s];
          brlen[s] = 0;
          --brecv_if;
          fill_recv(right, recvB_bytes, brecv_next, brecv_if, broff, brlen);
        }
      }
    }

    if (bsend_if != 0 || arecv_if != 0) {
      int n = connections_[left]->pollCq(
          static_cast<int>(wcs.size()), wcs.data());
      for (int i = 0; i < n; ++i) {
        const ibv_wc& wc = wcs[i];
        TORCH_CHECK_WITH(
            DistBackendError,
            wc.status == IBV_WC_SUCCESS,
            "TCCL bidir-ring: WC failed status=",
            static_cast<int>(wc.status),
            " (peer=",
            left,
            ", wr_id=",
            wc.wr_id,
            ")");
        made_progress = true;
        const int s = tccl_wr::slot(wc.wr_id);
        if (tccl_wr::isSend(wc.wr_id)) {
          bsend_done += bslen[s];
          bslen[s] = 0;
          --bsend_if;
          fill_send(
              left, sendB_base, sendB_bytes, bsend_next, bsend_if, bsoff, bslen);
        } else {
          on_recv_A(
              aroff[s],
              pipe_recv_buffers_[static_cast<std::size_t>(left) * D + s].data(),
              arlen[s]);
          arecv_done += arlen[s];
          arlen[s] = 0;
          --arecv_if;
          fill_recv(left, recvA_bytes, arecv_next, arecv_if, aroff, arlen);
        }
      }
    }

    if (made_progress) {
      deadline = std::chrono::steady_clock::now() + timeout_;
    } else if (std::chrono::steady_clock::now() > deadline) {
      TORCH_CHECK_WITH(
          DistBackendError,
          false,
          "TCCL bidir-ring: timed out after ",
          timeout_.count(),
          " ms. A send ",
          asend_done,
          "/",
          sendA_bytes,
          "->",
          right,
          ", A recv ",
          arecv_done,
          "/",
          recvA_bytes,
          "<-",
          left,
          ", B send ",
          bsend_done,
          "/",
          sendB_bytes,
          "->",
          left,
          ", B recv ",
          brecv_done,
          "/",
          recvB_bytes,
          "<-",
          right,
          ". The poll bound is the process-group timeout; a persistent stall "
          "here indicates a dropped UC packet (no hardware retransmission).");
    }
  }
}

// ring_all_reduce
//
// Bidirectional (double) ring: two counter-rotating rings over disjoint halves
// of `data`, in place. Ring A (clockwise, d=+1) reduces the first half; ring B
// (counter-clockwise, d=-1) reduces the second half. Both run their reduce-
// scatter (size_-1 steps) + all-gather (size_-1 steps) driven IN LOCKSTEP by
// bidir_ring_step, so both neighbor links carry traffic in both directions
// (~2x the unidirectional ring bandwidth). d=+1 reproduces the classic single-
// ring index math; d=-1 is the same algorithm on the reversed ring.
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

  // Two halves on the same buffer: ring A = [0, countA), ring B = [countA, count).
  const std::size_t countA = count / 2;
  const std::size_t countB = count - countA;
  T* dataA = data;
  T* dataB = data + countA;

  // Per-half chunking into N pieces (ragged/empty last piece via the clamps).
  const std::size_t chunkA = (countA + static_cast<std::size_t>(N) - 1) / N;
  const std::size_t chunkB = (countB + static_cast<std::size_t>(N) - 1) / N;
  auto ptrA = [&](int c) {
    return reinterpret_cast<char*>(dataA) +
        std::min<std::size_t>(static_cast<std::size_t>(c) * chunkA, countA) *
        sizeof(T);
  };
  auto nbA = [&](int c) {
    const std::size_t lo =
        std::min<std::size_t>(static_cast<std::size_t>(c) * chunkA, countA);
    const std::size_t hi = std::min<std::size_t>(
        (static_cast<std::size_t>(c) + 1) * chunkA, countA);
    return (hi - lo) * sizeof(T);
  };
  auto ptrB = [&](int c) {
    return reinterpret_cast<char*>(dataB) +
        std::min<std::size_t>(static_cast<std::size_t>(c) * chunkB, countB) *
        sizeof(T);
  };
  auto nbB = [&](int c) {
    const std::size_t lo =
        std::min<std::size_t>(static_cast<std::size_t>(c) * chunkB, countB);
    const std::size_t hi = std::min<std::size_t>(
        (static_cast<std::size_t>(c) + 1) * chunkB, countB);
    return (hi - lo) * sizeof(T);
  };

  // Reduce-scatter: A clockwise, B counter-clockwise.
  for (int s = 0; s < N - 1; ++s) {
    const int aSend = ((rank_ - s) % N + N) % N;
    const int aRecv = ((aSend - 1) % N + N) % N;
    const int bSend = ((rank_ + s) % N + N) % N;
    const int bRecv = ((bSend + 1) % N + N) % N;
    char* aRecvDst = ptrA(aRecv);
    char* bRecvDst = ptrB(bRecv);
    bidir_ring_step(
        right,
        left,
        ptrA(aSend),
        nbA(aSend),
        nbA(aRecv),
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          reduce_op(
              static_cast<const T*>(src),
              reinterpret_cast<T*>(aRecvDst + off),
              nbytes / sizeof(T));
        },
        ptrB(bSend),
        nbB(bSend),
        nbB(bRecv),
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          reduce_op(
              static_cast<const T*>(src),
              reinterpret_cast<T*>(bRecvDst + off),
              nbytes / sizeof(T));
        });
  }

  // All-gather: A clockwise, B counter-clockwise.
  for (int s = 0; s < N - 1; ++s) {
    const int aSend = ((rank_ - s + 1) % N + N) % N;
    const int aRecv = ((aSend - 1) % N + N) % N;
    const int bSend = ((rank_ + s - 1) % N + N) % N;
    const int bRecv = ((bSend + 1) % N + N) % N;
    char* aRecvDst = ptrA(aRecv);
    char* bRecvDst = ptrB(bRecv);
    bidir_ring_step(
        right,
        left,
        ptrA(aSend),
        nbA(aSend),
        nbA(aRecv),
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          std::memcpy(aRecvDst + off, src, nbytes);
        },
        ptrB(bSend),
        nbB(bSend),
        nbB(bRecv),
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          std::memcpy(bRecvDst + off, src, nbytes);
        });
  }
}

// ring_all_gather
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
  // Bidirectional: split each shard's bytes in half. Ring A circulates half A of
  // every shard clockwise (send shard (rank-k) right, recv shard (rank-k-1) left);
  // ring B circulates half B counter-clockwise (send shard (rank+k) left, recv
  // shard (rank+k+1) right). Both cover all N-1 other shards, so out_ptrs[r] gets
  // its half A via ring A and half B via ring B.
  const std::size_t halfA = per_rank_bytes / 2;
  const std::size_t halfB = per_rank_bytes - halfA;
  for (int k = 0; k < N - 1; ++k) {
    const int aSend = ((rank_ - k) % N + N) % N;
    const int aRecv = ((aSend - 1) % N + N) % N;
    const int bSend = ((rank_ + k) % N + N) % N;
    const int bRecv = ((bSend + 1) % N + N) % N;
    char* aRecvDst = reinterpret_cast<char*>(out_ptrs[aRecv]);
    char* bRecvDst = reinterpret_cast<char*>(out_ptrs[bRecv]) + halfA;
    bidir_ring_step(
        right,
        left,
        reinterpret_cast<const char*>(out_ptrs[aSend]),
        halfA,
        halfA,
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          std::memcpy(aRecvDst + off, src, nbytes);
        },
        reinterpret_cast<const char*>(out_ptrs[bSend]) + halfA,
        halfB,
        halfB,
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          std::memcpy(bRecvDst + off, src, nbytes);
        });
  }
}

// ring_reduce_scatter
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
  // copy of all N input chunks (mesh reduce_scatter reduces straight into `out`
  // with no scratch). Bidirectional: split each chunk in half - ring A reduce-
  // scatters half A clockwise (d=+1), ring B half B counter-clockwise (d=-1),
  // driven in lockstep by bidir_ring_step so both neighbor links carry traffic
  // both ways (~2x the unidirectional bandwidth, mirroring ring_all_gather).
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
  const std::size_t abytes = (count_per_rank / 2) * sizeof(T);  // half A
  const std::size_t bbytes = chunk_bytes - abytes;              // half B
  // Ring A (d=+1): send chunk (rank-s-1) right, recv+reduce (rank-s-2) from left.
  // Ring B (d=-1): the mirror - send chunk (rank+s+1) left, recv+reduce (rank+s+2)
  // from right. Each lands work[rank_]'s half fully reduced after N-1 steps.
  for (int s = 0; s < N - 1; ++s) {
    const int aSend = ((rank_ - s - 1) % N + N) % N;
    const int aRecv = ((aSend - 1) % N + N) % N;
    const int bSend = ((rank_ + s + 1) % N + N) % N;
    const int bRecv = ((bSend + 1) % N + N) % N;
    char* aRecvHalf = chunk(aRecv);
    char* bRecvHalf = chunk(bRecv) + abytes;
    bidir_ring_step(
        right,
        left,
        chunk(aSend),
        abytes,
        abytes,
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          reduce_op(
              static_cast<const T*>(src),
              reinterpret_cast<T*>(aRecvHalf + off),
              nbytes / sizeof(T));
        },
        chunk(bSend) + abytes,
        bbytes,
        bbytes,
        [&](std::size_t off, const void* src, std::size_t nbytes) {
          reduce_op(
              static_cast<const T*>(src),
              reinterpret_cast<T*>(bRecvHalf + off),
              nbytes / sizeof(T));
        });
  }
  std::memcpy(out, chunk(rank_), chunk_bytes);
}

// ring_broadcast
//
// Store-and-forward along the ring: each non-root rank receives a chunk from its
// left, writes it, and (unless it is the chain tail = root's left neighbor)
// forwards it right. RDMA does not relay, so forwarding is explicit. Per-chunk
// synchronous, recv before send (UC credit-flow), wc.status check, timeout watchdog.
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
  // 0 = root ... N-1 = chain tail
  const int pos = ((rank_ - root) % N + N) % N;
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
    // Receive this chunk from the left, write to output
    if (!is_root) {
      connections_[left]->postRecv(recv_buffers_[left], n, tccl_wr::makeRecv(left));
      poll_one(left, /*want_send=*/false);
      std::memcpy(base + off, recv_buffers_[left].data(), n);
    }
    // Forward - root sends from data, others the just-received chunk
    if (!is_tail) {
      std::memcpy(send_buffers_[right].data(), base + off, n);
      connections_[right]->postSend(send_buffers_[right], n, tccl_wr::makeSend(right));
      poll_one(right, /*want_send=*/true);
    }
    off += n;
  }
}

// broadcast
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

// all_gather
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

  mesh_exchange<true>(
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

// gather (rooted): root collects each shard; non-root sends to root.
inline void TCCLEngine::gather(
    const void* in,
    const std::vector<void*>& out_ptrs,
    std::size_t per_rank_bytes,
    int root) {
  TORCH_CHECK_WITH(
      DistBackendError, in != nullptr,
      "TCCL mesh: gather called with null input pointer.");
  TORCH_CHECK_WITH(
      DistBackendError, root >= 0 && root < size_,
      "TCCL mesh: gather root ", root, " out of range [0, ", size_, ").");
  const bool is_root = (rank_ == root);
  const char* in_base = reinterpret_cast<const char*>(in);
  if (is_root) {
    TORCH_CHECK_WITH(
        DistBackendError, static_cast<int>(out_ptrs.size()) == size_,
        "TCCL mesh: gather expects ", size_, " output pointers on the root.");
    // Root's own shard.
    std::memcpy(out_ptrs[rank_], in_base, per_rank_bytes);
  }
  if (size_ <= 1) {
    return;
  }
  mesh_exchange<true>(
      per_rank_bytes,
      // Non-root sends its shard to root.
      [&](int peer, std::size_t offset) -> const void* {
        return (!is_root && peer == root) ? (in_base + offset) : nullptr;
      },
      // Only root receives.
      [&](int /*peer*/) { return is_root; },
      // Place each peer's shard.
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

// scatter (rooted): root sends each slice; non-root receives from root.
inline void TCCLEngine::scatter(
    const std::vector<const void*>& in_ptrs,
    void* out,
    std::size_t per_rank_bytes,
    int root) {
  TORCH_CHECK_WITH(
      DistBackendError, out != nullptr,
      "TCCL mesh: scatter called with null output pointer.");
  TORCH_CHECK_WITH(
      DistBackendError, root >= 0 && root < size_,
      "TCCL mesh: scatter root ", root, " out of range [0, ", size_, ").");
  const bool is_root = (rank_ == root);
  char* out_base = reinterpret_cast<char*>(out);
  if (is_root) {
    TORCH_CHECK_WITH(
        DistBackendError, static_cast<int>(in_ptrs.size()) == size_,
        "TCCL mesh: scatter expects ", size_, " input pointers on the root.");
    // Root's own slice.
    std::memcpy(out_base, in_ptrs[rank_], per_rank_bytes);
  }
  if (size_ <= 1) {
    return;
  }
  mesh_exchange<true>(
      per_rank_bytes,
      // Root sends each peer its slice.
      [&](int peer, std::size_t offset) -> const void* {
        return is_root
            ? (reinterpret_cast<const char*>(in_ptrs[peer]) + offset)
            : nullptr;
      },
      // Only non-root receives, from root.
      [&](int peer) { return !is_root && peer == root; },
      // Store the received slice.
      [&](int /*peer*/,
          const void* recv_ptr,
          std::size_t offset,
          std::size_t chunk_bytes) {
        std::memcpy(out_base + offset, recv_ptr, chunk_bytes);
      });
}

// reduce_scatter
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

  // Seed our output with our own contribution - our chunk destined for rank_.
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

// point-to-point send / recv
inline void TCCLEngine::p2p_send(
    int dst, const char* in, std::size_t nbytes) {
  TORCH_CHECK_WITH(
      DistBackendError,
      dst >= 0 && dst < size_ && dst != rank_,
      "TCCL p2p_send: invalid dst ", dst, " (rank ", rank_, ", size ", size_, ").");
  constexpr int D = kPipelineDepth;
  std::vector<ibv_wc> wcs(D + 2);
  // Depth-D pipeline: up to D pieces in flight, double-buffered staging, so the
  // per-piece memcpy overlaps the wire (PIPELINE=1 ran them serially).
  std::size_t next_off = 0, done = 0;
  int inflight = 0;
  std::size_t slen[D] = {0};
  auto fill = [&]() {
    for (int s = 0; s < D; ++s) {
      if (slen[s] == 0 && next_off < nbytes) {
        const std::size_t L = std::min(kChunkSize, nbytes - next_off);
        auto& buf = pipe_send_buffers_[static_cast<std::size_t>(dst) * D + s];
        std::memcpy(buf.data(), in + next_off, L);
        slen[s] = L;
        connections_[dst]->postSend(buf, L, tccl_wr::makeSend(dst, s));
        next_off += L;
        ++inflight;
      }
    }
  };
  fill();
  auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (done < nbytes) {
    int n = connections_[dst]->pollCq(static_cast<int>(wcs.size()), wcs.data());
    if (n == 0) {
      TORCH_CHECK_WITH(
          DistBackendError,
          std::chrono::steady_clock::now() <= deadline,
          "TCCL p2p_send: timed out after ", timeout_.count(), " ms sending ",
          done, "/", nbytes, " bytes to peer ", dst,
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
      const int s = tccl_wr::slot(wcs[i].wr_id);
      done += slen[s];
      slen[s] = 0;
      --inflight;
    }
    fill();
    deadline = std::chrono::steady_clock::now() + timeout_;
  }
}

inline void TCCLEngine::p2p_recv(
    int src, char* out, std::size_t nbytes) {
  TORCH_CHECK_WITH(
      DistBackendError,
      src >= 0 && src < size_ && src != rank_,
      "TCCL p2p_recv: invalid src ", src, " (rank ", rank_, ", size ", size_, ").");
  constexpr int D = kPipelineDepth;
  std::vector<ibv_wc> wcs(D + 2);
  // Depth-D pipeline: up to D recvs in flight; the copy-out of one piece overlaps
  // the wire arrival of the next. Per-slot offset recorded so completions (which a
  // single QP delivers in post order) copy to the right place regardless.
  std::size_t next_off = 0, done = 0;
  int inflight = 0;
  std::size_t soff[D] = {0}, slen[D] = {0};
  auto fill = [&]() {
    for (int s = 0; s < D; ++s) {
      if (slen[s] == 0 && next_off < nbytes) {
        const std::size_t L = std::min(kChunkSize, nbytes - next_off);
        auto& buf = pipe_recv_buffers_[static_cast<std::size_t>(src) * D + s];
        soff[s] = next_off;
        slen[s] = L;
        connections_[src]->postRecv(buf, L, tccl_wr::makeRecv(src, s));
        next_off += L;
        ++inflight;
      }
    }
  };
  fill();
  auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (done < nbytes) {
    int n = connections_[src]->pollCq(static_cast<int>(wcs.size()), wcs.data());
    if (n == 0) {
      TORCH_CHECK_WITH(
          DistBackendError,
          std::chrono::steady_clock::now() <= deadline,
          "TCCL p2p_recv: timed out after ", timeout_.count(), " ms receiving ",
          done, "/", nbytes, " bytes from peer ", src,
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
      const int s = tccl_wr::slot(wcs[i].wr_id);
      std::memcpy(
          out + soff[s],
          pipe_recv_buffers_[static_cast<std::size_t>(src) * D + s].data(),
          slen[s]);
      done += slen[s];
      slen[s] = 0;
      --inflight;
    }
    fill();
    deadline = std::chrono::steady_clock::now() + timeout_;
  }
}

// all-to-all (mesh / direct)
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

// all-to-all (ring / store-and-forward, equal splits)
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
    // Peel the last segment of the received block - destined for this rank,
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
