// Copyright © 2026 Apple Inc.

#pragma once

#include <memory>
#include <span>

#include "jaccl/rdma.h"

constexpr int MESH_MAX_PEERS = 8;
constexpr int MESH_PIPELINE = 2;
constexpr int64_t MAX_BUFFER_SIZE = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));

namespace jaccl {

class MeshImpl {
 public:
  MeshImpl(
      int rank,
      int size,
      std::vector<Connection>& conns,
      std::vector<SharedBuffer>& buffers)
      : rank_(rank),
        size_(size),
        connections_(conns),
        buffers_(buffers),
        staging_mem_(
            std::make_unique<char[]>(MESH_PIPELINE * MAX_BUFFER_SIZE)) {}

  MeshImpl() : rank_(0), size_(1) {}

  template <typename T, typename ReduceOp>
  void all_reduce(const T* in, T* out, int64_t size, ReduceOp reduce_op) {
    // Fully connected all reduce with deterministic reduction order.
    //
    // We copy rank 0's data to the output buffer and then we reduce every
    // subsequent rank in-place in the output.
    //
    // Our own data is copied to a staging buffer to ensure we can reduce it in
    // the output when needed.

    auto [sz, buffer_size] = buffer_size_from_message(size * sizeof(T));
    int64_t N = buffer_size / sizeof(T);
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MESH_MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(size);
    int num_peers = size_ - 1;

    // A helper for convenient access to the staging buffer.
    auto local_staging = [&](int buff) -> T* {
      return reinterpret_cast<T*>(staging_mem_.get() + buff * MAX_BUFFER_SIZE);
    };

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int recv_end[MESH_MAX_PEERS] = {0};
    int reduce_chunk = 0;
    int reduce_rank = 0;

    // Total number of chunks
    int64_t total_chunks = (total + N - 1) / N;

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {
      post_recv_all(sz, buff);

      // Copy the local data to send buffer and staging buffer
      int64_t elems = std::min(N, total - read_offset);
      std::copy(
          in + read_offset, in + read_offset + elems, local_staging(buff));
      std::copy(
          in + read_offset,
          in + read_offset + elems,
          send_buffer(sz, buff).begin<T>());
      recv_end[rank_]++;
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop
    while (reduce_chunk < total_chunks) {
      // Poll the hardware for completions.
      //
      // If a send was completed mark how many completions we have received
      // for that buffer. If we have sent the buffer to all peers we can
      // reuse the buffer so copy the next chunk of data and send it to all.
      // Also copy the next chunk into the staging area and advance our
      // completed "receives".
      //
      // If a receive is completed then advance the pointer of completed
      // receives.
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            int64_t elems = std::min(N, total - read_offset);
            std::copy(
                in + read_offset,
                in + read_offset + elems,
                local_staging(buff));
            std::copy(
                in + read_offset,
                in + read_offset + elems,
                send_buffer(sz, buff).begin<T>());
            recv_end[rank_]++;
            post_send_all(sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        else if (work_type == RECV_WR) {
          recv_end[rank]++;
        }
      }

      // Process the received chunks in order.
      //
      // Rank 0 is always copied as is. Our rank is always read from the
      // staging area.
      while (reduce_chunk < total_chunks) {
        // w is our write location so break if it is ahead of the read location.
        int64_t w = static_cast<int64_t>(reduce_chunk) * N;
        if (w >= read_offset) {
          break;
        }
        // We want to reduce the 'reduce_chunk' chunk but it hasn't arrived
        // yet.
        if (recv_end[reduce_rank] <= reduce_chunk) {
          break;
        }
        int b = reduce_chunk % PIPELINE;
        int64_t elems = std::min(N, total - w);

        // Data is read from the staging area
        if (reduce_rank == rank_) {
          if (reduce_rank == 0) {
            std::copy_n(local_staging(b), elems, out + w);
          } else {
            reduce_op(local_staging(b), out + w, elems);
          }
        }

        // Data is read from the recv buffers
        else {
          if (reduce_rank == 0) {
            std::copy_n(
                recv_buffer(sz, b, reduce_rank).begin<T>(), elems, out + w);
          } else {
            reduce_op(
                recv_buffer(sz, b, reduce_rank).begin<T>(), out + w, elems);
          }

          // Check if we need to post another receive
          int64_t next_chunk = static_cast<int64_t>(reduce_chunk) + PIPELINE;
          if (next_chunk < total_chunks) {
            recv_from(sz, reduce_rank, b);
            in_flight++;
          }
        }

        // Means we processed that chunk so move to the next one
        reduce_rank++;
        if (reduce_rank >= size_) {
          reduce_rank = 0;
          reduce_chunk++;
        }
      }
    }

    // Drain remaining in-flight completions (outstanding sends).
    while (in_flight > 0) {
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      in_flight -= n;
    }
  }

  void all_gather(const char* in_ptr, char* out_ptr, int64_t n_bytes) {
    // Copy our data to the appropriate place
    std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

    // Fully connected all gather
    char* data = out_ptr;
    char* our_data = out_ptr + rank_ * n_bytes;
    auto [sz, N] = buffer_size_from_message(n_bytes);
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MESH_MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(n_bytes);
    int num_peers = size_ - 1;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int write_offset[MESH_MAX_PEERS] = {0};

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {
      post_recv_all(sz, buff);
      std::copy(
          our_data + read_offset,
          our_data + std::min(read_offset + N, total),
          send_buffer(sz, buff).begin<char>());
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    while (in_flight > 0) {
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        // Send completed. If all sends completed then send the next chunk.
        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            std::copy(
                our_data + read_offset,
                our_data + std::min(read_offset + N, total),
                send_buffer(sz, buff).begin<char>());
            post_send_all(sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        // Recv completed. If we have more chunks then post another recv.
        else if (work_type == RECV_WR) {
          std::copy(
              recv_buffer(sz, buff, rank).begin<char>(),
              recv_buffer(sz, buff, rank).begin<char>() +
                  std::min(N, total - write_offset[rank]),
              data + rank * n_bytes + write_offset[rank]);
          write_offset[rank] += N;
          if (write_offset[rank] + N * (PIPELINE - 1) < total) {
            recv_from(sz, rank, buff);
            in_flight++;
          }
        }
      }
    }
  }

  void send(const char* in_ptr, int64_t n_bytes, int dst) {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t read_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < n_bytes && buff < PIPELINE) {
      std::copy(
          in_ptr + read_offset,
          in_ptr + std::min(read_offset + N, n_bytes),
          send_buffer(sz, buff).begin<char>());
      send_to(sz, dst, buff);

      buff++;
      read_offset += N;
      in_flight++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a send was completed and we have more data to send then go ahead
      // and send them.
      ibv_wc wc[WC_NUM];
      int n = connections_[dst].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (read_offset < n_bytes) {
          std::copy(
              in_ptr + read_offset,
              in_ptr + std::min(read_offset + N, n_bytes),
              send_buffer(sz, buff).begin<char>());
          send_to(sz, dst, buff);

          read_offset += N;
          in_flight++;
        }
      }
    }
  }

  void recv(char* out_ptr, int64_t n_bytes, int src) {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t write_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (N * buff < n_bytes && buff < PIPELINE) {
      recv_from(sz, src, buff);

      in_flight++;
      buff++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a recv was completed copy it to the output and if we have more
      // data to fetch post another recv.
      ibv_wc wc[WC_NUM];
      int n = connections_[src].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        std::copy(
            recv_buffer(sz, buff, src).begin<char>(),
            recv_buffer(sz, buff, src).begin<char>() +
                std::min(n_bytes - write_offset, static_cast<int64_t>(N)),
            out_ptr + write_offset);
        write_offset += N;

        if (write_offset + (PIPELINE - 1) * N < n_bytes) {
          recv_from(sz, src, buff);

          in_flight++;
        }
      }
    }
  }

 private:
  void send_to(int sz, int rank, int buff) {
    connections_[rank].post_send(
        send_buffer(sz, buff), SEND_WR << 16 | buff << 8 | rank);
  }

  void recv_from(int sz, int rank, int buff) {
    connections_[rank].post_recv(
        recv_buffer(sz, buff, rank), RECV_WR << 16 | buff << 8 | rank);
  }

  SharedBuffer& send_buffer(int sz, int buff) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank_];
  }

  SharedBuffer& recv_buffer(int sz, int buff, int rank) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank];
  }

  void post_send_all(int sz, int buff) {
    auto& b = send_buffer(sz, buff);
    int wr_id = SEND_WR << 16 | buff << 8;
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      connections_[i].post_send(b, wr_id | i);
    }
  }

  void post_recv_all(int sz, int buff) {
    int b = sz * NUM_BUFFERS * size_ + buff * size_;
    int wr_id = RECV_WR << 16 | buff << 8;
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      connections_[i].post_recv(buffers_[b + i], wr_id | i);
    }
  }

  int rank_;
  int size_;
  std::span<Connection> connections_;
  std::span<SharedBuffer> buffers_;
  std::unique_ptr<char[]> staging_mem_;
};

} // namespace jaccl
