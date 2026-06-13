// Copyright © 2026 Apple Inc.

#pragma once

#include <span>

#include "jaccl/rdma.h"

constexpr int RING_MAX_CONNS = 4;

namespace jaccl {

class RingImpl {
 public:
  RingImpl(
      int rank,
      int size,
      std::vector<Connection>& left,
      std::vector<Connection>& right,
      std::vector<SharedBuffer>& send_buffers,
      std::vector<SharedBuffer>& recv_buffers)
      : rank_(rank),
        size_(size),
        n_conns_(left.size()),
        left_(left),
        right_(right),
        send_buffers_(send_buffers),
        recv_buffers_(recv_buffers) {}

  RingImpl(
      int rank,
      int size,
      Connection* left_begin,
      Connection* right_begin,
      size_t n_conns,
      std::vector<SharedBuffer>& send_buffers,
      std::vector<SharedBuffer>& recv_buffers)
      : rank_(rank),
        size_(size),
        n_conns_(n_conns),
        left_(left_begin, n_conns),
        right_(right_begin, n_conns),
        send_buffers_(send_buffers),
        recv_buffers_(recv_buffers) {}

  RingImpl() : rank_(0), size_(1), n_conns_(0) {}

  template <int MAX_DIR, typename T, typename ReduceOp>
  void all_reduce(
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      int n_wires,
      ReduceOp reduce_op) {
    // If not inplace all reduce then copy the input to the output first
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, size * sizeof(T));
    }

    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * RING_MAX_CONNS * 2 * MAX_DIR;
    int64_t chunk_size = (size + size_ - 1) / size_;
    int64_t size_per_wire =
        (chunk_size + (MAX_DIR * n_wires) - 1) / (MAX_DIR * n_wires);
    auto [sz, N] = buffer_size_from_message(size_per_wire * sizeof(T));
    N /= sizeof(T);
    int64_t n_steps = (size_per_wire + N - 1) / N;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t chunk_multiple_size = size_ * chunk_size;
    int64_t send_offset[MAX_DIR];
    int64_t recv_offset[MAX_DIR];
    int64_t send_limits[MAX_DIR];
    int64_t recv_limits[MAX_DIR];
    int send_count[MAX_DIR * RING_MAX_CONNS] = {0};
    int recv_count[MAX_DIR * RING_MAX_CONNS] = {0};
    send_offset[0] = rank_ * chunk_size;
    recv_offset[0] = ((rank_ + size_ - 1) % size_) * chunk_size;
    if constexpr (MAX_DIR == 2) {
      send_offset[1] = rank_ * chunk_size;
      recv_offset[1] = ((rank_ + 1) % size_) * chunk_size;
      send_limits[0] = std::min(
          n_wires * size_per_wire, std::max<int64_t>(0, size - send_offset[0]));
      send_limits[1] =
          std::min(chunk_size, std::max<int64_t>(0, size - send_offset[1]));
      recv_limits[0] = std::min(
          n_wires * size_per_wire, std::max<int64_t>(0, size - recv_offset[0]));
      recv_limits[1] =
          std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[1]));
    } else {
      send_limits[0] =
          std::min(chunk_size, std::max<int64_t>(0, size - send_offset[0]));
      recv_limits[0] =
          std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[0]));
    }

    // First reduce scatter
    //
    // Possible perf improvement by not syncing at every step but running ahead
    // as needed.
    for (int k = 0; k < size_ - 1; k++) {
      // Prefill the pipeline
      int buff = 0;
      while (buff < n_steps && buff < PIPELINE) {
        post_recv_all<MAX_DIR>(sz, buff, n_wires);
        for (int lr = 0; lr < MAX_DIR; lr++) {
          for (int lw = 0; lw < n_wires; lw++) {
            int64_t offset = lw * N +
                send_count[lr * RING_MAX_CONNS + lw] * n_wires * N +
                lr * n_wires * size_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, send_limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<T>());
            send_count[lr * RING_MAX_CONNS + lw]++;
          }
        }
        post_send_all<MAX_DIR>(sz, buff, n_wires);

        buff++;
        in_flight += 2 * MAX_DIR * n_wires;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = poll(left_, right_, WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int wire = wc[i].wr_id & 0xff;
          int lr = wire / RING_MAX_CONNS;
          int lw = wire % RING_MAX_CONNS;

          in_flight--;

          if (work_type == SEND_WR && send_count[wire] < n_steps) {
            int64_t offset = lw * N + send_count[wire] * n_wires * N +
                lr * n_wires * size_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, send_limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<T>());
            send_to(sz, buff, lr, lw);
            in_flight++;
            send_count[wire]++;
          }

          else if (work_type == RECV_WR) {
            int64_t offset = lw * N + recv_count[wire] * n_wires * N +
                lr * n_wires * size_per_wire;
            reduce_op(
                recv_buffer(sz, buff, lr, lw).begin<T>(),
                out_ptr + recv_offset[lr] + offset,
                std::max<int64_t>(0, std::min(N, recv_limits[lr] - offset)));
            recv_count[wire]++;
            if (recv_count[wire] + (PIPELINE - 1) < n_steps) {
              recv_from(sz, buff, lr, lw);
              in_flight++;
            }
          }
        }
      }

      send_offset[0] = (send_offset[0] + chunk_multiple_size - chunk_size) %
          chunk_multiple_size;
      recv_offset[0] = (recv_offset[0] + chunk_multiple_size - chunk_size) %
          chunk_multiple_size;
      if constexpr (MAX_DIR == 2) {
        send_offset[1] = (send_offset[1] + chunk_size) % chunk_multiple_size;
        recv_offset[1] = (recv_offset[1] + chunk_size) % chunk_multiple_size;
        send_limits[0] = std::min(
            n_wires * size_per_wire,
            std::max<int64_t>(0, size - send_offset[0]));
        send_limits[1] =
            std::min(chunk_size, std::max<int64_t>(0, size - send_offset[1]));
        recv_limits[0] = std::min(
            n_wires * size_per_wire,
            std::max<int64_t>(0, size - recv_offset[0]));
        recv_limits[1] =
            std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[1]));
      } else {
        send_limits[0] =
            std::min(chunk_size, std::max<int64_t>(0, size - send_offset[0]));
        recv_limits[0] =
            std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[0]));
      }
      for (int i = 0; i < MAX_DIR * RING_MAX_CONNS; i++) {
        send_count[i] = recv_count[i] = 0;
      }
    }

    // Secondly all gather
    //
    // The offsets are correct from the scatter reduce
    for (int k = 0; k < size_ - 1; k++) {
      // Prefill the pipeline
      int buff = 0;
      while (buff < n_steps && buff < PIPELINE) {
        post_recv_all<MAX_DIR>(sz, buff, n_wires);
        for (int lr = 0; lr < MAX_DIR; lr++) {
          for (int lw = 0; lw < n_wires; lw++) {
            int64_t offset = lw * N +
                send_count[lr * RING_MAX_CONNS + lw] * n_wires * N +
                lr * n_wires * size_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, send_limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<T>());
            send_count[lr * RING_MAX_CONNS + lw]++;
          }
        }
        post_send_all<MAX_DIR>(sz, buff, n_wires);

        buff++;
        in_flight += 2 * MAX_DIR * n_wires;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = poll(left_, right_, WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int wire = wc[i].wr_id & 0xff;
          int lr = wire / RING_MAX_CONNS;
          int lw = wire % RING_MAX_CONNS;

          in_flight--;

          if (work_type == SEND_WR && send_count[wire] < n_steps) {
            int64_t offset = lw * N + send_count[wire] * n_wires * N +
                lr * n_wires * size_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, send_limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<T>());
            send_to(sz, buff, lr, lw);
            in_flight++;
            send_count[wire]++;
          }

          else if (work_type == RECV_WR) {
            int64_t offset = lw * N + recv_count[wire] * n_wires * N +
                lr * n_wires * size_per_wire;
            std::copy(
                recv_buffer(sz, buff, lr, lw).begin<T>(),
                recv_buffer(sz, buff, lr, lw).begin<T>() +
                    std::max<int64_t>(0, std::min(N, recv_limits[lr] - offset)),
                out_ptr + recv_offset[lr] + offset);
            recv_count[wire]++;
            if (recv_count[wire] + (PIPELINE - 1) < n_steps) {
              recv_from(sz, buff, lr, lw);
              in_flight++;
            }
          }
        }
      }

      send_offset[0] = (send_offset[0] + chunk_multiple_size - chunk_size) %
          chunk_multiple_size;
      recv_offset[0] = (recv_offset[0] + chunk_multiple_size - chunk_size) %
          chunk_multiple_size;
      if constexpr (MAX_DIR == 2) {
        send_offset[1] = (send_offset[1] + chunk_size) % chunk_multiple_size;
        recv_offset[1] = (recv_offset[1] + chunk_size) % chunk_multiple_size;
        send_limits[0] = std::min(
            n_wires * size_per_wire,
            std::max<int64_t>(0, size - send_offset[0]));
        send_limits[1] =
            std::min(chunk_size, std::max<int64_t>(0, size - send_offset[1]));
        recv_limits[0] = std::min(
            n_wires * size_per_wire,
            std::max<int64_t>(0, size - recv_offset[0]));
        recv_limits[1] =
            std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[1]));
      } else {
        send_limits[0] =
            std::min(chunk_size, std::max<int64_t>(0, size - send_offset[0]));
        recv_limits[0] =
            std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[0]));
      }
      for (int i = 0; i < MAX_DIR * RING_MAX_CONNS; i++) {
        send_count[i] = recv_count[i] = 0;
      }
    }
  }

  void
  all_gather(const char* in_ptr, char* out_ptr, int64_t n_bytes, int n_wires) {
    // Copy our data to the appropriate place
    std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * RING_MAX_CONNS * 2 * 2;
    size_t n_bytes_per_wire = (n_bytes + (2 * n_wires) - 1) / (2 * n_wires);
    size_t out_bytes = n_bytes * size_;
    auto [sz, N] = buffer_size_from_message(n_bytes_per_wire);
    int n_steps = (n_bytes_per_wire + N - 1) / N;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t send_offset[2];
    int64_t recv_offset[2];
    int64_t limits[2];
    int send_count[2 * RING_MAX_CONNS] = {0};
    int recv_count[2 * RING_MAX_CONNS] = {0};
    send_offset[0] = send_offset[1] = rank_ * n_bytes;
    recv_offset[0] = ((rank_ + size_ - 1) % size_) * n_bytes;
    recv_offset[1] = ((rank_ + 1) % size_) * n_bytes;
    limits[0] = n_wires * n_bytes_per_wire;
    limits[1] = n_bytes;

    // Possible perf improvement by not syncing at every step but running ahead
    // as needed.
    for (int k = 0; k < size_ - 1; k++) {
      // Prefill the pipeline
      int buff = 0;
      while (buff < n_steps && buff < PIPELINE) {
        post_recv_all(sz, buff);
        for (int lr = 0; lr < 2; lr++) {
          for (int lw = 0; lw < n_wires; lw++) {
            int64_t offset = lw * N +
                send_count[lr * RING_MAX_CONNS + lw] * n_wires * N +
                lr * n_wires * n_bytes_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<char>());
            send_count[lr * RING_MAX_CONNS + lw]++;
          }
        }
        post_send_all(sz, buff);

        buff++;
        in_flight += 2 * 2 * n_wires;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = poll(left_, right_, WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int wire = wc[i].wr_id & 0xff;
          int lr = wire / RING_MAX_CONNS;
          int lw = wire % RING_MAX_CONNS;

          in_flight--;

          if (work_type == SEND_WR && send_count[wire] < n_steps) {
            int64_t offset = lw * N + send_count[wire] * n_wires * N +
                lr * n_wires * n_bytes_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<char>());
            send_to(sz, buff, lr, lw);
            in_flight++;
            send_count[wire]++;
          }

          else if (work_type == RECV_WR) {
            int64_t offset = lw * N + recv_count[wire] * n_wires * N +
                lr * n_wires * n_bytes_per_wire;
            std::copy(
                recv_buffer(sz, buff, lr, lw).begin<char>(),
                recv_buffer(sz, buff, lr, lw).begin<char>() +
                    std::max<int64_t>(0, std::min(N, limits[lr] - offset)),
                out_ptr + recv_offset[lr] + offset);
            recv_count[wire]++;
            if (recv_count[wire] + (PIPELINE - 1) < n_steps) {
              recv_from(sz, buff, lr, lw);
              in_flight++;
            }
          }
        }
      }

      send_offset[0] = (send_offset[0] + out_bytes - n_bytes) % out_bytes;
      recv_offset[0] = (recv_offset[0] + out_bytes - n_bytes) % out_bytes;
      send_offset[1] = (send_offset[1] + n_bytes) % out_bytes;
      recv_offset[1] = (recv_offset[1] + n_bytes) % out_bytes;
      for (int i = 0; i < 2 * RING_MAX_CONNS; i++) {
        send_count[i] = recv_count[i] = 0;
      }
    }
  }

  void send(const char* in_ptr, int64_t n_bytes, int dst, int n_wires) {
    int left = (rank_ + size_ - 1) % size_;

    // In the case that size_ == 2 then left == right so we bias send towards
    // left and recv towards right so that the selections will be correct for
    // the 2 node case.
    auto& conns = (dst == left) ? left_ : right_;
    int dir = dst == left;

    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * RING_MAX_CONNS;

    int64_t bytes_per_wire = (n_bytes + n_wires - 1) / n_wires;
    auto [sz, N] = buffer_size_from_message(bytes_per_wire);

    int in_flight = 0;
    int64_t read_offset[RING_MAX_CONNS];
    int64_t limits[RING_MAX_CONNS];
    for (int lw = 0; lw < n_wires; lw++) {
      read_offset[lw] = std::min(lw * bytes_per_wire, n_bytes);
      limits[lw] = std::min((lw + 1) * bytes_per_wire, n_bytes);
    }

    // Prefill the pipeline
    for (int lw = 0; lw < n_wires; lw++) {
      int buff = 0;
      while (read_offset[lw] < limits[lw] && buff < PIPELINE) {
        std::copy(
            in_ptr + read_offset[lw],
            in_ptr + std::min(read_offset[lw] + N, limits[lw]),
            send_buffer(sz, buff, dir, lw).begin<char>());
        send_to(sz, buff, dir, lw);

        buff++;
        read_offset[lw] += N;
        in_flight++;
      }
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a send was completed and we have more data to send then go ahead
      // and send them.
      ibv_wc wc[WC_NUM];
      int n = poll(conns, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int wire = wc[i].wr_id & 0xff;
        int lw = wire % RING_MAX_CONNS;

        in_flight--;

        if (read_offset[lw] < limits[lw]) {
          std::copy(
              in_ptr + read_offset[lw],
              in_ptr + std::min(read_offset[lw] + N, limits[lw]),
              send_buffer(sz, buff, dir, lw).begin<char>());
          send_to(sz, buff, dir, lw);

          read_offset[lw] += N;
          in_flight++;
        }
      }
    }
  }

  void recv(char* out_ptr, int64_t n_bytes, int src, int n_wires) {
    int right = (rank_ + 1) % size_;

    // In the case that size_ == 2 then left == right so we bias send towards
    // left and recv towards right so that the selections will be correct for
    // the 2 node case.
    auto& conns = (src == right) ? right_ : left_;
    int dir = src == right;

    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * RING_MAX_CONNS;

    int64_t bytes_per_wire = (n_bytes + n_wires - 1) / n_wires;
    auto [sz, N] = buffer_size_from_message(bytes_per_wire);

    int in_flight = 0;
    int64_t write_offset[RING_MAX_CONNS];
    int64_t limits[RING_MAX_CONNS];
    for (int lw = 0; lw < n_wires; lw++) {
      write_offset[lw] = std::min(lw * bytes_per_wire, n_bytes);
      limits[lw] = std::min((lw + 1) * bytes_per_wire, n_bytes);
    }

    // Prefill the pipeline
    for (int lw = 0; lw < n_wires; lw++) {
      int buff = 0;
      while (N * buff < limits[lw] && buff < PIPELINE) {
        recv_from(sz, buff, dir, lw);

        buff++;
        in_flight++;
      }
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a recv was completed copy it to the output and if we have more
      // data to fetch post another recv.
      ibv_wc wc[WC_NUM];
      int n = poll(conns, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int wire = wc[i].wr_id & 0xff;
        int lw = wire % RING_MAX_CONNS;

        in_flight--;

        std::copy(
            recv_buffer(sz, buff, dir, lw).begin<char>(),
            recv_buffer(sz, buff, dir, lw).begin<char>() +
                std::max<int64_t>(
                    0, std::min<int64_t>(limits[lw] - write_offset[lw], N)),
            out_ptr + write_offset[lw]);
        write_offset[lw] += N;

        if (write_offset[lw] + (PIPELINE - 1) * N < limits[lw]) {
          recv_from(sz, buff, dir, lw);

          in_flight++;
        }
      }
    }
  }

 private:
  void send_to(int sz, int buff, int left_right, int wire) {
    if (left_right) {
      left_[wire].post_send(
          send_buffer_left(sz, buff, wire),
          SEND_WR << 16 | buff << 8 | (RING_MAX_CONNS + wire));
    } else {
      right_[wire].post_send(
          send_buffer_right(sz, buff, wire), SEND_WR << 16 | buff << 8 | wire);
    }
  }

  void recv_from(int sz, int buff, int left_right, int wire) {
    if (left_right) {
      right_[wire].post_recv(
          recv_buffer_right(sz, buff, wire),
          RECV_WR << 16 | buff << 8 | (RING_MAX_CONNS + wire));
    } else {
      left_[wire].post_recv(
          recv_buffer_left(sz, buff, wire), RECV_WR << 16 | buff << 8 | wire);
    }
  }

  SharedBuffer& send_buffer_right(int sz, int buff, int wire) {
    return send_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 + wire];
  }

  SharedBuffer& send_buffer_left(int sz, int buff, int wire) {
    return send_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 + n_conns_ +
         wire];
  }

  SharedBuffer& send_buffer(int sz, int buff, int left_right, int wire) {
    return send_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 +
         left_right * n_conns_ + wire];
  }

  SharedBuffer& recv_buffer_left(int sz, int buff, int wire) {
    return recv_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 + wire];
  }

  SharedBuffer& recv_buffer_right(int sz, int buff, int wire) {
    return recv_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 + n_conns_ +
         wire];
  }

  SharedBuffer& recv_buffer(int sz, int buff, int left_right, int wire) {
    return recv_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 +
         left_right * n_conns_ + wire];
  }

  template <int MAX_DIR>
  void post_recv_all(int sz, int buff, int n_wires) {
    for (int lr = 0; lr < MAX_DIR; lr++) {
      for (int lw = 0; lw < n_wires; lw++) {
        recv_from(sz, buff, lr, lw);
      }
    }
  }

  void post_recv_all(int sz, int buff) {
    post_recv_all<2>(sz, buff, n_conns_);
  }

  template <int MAX_DIR>
  void post_send_all(int sz, int buff, int n_wires) {
    for (int lr = 0; lr < MAX_DIR; lr++) {
      for (int lw = 0; lw < n_wires; lw++) {
        send_to(sz, buff, lr, lw);
      }
    }
  }

  void post_send_all(int sz, int buff) {
    post_send_all<2>(sz, buff, n_conns_);
  }

  int rank_;
  int size_;
  int n_conns_;
  std::span<Connection> left_;
  std::span<Connection> right_;
  std::span<SharedBuffer> send_buffers_;
  std::span<SharedBuffer> recv_buffers_;
};

} // namespace jaccl
