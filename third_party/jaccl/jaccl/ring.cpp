// Copyright © 2026 Apple Inc.

#include "jaccl/ring.h"
#include "jaccl/reduction_ops.h"
#include "jaccl/types.h"

namespace jaccl {

RingGroup::RingGroup(
    int rank,
    int size,
    const std::vector<std::string>& left_devices,
    const std::vector<std::string>& right_devices,
    const std::string& coordinator_addr)
    : rank_(rank),
      size_(size),
      n_conns_(left_devices.size()),
      side_channel_(rank_, size_, coordinator_addr.c_str()),
      left_(create_connections(left_devices)),
      right_(create_connections(right_devices)) {
  if (left_.size() > RING_MAX_CONNS || right_.size() > RING_MAX_CONNS) {
    std::ostringstream msg;
    msg << "[jaccl] Up to " << RING_MAX_CONNS << " per direction supported but "
        << left_.size() << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing.
  side_channel_.barrier();

  // Create the ring implementation object
  ring_ = RingImpl(rank_, size_, left_, right_, send_buffers_, recv_buffers_);
}

void RingGroup::initialize() {
  // Create the queue pairs
  for (auto& conn : left_) {
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }
  for (auto& conn : right_) {
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }

  // Allocate the buffers
  allocate_buffers();

  // Initialize the conections
  for (auto& conn : left_) {
    conn.queue_pair_init();
  }
  for (auto& conn : right_) {
    conn.queue_pair_init();
  }

  // Gather the information to be exchanged, this also serves as a barrier
  // so that all peers have initialized their connections before attempting
  // to transition to RTS.
  std::vector<Destination> left_info;
  for (auto& conn : left_) {
    left_info.emplace_back(conn.info());
  }
  std::vector<Destination> right_info;
  for (auto& conn : right_) {
    right_info.emplace_back(conn.info());
  }
  auto all_left_infos = side_channel_.all_gather(left_info);
  auto all_right_infos = side_channel_.all_gather(right_info);

  // Transition queue pairs to RTS
  int left_peer = (rank_ + size_ - 1) % size_;
  for (int i = 0; i < left_.size(); i++) {
    auto peer_info = all_right_infos[left_peer][i];
    left_[i].queue_pair_rtr(peer_info);
    left_[i].queue_pair_rts();
  }
  int right_peer = (rank_ + 1) % size_;
  for (int i = 0; i < right_.size(); i++) {
    auto peer_info = all_left_infos[right_peer][i];
    right_[i].queue_pair_rtr(peer_info);
    right_[i].queue_pair_rts();
  }
}

void RingGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  send_buffers_.clear();
  recv_buffers_.clear();

  // Allocate the memory
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < n_conns_ * 2; j++) {
        send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  // Register the buffers with the corresponding connections
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < n_conns_ * 2; j++) {
        int wire = j % n_conns_;
        int lr = j / n_conns_;
        if (lr) {
          send_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
              .register_to_protection_domain(left_[wire].protection_domain);
          recv_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
              .register_to_protection_domain(right_[wire].protection_domain);
        } else {
          send_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
              .register_to_protection_domain(right_[wire].protection_domain);
          recv_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
              .register_to_protection_domain(left_[wire].protection_domain);
        }
      }
    }
  }
}

void RingGroup::all_sum(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, SumOp<T>{});
  });
}

void RingGroup::all_max(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, MaxOp<T>{});
  });
}

void RingGroup::all_min(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, MinOp<T>{});
  });
}

void RingGroup::all_gather(const void* input, void* output, size_t n_bytes) {
  ring_.all_gather(
      static_cast<const char*>(input),
      static_cast<char*>(output),
      n_bytes,
      n_conns_);
}

void RingGroup::send(const void* input, size_t n_bytes, int dst) {
  int right = (rank_ + 1) % size_;
  int left = (rank_ + size_ - 1) % size_;
  if (dst != right && dst != left) {
    std::ostringstream msg;
    msg << "[jaccl] In ring mode send is only supported to direct neighbors "
        << "but tried to send to " << dst << " from " << rank_ << std::endl;
    throw std::runtime_error(msg.str());
  }
  ring_.send(static_cast<const char*>(input), n_bytes, dst, n_conns_);
}

void RingGroup::recv(void* output, size_t n_bytes, int src) {
  int right = (rank_ + 1) % size_;
  int left = (rank_ + size_ - 1) % size_;
  if (src != right && src != left) {
    std::ostringstream msg;
    msg << "[jaccl] In ring mode recv is only supported to direct neighbors "
        << "but tried to recv from " << src << " to " << rank_ << std::endl;
    throw std::runtime_error(msg.str());
  }
  ring_.recv(static_cast<char*>(output), n_bytes, src, n_conns_);
}

void RingGroup::barrier() {
  uint8_t b = 0;
  all_sum(&b, &b, sizeof(b), Dtype::UInt8);
}

template <typename T, typename ReduceOp>
void RingGroup::all_reduce(
    const void* input,
    void* output,
    size_t n_bytes,
    ReduceOp reduce_op) {
  auto in_ptr = static_cast<const T*>(input);
  auto out_ptr = static_cast<T*>(output);
  int64_t count = n_bytes / sizeof(T);
  if (count < size_ * 2 * n_conns_) {
    ring_.all_reduce<1, T, ReduceOp>(in_ptr, out_ptr, count, 1, reduce_op);
    return;
  }

  if (n_bytes <= 65536) {
    ring_.all_reduce<2, T, ReduceOp>(in_ptr, out_ptr, count, 1, reduce_op);
    return;
  }

  ring_.all_reduce<2, T, ReduceOp>(in_ptr, out_ptr, count, n_conns_, reduce_op);
}

} // namespace jaccl
