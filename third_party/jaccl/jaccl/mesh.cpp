// Copyright © 2026 Apple Inc.

#include "jaccl/mesh.h"
#include "jaccl/reduction_ops.h"
#include "jaccl/types.h"

namespace jaccl {

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::string>& device_names,
    const std::string& coordinator_addr)
    : rank_(rank),
      size_(device_names.size()),
      side_channel_(rank_, size_, coordinator_addr.c_str()),
      connections_(create_connections(device_names)) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.barrier();

  // Create the mesh implementation object
  mesh_ = MeshImpl(rank_, size_, connections_, buffers_);
  ring_ = RingImpl(
      rank_,
      size_,
      &connections_[(rank_ + size_ - 1) % size_],
      &connections_[(rank_ + 1) % size_],
      1,
      ring_send_buffers_,
      ring_recv_buffers_);
}

void MeshGroup::initialize() {
  // Create the queue pairs
  for (auto& conn : connections_) {
    if (conn.ctx == nullptr) {
      continue;
    }
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }

  allocate_buffers();

  // First init all connections
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_init();
  }

  // Gather the information to be exchanged, this also serves as a barrier
  // so that all peers have initialized their connections before attempting
  // to transition to RTS.
  std::vector<Destination> info;
  for (auto& conn : connections_) {
    info.emplace_back(conn.info());
  }
  auto all_infos = side_channel_.all_gather(info);

  // Transition queue pairs to RTS
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    auto peer_info = all_infos[peer][rank_];
    connections_[peer].queue_pair_rtr(peer_info);
    connections_[peer].queue_pair_rts();
  }
}

void MeshGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  buffers_.clear();
  ring_send_buffers_.clear();
  ring_recv_buffers_.clear();

  // Allocate the memory
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      // Mesh buffers
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
      // Ring buffers (1 for each direction)
      for (int j = 0; j < 2; j++) {
        ring_send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        ring_recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      // Mesh buffers
      for (int j = 0; j < size_; j++) {
        if (j == rank_) {
          // This is our send buffer so register it with all pds so we can
          // send it to all connected devices.
          for (auto& conn : connections_) {
            if (conn.ctx != nullptr) {
              buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                  .register_to_protection_domain(conn.protection_domain);
            }
          }
        } else {
          // This is the recv buffer from rank j so register it to rank j's
          // protection domain.
          buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
              .register_to_protection_domain(connections_[j].protection_domain);
        }
      }

      // Ring buffers (see ring group for the logic below)
      int left = (rank_ + size_ - 1) % size_;
      int right = (rank_ + 1) % size_;
      // We register send buffers to both the right and the left.
      ring_send_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 0]
          .register_to_protection_domain(connections_[right].protection_domain);
      ring_recv_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 0]
          .register_to_protection_domain(connections_[left].protection_domain);
      ring_send_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 1]
          .register_to_protection_domain(connections_[left].protection_domain);
      ring_recv_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 1]
          .register_to_protection_domain(connections_[right].protection_domain);
    }
  }
}

void MeshGroup::all_sum(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, SumOp<T>{});
  });
}

void MeshGroup::all_max(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, MaxOp<T>{});
  });
}

void MeshGroup::all_min(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, MinOp<T>{});
  });
}

void MeshGroup::all_gather(const void* input, void* output, size_t n_bytes) {
  mesh_.all_gather(
      static_cast<const char*>(input), static_cast<char*>(output), n_bytes);
}

void MeshGroup::send(const void* input, size_t n_bytes, int dst) {
  mesh_.send(static_cast<const char*>(input), n_bytes, dst);
}

void MeshGroup::recv(void* output, size_t n_bytes, int src) {
  mesh_.recv(static_cast<char*>(output), n_bytes, src);
}

void MeshGroup::barrier() {
  uint8_t b = 0;
  all_sum(&b, &b, sizeof(b), Dtype::UInt8);
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
    const void* input,
    void* output,
    size_t n_bytes,
    ReduceOp reduce_op) {
  auto in_ptr = static_cast<const T*>(input);
  auto out_ptr = static_cast<T*>(output);
  int64_t count = n_bytes / sizeof(T);
  if (size_ > 2 &&
      ((std::is_same_v<T, bfloat16_t> && count > 256 * 1024) ||
       count >= 8 * 1024 * 1024 / static_cast<int64_t>(sizeof(T)))) {
    ring_.all_reduce<2>(in_ptr, out_ptr, count, 1, reduce_op);
  } else {
    mesh_.all_reduce(in_ptr, out_ptr, count, reduce_op);
  }
}

} // namespace jaccl
