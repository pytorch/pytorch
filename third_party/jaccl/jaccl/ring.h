// Copyright © 2026 Apple Inc.

#pragma once

#include "jaccl/group.h"
#include "jaccl/rdma.h"
#include "jaccl/ring_impl.h"

namespace jaccl {

/**
 * The JACCL communication group for a ring where each node is connected to its
 * two neighboring nodes. It should be the highest bandwidth communication
 * group for large messages when many connections per peer are used.
 *
 * Like all JACCL groups it uses a side channel to exchange the necessary
 * information and then configure the connections to be ready for RDMA
 * operations.
 */
class RingGroup : public Group {
 public:
  RingGroup(
      int rank,
      int size,
      const std::vector<std::string>& left_devices,
      const std::vector<std::string>& right_devices,
      const std::string& coordinator_addr);

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_max(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_min(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_gather(const void* input, void* output, size_t n_bytes) override;

  void send(const void* input, size_t n_bytes, int dst) override;
  void recv(void* output, size_t n_bytes, int src) override;

  void barrier() override;

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  /**
   * Performs the connection initialization. Namely, after this call all
   * Connection objects should have a queue pair in RTS state and all buffers
   * should have been allocated.
   */
  void initialize();

  /**
   * Allocate all the buffers that we will use in the communication group.
   */
  void allocate_buffers();

  int rank_;
  int size_;
  int n_conns_;
  SideChannel side_channel_;
  std::vector<Connection> left_;
  std::vector<Connection> right_;
  std::vector<SharedBuffer> send_buffers_;
  std::vector<SharedBuffer> recv_buffers_;
  RingImpl ring_;
};

} // namespace jaccl
