// Copyright © 2025 Apple Inc.

#pragma once

#include <infiniband/verbs.h>

#include <span>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "jaccl/tcp.h"

constexpr const char* IBV_TAG = "[jaccl]";
constexpr int SEND_WR = 1;
constexpr int RECV_WR = 2;
constexpr int MAX_SEND_WR = 32;
constexpr int MAX_RECV_WR = 32;
constexpr int BUFFER_SIZES = 8;
constexpr int NUM_BUFFERS = 2;
constexpr int FRAME_SIZE = 4096;

namespace {

template <typename T, typename = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<
    T,
    std::void_t<typename T::value_type, typename T::iterator>>
    : std::true_type {};

inline std::pair<int, int64_t> buffer_size_from_message(int64_t msg) {
  if (__builtin_available(macOS 26.3, iOS 26.3, tvOS 26.3, visionOS 26.3, *)) {
    for (int k = BUFFER_SIZES - 1; k > 0; k--) {
      if (msg >= FRAME_SIZE * (1 << k)) {
        return {k, FRAME_SIZE * (1 << k)};
      }
    }
  }
  return {0, FRAME_SIZE};
}

} // namespace

namespace jaccl {

/**
 * Wrapper for the ibverbs API.
 */
struct IBVWrapper {
  IBVWrapper();
  bool is_available() {
    return librdma_handle_ != nullptr;
  }

  // API
  ibv_device** (*get_device_list)(int*);
  const char* (*get_device_name)(ibv_device*);
  ibv_context* (*open_device)(ibv_device*);
  void (*free_device_list)(ibv_device**);
  int (*close_device)(ibv_context*);

  ibv_pd* (*alloc_pd)(ibv_context*);
  ibv_qp* (*create_qp)(ibv_pd*, ibv_qp_init_attr*);
  ibv_cq* (*create_cq)(ibv_context*, int, void*, ibv_comp_channel*, int);
  int (*destroy_cq)(ibv_cq*);
  int (*destroy_qp)(ibv_qp*);
  int (*dealloc_pd)(ibv_pd*);

  int (*query_port)(ibv_context*, uint8_t, ibv_port_attr*);
  int (*query_gid)(ibv_context*, uint8_t, int, ibv_gid*);
  int (*modify_qp)(ibv_qp*, ibv_qp_attr*, int);
  ibv_mr* (*reg_mr)(ibv_pd*, void*, size_t, int);
  int (*dereg_mr)(ibv_mr*);

 private:
  void* librdma_handle_;
};

IBVWrapper& ibv();

/**
 * Contains the information that defines a destination to a remote device.
 * Basically we can compute our own destination and share it with remote hosts
 * over the side channel.
 */
struct Destination {
  int local_id;
  int queue_pair_number;
  int packet_sequence_number;
  ibv_gid global_identifier;
};

/**
 * A buffer that can be registered to a number of protection domains.
 */
class SharedBuffer {
 public:
  SharedBuffer(size_t num_bytes);
  SharedBuffer(SharedBuffer&& b);
  ~SharedBuffer();

  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;

  void register_to_protection_domain(ibv_pd* protection_domain);

  size_t size() const {
    return num_bytes_;
  }

  uint32_t local_key(ibv_pd* protection_domain) const {
    return memory_regions_.at(protection_domain)->lkey;
  }

  ibv_sge to_scatter_gather_entry(ibv_pd* protection_domain) const {
    ibv_sge entry;
    entry.addr = reinterpret_cast<uintptr_t>(data_);
    entry.length = size();
    entry.lkey = local_key(protection_domain);
    return entry;
  }

  template <typename T>
  T* data() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  T* begin() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  T* end() {
    return static_cast<T*>(data_) + size() / sizeof(T);
  }

 private:
  void* data_;
  size_t num_bytes_;
  std::unordered_map<ibv_pd*, ibv_mr*> memory_regions_;
};

/**
 * Manipulates an RDMA connection. Enables (among other things)
 *
 *   - Creating a queue pair
 *   - Sending and receiving
 *   - Checking completion
 */
struct Connection {
  ibv_context* ctx;
  ibv_pd* protection_domain;
  ibv_cq* completion_queue;
  ibv_qp* queue_pair;
  Destination src; // holds the local information

  Connection(ibv_context* ctx_);
  Connection(Connection&& c);

  Connection(const Connection&) = delete;
  Connection& operator=(Connection&) = delete;

  ~Connection();
  void allocate_protection_domain();
  void create_completion_queue(int num_entries);
  void create_queue_pair();

  const Destination& info();
  void queue_pair_init();
  void queue_pair_rtr(const Destination& dst);
  void queue_pair_rts();

  void post_send(const SharedBuffer& buff, uint64_t work_request_id) {
    ibv_send_wr work_request, *bad_work_request;

    auto entry = buff.to_scatter_gather_entry(protection_domain);
    work_request.wr_id = work_request_id;
    work_request.sg_list = &entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_SEND;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.next = nullptr;

    if (int status =
            ibv_post_send(queue_pair, &work_request, &bad_work_request);
        status != 0) {
      std::ostringstream msg;
      msg << "[jaccl] Send failed with error code " << status;
      throw std::invalid_argument(msg.str());
    }
  }

  void post_recv(const SharedBuffer& buff, uint64_t work_request_id) {
    ibv_recv_wr work_request, *bad_work_request;

    auto entry = buff.to_scatter_gather_entry(protection_domain);
    work_request.wr_id = work_request_id;
    work_request.sg_list = &entry;
    work_request.num_sge = 1;
    work_request.next = nullptr;

    if (int status =
            ibv_post_recv(queue_pair, &work_request, &bad_work_request);
        status != 0) {
      std::ostringstream msg;
      msg << "[jaccl] Recv failed with error code " << status;
      throw std::invalid_argument(msg.str());
    }
  }

  int poll(int num_completions, ibv_wc* work_completions) {
    return ibv_poll_cq(completion_queue, num_completions, work_completions);
  }
};

std::vector<Connection> create_connections(
    const std::vector<std::string>& device_names);

inline int poll(
    std::span<const Connection> connections,
    int num_completions,
    ibv_wc* work_completions) {
  int completions = 0;
  for (auto& c : connections) {
    if (c.ctx == nullptr) {
      continue;
    }
    if (completions >= num_completions) {
      return completions;
    }

    int n = ibv_poll_cq(
        c.completion_queue,
        num_completions - completions,
        work_completions + completions);

    completions += n;
  }
  return completions;
}

inline int poll(
    std::span<const Connection> connections_1,
    std::span<const Connection> connections_2,
    int num_completions,
    ibv_wc* work_completions) {
  int completions = 0;
  completions += poll(connections_1, num_completions, work_completions);
  completions += poll(
      connections_2,
      num_completions - completions,
      work_completions + completions);
  return completions;
}

/**
 * Implement a TCP side channel to exchange information about the RDMA
 * connections.
 *
 * Implements a simple all gather where every node sends to rank 0 and rank 0
 * broadcasts to every node.
 */
class SideChannel {
 public:
  SideChannel(int rank, int size, const char* addr);
  SideChannel(SideChannel&& sc);

  SideChannel(const SideChannel&) = delete;
  SideChannel& operator=(const SideChannel&) = delete;

  template <typename T>
  std::vector<T> all_gather(const T& v) {
    std::vector<T> result(size_);

    // T is a container of stuff like std::vector or std::string
    if constexpr (is_container<T>::value) {
      using U = typename T::value_type;

      // Share the lengths first and set the communication size to be the
      // maximum length of the containers.
      auto lengths = all_gather<int>(v.size());
      auto max_len = *std::max_element(lengths.begin(), lengths.end());
      for (auto& s : result) {
        s.resize(max_len);
      }

      // All gather of length max_len
      if (rank_ == 0) {
        std::copy(v.begin(), v.end(), result[rank_].begin());
        for (int i = 1; i < size_; i++) {
          sockets_[i - 1].recv(IBV_TAG, result[i].data(), sizeof(U) * max_len);
        }
        for (int i = 1; i < size_; i++) {
          for (int j = 0; j < size_; j++) {
            sockets_[i - 1].send(
                IBV_TAG, result[j].data(), sizeof(U) * max_len);
          }
        }
      } else {
        std::copy(v.begin(), v.end(), result[rank_].begin());
        sockets_[0].send(IBV_TAG, result[rank_].data(), sizeof(U) * max_len);
        for (int i = 0; i < size_; i++) {
          sockets_[0].recv(IBV_TAG, result[i].data(), sizeof(U) * max_len);
        }
      }

      // Resize the outputs back to the original length
      for (int i = 0; i < size_; i++) {
        result[i].resize(lengths[i]);
      }
    }

    // T is a scalar
    else {
      if (rank_ == 0) {
        result[rank_] = v;
        for (int i = 1; i < size_; i++) {
          sockets_[i - 1].recv(IBV_TAG, &result[i], sizeof(T));
        }
        for (int i = 1; i < size_; i++) {
          sockets_[i - 1].send(IBV_TAG, result.data(), size_ * sizeof(T));
        }
      } else {
        sockets_[0].send(IBV_TAG, &v, sizeof(T));
        sockets_[0].recv(IBV_TAG, result.data(), size_ * sizeof(T));
      }
    }

    return result;
  }

  void barrier() {
    // Twice has proven to be more robust to initialization issues.
    all_gather<int>(0);
    all_gather<int>(0);
  }

 private:
  int rank_;
  int size_;
  std::vector<TCPSocket> sockets_;
};

} // namespace jaccl
