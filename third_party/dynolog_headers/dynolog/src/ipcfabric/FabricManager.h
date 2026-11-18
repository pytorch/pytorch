// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include "dynolog/src/ipcfabric/Endpoint.h"
#include "dynolog/src/ipcfabric/Utils.h"

// If building inside Kineto, use its logger, otherwise use glog
#if defined(KINETO_NAMESPACE) && defined(ENABLE_IPC_FABRIC)
// We need to include the Logger header here for LOG() macros.
// However this can alias with other files that include this and
// also use glog. TODO(T131440833). Thus, the user should also set
#include "Logger.h" // @manual
// set error to use kineto version
#define ERROR libkineto::ERROR

#else // KINETO_NAMESPACE && ENABLE_IPC_FABRIC
#include <glog/logging.h>
#endif // KINETO_NAMESPACE && ENABLE_IPC_FABRIC

namespace dynolog::ipcfabric {

constexpr int TYPE_SIZE = 32;

struct Metadata {
  size_t size = 0;
  char type[TYPE_SIZE] = "";
};

struct Message {
  template <class T>
  static std::unique_ptr<Message> constructMessage(
      const T& data,
      const std::string& type) {
    std::unique_ptr<Message> msg = std::make_unique<Message>(Message());
    memcpy(msg->metadata.type, type.c_str(), type.size() + 1);
    // TODO CXX 17 - https://github.com/pytorch/kineto/issues/650
    // if constexpr (std::is_same_v<std::string, T>) {
    // Without constexpr following is not possible
#if __cplusplus >= 201703L
    if constexpr (std::is_same<std::string, T>::value == true) {
      msg->metadata.size = data.size();
      msg->buf = std::make_unique<unsigned char[]>(msg->metadata.size);
      memcpy(msg->buf.get(), data.c_str(), msg->metadata.size);
    } else {
#endif
      // ensure memcpy works on T
      // TODO CXX 17 - https://github.com/pytorch/kineto/issues/650
      // static_assert(std::is_trivially_copyable_v<T>);
      static_assert(std::is_trivially_copyable<T>::value);
      msg->metadata.size = sizeof(data);
      msg->buf = std::make_unique<unsigned char[]>(msg->metadata.size);
      memcpy(msg->buf.get(), &data, msg->metadata.size);
#if __cplusplus >= 201703L
    }
#endif
    return msg;
  }

  // Construct message where T is a trivially copyable struct with
  // a flexible array at the end. The items in the flexible array are
  // of (trivially copyable) type U and there are n of them.
  template <class T, class U>
  static std::unique_ptr<Message>
  constructMessage(const T& data, const std::string& type, int n) {
    std::unique_ptr<Message> msg = std::make_unique<Message>(Message());
    memcpy(msg->metadata.type, type.c_str(), type.size() + 1);
    // ensure memcpy works on T and U
    // TODO CXX 17 - https://github.com/pytorch/kineto/issues/650
    // static_assert(std::is_trivially_copyable_v<T>);
    // static_assert(std::is_trivially_copyable_v<U>);
    static_assert(std::is_trivially_copyable<T>::value);
    static_assert(std::is_trivially_copyable<U>::value);
    msg->metadata.size = sizeof(data) + sizeof(U) * n;
    msg->buf = std::make_unique<unsigned char[]>(msg->metadata.size);
    memcpy(msg->buf.get(), &data, msg->metadata.size);
    return msg;
  }

  Metadata metadata;
  std::unique_ptr<unsigned char[]> buf;
  // endpoint name of the sender
  std::string src;
};

class FabricManager {
 public:
  FabricManager(const FabricManager&) = delete;
  FabricManager& operator=(const FabricManager&) = delete;

  static std::unique_ptr<FabricManager> factory(
      std::string endpoint_name = "") {
    try {
      return std::unique_ptr<FabricManager>(new FabricManager(endpoint_name));
    } catch (std::exception& e) {
      LOG(ERROR) << "Error when initializing FabricManager: " << e.what();
    }
    return nullptr;
  }

  // warning: this will block for user passed in time with exponential increase
  // if send keeps failing
  bool sync_send(
      const Message& msg,
      const std::string& dest_name,
      int num_retries = 10,
      int sleep_time_us = 10000) {
    if (dest_name.size() == 0) {
      LOG(ERROR) << "Cannot send to empty socket name";
      return false;
    }

    std::vector<Payload> payload{
        Payload(sizeof(struct Metadata), (void*)&msg.metadata),
        Payload(msg.metadata.size, msg.buf.get())};
    int i = 0;
    try {
      auto ctxt = ep_.buildSendCtxt(dest_name, payload, std::vector<int>());
      while (!ep_.trySendMsg(*ctxt) && i < num_retries) {
        i++;
        /* sleep override */
        usleep(sleep_time_us);
        sleep_time_us *= 2;
      }
    } catch (std::exception& e) {
      LOG(ERROR) << "Error when sync_send(): " << e.what();
      return false;
    }
    return i < num_retries;
  }

  bool recv() {
    try {
      Metadata receive_metadata;
      std::vector<Payload> peek_payload{
          Payload(sizeof(struct Metadata), &receive_metadata)};
      auto peek_ctxt = ep_.buildRcvCtxt(peek_payload);
      // unix socket only fills the data for the iov that have a non NULL
      // buffer. Leverage that to read metadata to find buffer size by:
      //   1) FabricManager assumes metadata in first iov, data in second
      //   2) peek with only metadata buffer in iov
      //   3) read metadata
      //   4) use metadata to find the desired size for the buffer to allocate.
      //   5) read metadata + data with allocated buffer
      bool success;
      try {
        success = ep_.tryPeekMsg(*peek_ctxt);
      } catch (std::exception& e) {
        LOG(ERROR) << "Error when tryPeekMsg(): " << e.what();
        return false;
      }
      if (success) {
        std::unique_ptr<Message> msg = std::make_unique<Message>(Message());
        msg->metadata = receive_metadata;
        // new unsigned char[N] guarantees the maximum alignment to hold any
        // object thus we don't need to worry about memory alignment here see
        // https://stackoverflow.com/questions/10587879/does-new-char-actually-guarantee-aligned-memory-for-a-class-type
        // and
        // https://stackoverflow.com/questions/39668561/allocate-n-bytes-by-new-and-fill-it-with-any-type
        msg->buf = std::unique_ptr<unsigned char[]>(
            new unsigned char[receive_metadata.size]);
        msg->src = std::string(ep_.getName(*peek_ctxt));
        std::vector<Payload> payload{
            Payload(sizeof(struct Metadata), (void*)&msg->metadata),
            Payload(receive_metadata.size, msg->buf.get())};
        auto recv_ctxt = ep_.buildRcvCtxt(payload);

        // the deque can potentially grow very large
        try {
          success = ep_.tryRcvMsg(*recv_ctxt);
        } catch (std::exception& e) {
          LOG(ERROR) << "Error when tryRcvMsg(): " << e.what();
          return false;
        }
        if (success) {
          std::lock_guard<std::mutex> wguard(dequeLock_);
          message_deque_.push_back(std::move(msg));
          return true;
        }
      }
    } catch (std::exception& e) {
      LOG(ERROR) << "Error in recv(): " << e.what();
      return false;
    }
    return false;
  }

  std::unique_ptr<Message> retrieve_msg() {
    std::lock_guard<std::mutex> wguard(dequeLock_);
    if (message_deque_.empty()) {
      return nullptr;
    }
    std::unique_ptr<Message> msg = std::move(message_deque_.front());
    message_deque_.pop_front();
    return msg;
  }

  std::unique_ptr<Message> poll_recv(int max_retries, int sleep_time_us) {
    for (int i = 0; i < max_retries; i++) {
      if (recv()) {
        return retrieve_msg();
      }
      /* sleep override */
      usleep(sleep_time_us);
    }
    return nullptr;
  }

 private:
  explicit FabricManager(std::string endpoint_name = "") : ep_{endpoint_name} {}
  // message LIFO deque with oldest message at front
  std::deque<std::unique_ptr<Message>> message_deque_;
  EndPoint<0> ep_;
  std::mutex dequeLock_;
};

} // namespace dynolog::ipcfabric
