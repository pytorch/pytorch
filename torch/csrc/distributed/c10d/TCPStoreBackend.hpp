#pragma once

#include <chrono>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/socket.h>

#ifdef _WIN32
#include <io.h>
#include <winsock2.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

namespace c10d {
namespace detail {

enum class QueryType : uint8_t {
  SET,
  COMPARE_SET,
  GET,
  ADD,
  CHECK,
  WAIT,
  GETNUMKEYS,
  DELETE_KEY,
  APPEND,
  MULTI_GET,
  MULTI_SET,
  CANCEL_WAIT,
};

enum class CheckResponseType : uint8_t { READY, NOT_READY };

enum class WaitResponseType : uint8_t { STOP_WAITING, WAIT_CANCELED };

// Abstract base class to handle thread state for TCPStoreMasterDaemon.
// Contains the windows/unix implementations to signal a
// shutdown sequence for the thread
class BackgroundThread {
 public:
  explicit BackgroundThread();

  virtual ~BackgroundThread() = 0;
  virtual std::uint16_t port() const = 0;

  void start();
  bool stop_requested();

 protected:
  void dispose();
  virtual void run() = 0;
  virtual void stop() = 0;
  bool is_running() {
    return is_running_.load();
  }

 private:
  std::atomic<bool> is_running_;
  std::thread daemonThread_{};
};

std::unique_ptr<BackgroundThread> create_tcpstore_backend(
    const TCPStoreOptions& opts);
std::unique_ptr<BackgroundThread> create_libuv_tcpstore_backend(
    const TCPStoreOptions& opts);
bool is_libuv_tcpstore_backend_available();

} // namespace detail
} // namespace c10d
