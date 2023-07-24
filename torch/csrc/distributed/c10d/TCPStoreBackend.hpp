#pragma once

#include <chrono>
#include <thread>
#include <vector>

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

// Abstract base class to handle thread state for TCPStoreMasterDaemon.
// Contains the windows/unix implementations to signal a
// shutdown sequence for the thread
class BackgroundThread {
 public:
  explicit BackgroundThread(Socket&& storeListenSocket);

  virtual ~BackgroundThread() = 0;

 protected:
  void dispose();

  Socket storeListenSocket_;
  std::thread daemonThread_{};
  std::vector<Socket> sockets_{};
#ifdef _WIN32
  const std::chrono::milliseconds checkTimeout_ = std::chrono::milliseconds{10};
  HANDLE ghStopEvent_{};
#else
  std::array<int, 2> controlPipeFd_{{-1, -1}};
#endif

 private:
  // Initialization for shutdown signal
  void initStopSignal();
  // Triggers the shutdown signal
  void stop();
  // Joins the thread
  void join();
  // Clean up the shutdown signal
  void closeStopSignal();
};

} // namespace detail
} // namespace c10d
