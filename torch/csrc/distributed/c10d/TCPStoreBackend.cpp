#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>

#include <c10/util/irange.h>
#include <fcntl.h>
#include <algorithm>
#include <array>
#include <system_error>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#include <io.h>
#include <winsock2.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <torch/csrc/distributed/c10d/WinSockUtils.hpp>
#else
#include <torch/csrc/distributed/c10d/UnixSockUtils.hpp>
#endif

#include <torch/csrc/distributed/c10d/socket.h>

namespace c10d {
namespace detail {

// Background thread parent class methods
BackgroundThread::BackgroundThread(Socket&& storeListenSocket)
    : storeListenSocket_{std::move(storeListenSocket)} {
  // Signal instance destruction to the daemon thread.
  initStopSignal();
}

BackgroundThread::~BackgroundThread() = default;

// WARNING:
// Since we rely on the subclass for the daemon thread clean-up, we cannot
// destruct our member variables in the destructor. The subclass must call
// dispose() in its own destructor.
void BackgroundThread::dispose() {
  // Stop the run
  stop();
  // Join the thread
  join();
  // Close unclosed sockets
  sockets_.clear();
  // Now close the rest control pipe
  closeStopSignal();
}

void BackgroundThread::join() {
  daemonThread_.join();
}

#ifdef _WIN32
void BackgroundThread::initStopSignal() {
  ghStopEvent_ = CreateEvent(NULL, TRUE, FALSE, NULL);
  if (ghStopEvent_ == NULL) {
    TORCH_CHECK(
        false,
        "Failed to create the control pipe to start the "
        "BackgroundThread run");
  }
}

void BackgroundThread::closeStopSignal() {
  CloseHandle(ghStopEvent_);
}

void BackgroundThread::stop() {
  SetEvent(ghStopEvent_);
}
#else
void BackgroundThread::initStopSignal() {
  if (pipe(controlPipeFd_.data()) == -1) {
    TORCH_CHECK(
        false,
        "Failed to create the control pipe to start the "
        "BackgroundThread run");
  }
}

void BackgroundThread::closeStopSignal() {
  for (int fd : controlPipeFd_) {
    if (fd != -1) {
      ::close(fd);
    }
  }
}

void BackgroundThread::stop() {
  if (controlPipeFd_[1] != -1) {
    ::write(controlPipeFd_[1], "\0", 1);
    // close the write end of the pipe
    ::close(controlPipeFd_[1]);
    controlPipeFd_[1] = -1;
  }
}
#endif

} // namespace detail
} // namespace c10d
