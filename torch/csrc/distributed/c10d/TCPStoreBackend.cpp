
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

#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>

#ifdef _WIN32
#include <torch/csrc/distributed/c10d/WinSockUtils.hpp>
#else
#include <torch/csrc/distributed/c10d/UnixSockUtils.hpp>
#endif

#include <torch/csrc/distributed/c10d/socket.h>

namespace c10d {
namespace detail {

// Background thread parent class methods
BackgroundThread::BackgroundThread() {}

BackgroundThread::~BackgroundThread() = default;

// WARNING:
// Since we rely on the subclass for the daemon thread clean-up, we cannot
// destruct our member variables in the destructor. The subclass must call
// dispose() in its own destructor.
void BackgroundThread::dispose() {
  // Stop the run
  stop();
  // Join the thread
  daemonThread_.join();
}

void BackgroundThread::start() {
  daemonThread_ = std::thread{&BackgroundThread::run, this};
  is_running_.store(true);
}

// Separate thread that is only launched on master
class TCPStoreMasterDaemon : public BackgroundThread {
 public:
  explicit TCPStoreMasterDaemon(Socket&& storeListenSocket);

  ~TCPStoreMasterDaemon() override;

  uint16_t port() const override;

 protected:
  void run() override;
  void stop() override;

 private:
  void initStopSignal();
  void closeStopSignal();

  void queryFds(std::vector<struct pollfd>& fds);
  void query(int socket);

  void clearSocketWaitState(int socket);

  // The master runs on a single thread so only
  // one handler can be executed at a time
  void validateHandler(int socket);
  void setHandler(int socket);
  void compareSetHandler(int socket);
  void addHandler(int socket);
  void getHandler(int socket) const;
  void checkHandler(int socket) const;
  void getNumKeysHandler(int socket) const;
  void deleteHandler(int socket);
  void waitHandler(int socket);
  void appendHandler(int socket);
  void multiGetHandler(int socket);
  void multiSetHandler(int socket);
  void cancelWaitHandler(int socket);
  void addMiscellaneousSocket(int socket);
  void removeMiscellaneousSocket(int socket);
  bool isMiscellaneousSocket(int socket);

  bool checkKeys(const std::vector<std::string>& keys) const;
  // Helper function to alerts waiting workers, used in setHandler, getHandler
  void wakeupWaitingClients(const std::string& key);
  void doSet(const std::string& key, const std::vector<uint8_t>& newData);

  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // From key -> the list of sockets waiting on the key
  std::unordered_map<std::string, std::vector<int>> waitingSockets_;
  // From socket -> number of keys awaited
  std::unordered_map<int, size_t> keysAwaited_;
  // miscellaneous sockets
  std::unordered_set<int> miscellaneousSockets_;

  Socket storeListenSocket_;
  std::vector<Socket> sockets_{};
#ifdef _WIN32
  const std::chrono::milliseconds checkTimeout_ = std::chrono::milliseconds{10};
  HANDLE ghStopEvent_{};
#else
  std::array<int, 2> controlPipeFd_{{-1, -1}};
#endif
};

// Simply start the daemon thread
TCPStoreMasterDaemon::TCPStoreMasterDaemon(Socket&& storeListenSocket)
    : storeListenSocket_{std::move(storeListenSocket)} {
  initStopSignal();
}

TCPStoreMasterDaemon::~TCPStoreMasterDaemon() {
  dispose();
  // it's now safe for us to cleanup
  // Close unclosed sockets
  sockets_.clear();
  // Now close the rest control pipe
  closeStopSignal();
}

std::uint16_t TCPStoreMasterDaemon::port() const {
  return storeListenSocket_.port();
}

#ifdef _WIN32
void TCPStoreMasterDaemon::initStopSignal() {
  ghStopEvent_ = CreateEvent(NULL, TRUE, FALSE, NULL);
  if (ghStopEvent_ == NULL) {
    TORCH_CHECK(
        false,
        "Failed to create the control pipe to start the "
        "BackgroundThread run");
  }
}

void TCPStoreMasterDaemon::closeStopSignal() {
  CloseHandle(ghStopEvent_);
}

void TCPStoreMasterDaemon::stop() {
  SetEvent(ghStopEvent_);
}

#else
void TCPStoreMasterDaemon::initStopSignal() {
  if (pipe(controlPipeFd_.data()) == -1) {
    TORCH_CHECK(
        false,
        "Failed to create the control pipe to start the "
        "BackgroundThread run");
  }
}

void TCPStoreMasterDaemon::closeStopSignal() {
  for (int fd : controlPipeFd_) {
    if (fd != -1) {
      ::close(fd);
    }
  }
}

void TCPStoreMasterDaemon::stop() {
  if (controlPipeFd_[1] != -1) {
    ssize_t written_bytes = -1;
    while (true) {
      written_bytes = ::write(controlPipeFd_[1], "\0", 1);
      if (written_bytes < 0) {
        if (errno == EAGAIN) {
          continue;
        }
        TORCH_CHECK(false, "Failed to write the control pipe:", errno);
      }
      break;
    }
    if (written_bytes == 0) {
      TORCH_CHECK(false, "Failed to write the control pipe");
    }

    // close the write end of the pipe
    ::close(controlPipeFd_[1]);
    controlPipeFd_[1] = -1;
  }
}
#endif

void TCPStoreMasterDaemon::queryFds(std::vector<struct pollfd>& fds) {
  // Skipping the fds[0] and fds[1],
  // fds[0] is master's listening socket
  // fds[1] is control pipe's reading fd, it is not for Windows platform
  for (size_t fdIdx = CONNECT_SOCKET_OFFSET; fdIdx < fds.size(); ++fdIdx) {
    if (fds[fdIdx].revents == 0) {
      continue;
    }

    // Now query the socket that has the event
    try {
      query(fds[fdIdx].fd);
    } catch (...) {
      // There was an error when processing query. Probably an exception
      // occurred in recv/send what would indicate that socket on the other
      // side has been closed. If the closing was due to normal exit, then
      // the store should continue executing. Otherwise, if it was different
      // exception, other connections will get an exception once they try to
      // use the store. We will go ahead and close this connection whenever
      // we hit an exception here.
      clearSocketWaitState(fds[fdIdx].fd);

      fds.erase(fds.begin() + fdIdx);
      sockets_.erase(sockets_.begin() + fdIdx - CONNECT_SOCKET_OFFSET);
      --fdIdx;
      continue;
    }
  }
}

void TCPStoreMasterDaemon::clearSocketWaitState(int socket) {
  // Remove all the tracking state of the close FD
  for (auto it = waitingSockets_.begin(); it != waitingSockets_.end();) {
    for (auto vecIt = it->second.begin(); vecIt != it->second.end();) {
      if (*vecIt == socket) {
        vecIt = it->second.erase(vecIt);
      } else {
        ++vecIt;
      }
    }
    if (it->second.empty()) {
      it = waitingSockets_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = keysAwaited_.begin(); it != keysAwaited_.end();) {
    if (it->first == socket) {
      it = keysAwaited_.erase(it);
    } else {
      ++it;
    }
  }
}

// query communicates with the worker. The format
// of the query is as follows:
// type of query | size of arg1 | arg1 | size of arg2 | arg2 | ...
// or, in the case of wait
// type of query | number of args | size of arg1 | arg1 | ...
void TCPStoreMasterDaemon::query(int socket) {
  QueryType qt;
  tcputil::recvBytes<QueryType>(socket, &qt, 1);

  if (isMiscellaneousSocket(socket)) {
    removeMiscellaneousSocket(socket);
    if (qt == QueryType::VALIDATE) {
      validateHandler(socket);
    } else {
      // real miscellaneous client: the first msg is not VALIDATE
      TORCH_CHECK(
          false, "Miscellaneous client without VALIDATE query is detected");
    }
  } else if (qt == QueryType::SET) {
    setHandler(socket);

  } else if (qt == QueryType::COMPARE_SET) {
    compareSetHandler(socket);

  } else if (qt == QueryType::ADD) {
    addHandler(socket);

  } else if (qt == QueryType::GET) {
    getHandler(socket);

  } else if (qt == QueryType::CHECK) {
    checkHandler(socket);

  } else if (qt == QueryType::WAIT) {
    waitHandler(socket);

  } else if (qt == QueryType::GETNUMKEYS) {
    getNumKeysHandler(socket);

  } else if (qt == QueryType::DELETE_KEY) {
    deleteHandler(socket);
  } else if (qt == QueryType::APPEND) {
    appendHandler(socket);
  } else if (qt == QueryType::MULTI_GET) {
    multiGetHandler(socket);
  } else if (qt == QueryType::MULTI_SET) {
    multiSetHandler(socket);
  } else if (qt == QueryType::CANCEL_WAIT) {
    cancelWaitHandler(socket);
  } else {
    TORCH_CHECK(false, "Unexpected query type");
  }
}

void TCPStoreMasterDaemon::wakeupWaitingClients(const std::string& key) {
  auto socketsToWait = waitingSockets_.find(key);
  if (socketsToWait != waitingSockets_.end()) {
    for (int socket : socketsToWait->second) {
      if (--keysAwaited_[socket] == 0) {
        tcputil::sendValue<WaitResponseType>(
            socket, WaitResponseType::STOP_WAITING);
      }
    }
    waitingSockets_.erase(socketsToWait);
  }
}

void TCPStoreMasterDaemon::doSet(
    const std::string& key,
    const std::vector<uint8_t>& newData) {
  tcpStore_[key] = newData;
  // On "set", wake up all clients that have been waiting
  wakeupWaitingClients(key);
}

void TCPStoreMasterDaemon::validateHandler(int socket) {
  uint32_t validateNumber;
  tcputil::recvBytes<uint32_t>(socket, &validateNumber, 1);
  if (validateNumber != detail::validationMagicNumber) {
    TORCH_CHECK(
        false,
        "Miscellaneous client with incorrect VALIDATE query is detected");
  }
}

void TCPStoreMasterDaemon::setHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  std::vector<uint8_t> newData = tcputil::recvVector<uint8_t>(socket);
  doSet(key, newData);
}

void TCPStoreMasterDaemon::compareSetHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  std::vector<uint8_t> currentValue = tcputil::recvVector<uint8_t>(socket);
  std::vector<uint8_t> newValue = tcputil::recvVector<uint8_t>(socket);

  auto pos = tcpStore_.find(key);
  if (pos == tcpStore_.end()) {
    if (currentValue.empty()) {
      tcpStore_[key] = newValue;
      tcputil::sendVector<uint8_t>(socket, newValue);
    } else {
      // TODO: This code path is not ideal as we are "lying" to the caller in
      // case the key does not exist. We should come up with a working solution.
      tcputil::sendVector<uint8_t>(socket, currentValue);
    }
  } else {
    if (pos->second == currentValue) {
      pos->second = std::move(newValue);
    }
    tcputil::sendVector<uint8_t>(socket, pos->second);
  }
}

void TCPStoreMasterDaemon::addHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  int64_t addVal = tcputil::recvValue<int64_t>(socket);

  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    auto buf = reinterpret_cast<const char*>(it->second.data());
    auto len = it->second.size();
    addVal += std::stoll(std::string(buf, len));
  }
  auto addValStr = std::to_string(addVal);
  std::vector<uint8_t> newData =
      std::vector<uint8_t>(addValStr.begin(), addValStr.end());
  tcpStore_[key] = newData;
  // Now send the new value
  tcputil::sendValue<int64_t>(socket, addVal);
  // On "add", wake up all clients that have been waiting
  wakeupWaitingClients(key);
}

void TCPStoreMasterDaemon::getHandler(int socket) const {
  std::string key = tcputil::recvString(socket);
  auto data = tcpStore_.at(key);
  tcputil::sendVector<uint8_t>(socket, data);
}

void TCPStoreMasterDaemon::getNumKeysHandler(int socket) const {
  tcputil::sendValue<int64_t>(socket, tcpStore_.size());
}

void TCPStoreMasterDaemon::deleteHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  auto numDeleted = tcpStore_.erase(key);
  tcputil::sendValue<int64_t>(socket, numDeleted);
}

void TCPStoreMasterDaemon::checkHandler(int socket) const {
  SizeType nargs = 0;
  tcputil::recvBytes<SizeType>(socket, &nargs, 1);
  std::vector<std::string> keys(nargs);
  for (const auto i : c10::irange(nargs)) {
    keys[i] = tcputil::recvString(socket);
  }
  // Now we have received all the keys
  if (checkKeys(keys)) {
    tcputil::sendValue<CheckResponseType>(socket, CheckResponseType::READY);
  } else {
    tcputil::sendValue<CheckResponseType>(socket, CheckResponseType::NOT_READY);
  }
}

void TCPStoreMasterDaemon::waitHandler(int socket) {
  SizeType nargs = 0;
  tcputil::recvBytes<SizeType>(socket, &nargs, 1);
  std::vector<std::string> keys(nargs);
  for (const auto i : c10::irange(nargs)) {
    keys[i] = tcputil::recvString(socket);
  }
  if (checkKeys(keys)) {
    tcputil::sendValue<WaitResponseType>(
        socket, WaitResponseType::STOP_WAITING);
  } else {
    int numKeysToAwait = 0;
    for (auto& key : keys) {
      // Only count keys that have not already been set
      if (tcpStore_.find(key) == tcpStore_.end()) {
        waitingSockets_[key].push_back(socket);
        numKeysToAwait++;
      }
    }
    keysAwaited_[socket] = numKeysToAwait;
  }
}

void TCPStoreMasterDaemon::appendHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  std::vector<uint8_t> newData = tcputil::recvVector<uint8_t>(socket);
  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    it->second.insert(it->second.end(), newData.begin(), newData.end());
  } else {
    tcpStore_[key] = newData;
  }
  // we should not have clients waiting if we're appending, so it's all fine
  wakeupWaitingClients(key);
}

void TCPStoreMasterDaemon::multiGetHandler(int socket) {
  SizeType nargs = 0;
  tcputil::recvBytes<SizeType>(socket, &nargs, 1);
  for (const auto i : c10::irange(nargs)) {
    auto key = tcputil::recvString(socket);
    auto& data = tcpStore_.at(key);
    tcputil::sendVector<uint8_t>(socket, data, i < (nargs - 1));
  }
}

void TCPStoreMasterDaemon::multiSetHandler(int socket) {
  SizeType nargs = 0;
  tcputil::recvBytes<SizeType>(socket, &nargs, 1);
  for (auto _ : c10::irange(nargs)) {
    (void)_; // Suppress unused variable warning
    auto key = tcputil::recvString(socket);
    auto value = tcputil::recvVector<uint8_t>(socket);
    doSet(key, value);
  }
}

void TCPStoreMasterDaemon::cancelWaitHandler(int socket) {
  clearSocketWaitState(socket);

  // Send update to TCPStoreWorkerDaemon on client
  tcputil::sendValue<WaitResponseType>(
      socket, detail::WaitResponseType::WAIT_CANCELED);
}

bool TCPStoreMasterDaemon::checkKeys(
    const std::vector<std::string>& keys) const {
  return std::all_of(keys.begin(), keys.end(), [this](const std::string& s) {
    return tcpStore_.count(s) > 0;
  });
}

void TCPStoreMasterDaemon::addMiscellaneousSocket(int socket) {
  if (miscellaneousSockets_.find(socket) == miscellaneousSockets_.end()) {
    miscellaneousSockets_.insert(socket);
  }
}

void TCPStoreMasterDaemon::removeMiscellaneousSocket(int socket) {
  auto it = miscellaneousSockets_.find(socket);
  if (it != miscellaneousSockets_.end()) {
    miscellaneousSockets_.erase(it);
  }
}

bool TCPStoreMasterDaemon::isMiscellaneousSocket(int socket) {
  return miscellaneousSockets_.find(socket) != miscellaneousSockets_.end();
}

#ifdef _WIN32
void TCPStoreMasterDaemon::run() {
  std::vector<struct pollfd> fds;
  tcputil::addPollfd(fds, storeListenSocket_.handle(), POLLIN);

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (const auto i : c10::irange(sockets_.size())) {
      fds[i].revents = 0;
    }

    int res;
    SYSCHECK_ERR_RETURN_NEG1(
        res = WSAPoll(fds.data(), fds.size(), checkTimeout_.count()))
    if (res == 0) {
      auto rv = WaitForSingleObject(ghStopEvent_, 0);
      if (rv != WAIT_TIMEOUT) {
        finished = true;
        break;
      }
      continue;
    }

    // TCPStore's listening socket has an event and it should now be able to
    // accept new connections.
    if (fds[0].revents != 0) {
      if (!(fds[0].revents & POLLIN)) {
        C10_THROW_ERROR(
            DistStoreError,
            "Unexpected poll revent on the master's listening socket: " +
                std::to_string(fds[0].revents));
      }
      Socket socket = storeListenSocket_.accept();
      int rawSocket = socket.handle();
      sockets_.emplace_back(std::move(socket));
      tcputil::addPollfd(fds, rawSocket, POLLIN);
      addMiscellaneousSocket(rawSocket);
    }
    queryFds(fds);
  }
}
#else
void TCPStoreMasterDaemon::run() {
  std::vector<struct pollfd> fds;
  tcputil::addPollfd(fds, storeListenSocket_.handle(), POLLIN);
  // Although we haven't found any documentation or literature describing this,
  // we've seen cases that, under certain circumstances, the read end of the
  // pipe won't receive POLLHUP when the write end is closed. However, under
  // the same circumstances, writing to the pipe will guarantee POLLIN to be
  // received on the read end.
  //
  // For more reliable termination, the main thread will write a byte to the
  // pipe before closing it, and the background thread will poll for both
  // POLLIN and POLLHUP.
  tcputil::addPollfd(fds, controlPipeFd_[0], POLLIN | POLLHUP);

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (const auto i : c10::irange(sockets_.size())) {
      fds[i].revents = 0;
    }

    SYSCHECK_ERR_RETURN_NEG1(::poll(fds.data(), fds.size(), -1));

    // TCPStore's listening socket has an event and it should now be able to
    // accept new connections.
    if (fds[0].revents != 0) {
      if (fds[0].revents ^ POLLIN) {
        C10_THROW_ERROR(
            DistStoreError,
            "Unexpected poll revent on the master's listening socket: " +
                std::to_string(fds[0].revents));
      }
      Socket socket = storeListenSocket_.accept();
      int rawSocket = socket.handle();
      sockets_.emplace_back(std::move(socket));
      tcputil::addPollfd(fds, rawSocket, POLLIN);
      // all clients are miscellaneous before getting its validation query
      addMiscellaneousSocket(rawSocket);
    }

    // The pipe receives an event which tells us to shutdown the daemon
    if (fds[1].revents != 0) {
      // The main thread will write a byte to the pipe then close it before
      // joining the background thread
      if (fds[1].revents & ~(POLLIN | POLLHUP)) {
        C10_THROW_ERROR(
            DistStoreError,
            "Unexpected poll revent on the control pipe's reading fd: " +
                std::to_string(fds[1].revents));
      }
      finished = true;
      break;
    }
    queryFds(fds);
  }
}
#endif

std::unique_ptr<BackgroundThread> create_tcpstore_backend(
    const TCPStoreOptions& opts) {
  Socket socket = opts.masterListenFd.has_value()
      ? Socket::listenFromFd(*opts.masterListenFd, opts.port)
      : Socket::listen(opts.port);

  return std::make_unique<TCPStoreMasterDaemon>(std::move(socket));
}

} // namespace detail
} // namespace c10d
