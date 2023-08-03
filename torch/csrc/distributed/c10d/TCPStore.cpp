#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#include <fcntl.h>
#include <algorithm>
#include <array>
#include <system_error>
#include <thread>
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
namespace {

// Abstract base class to handle thread state for TCPStoreMasterDaemon and
// TCPStoreWorkerDaemon. Contains the windows/unix implementations to signal a
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

enum class QueryType : uint8_t {
  SET,
  COMPARE_SET,
  GET,
  ADD,
  CHECK,
  WAIT,
  GETNUMKEYS,
  WATCH_KEY,
  DELETE_KEY,
  APPEND,
  MULTI_GET,
  MULTI_SET,
  CANCEL_WAIT,
};

enum class CheckResponseType : uint8_t { READY, NOT_READY };

enum class WaitResponseType : uint8_t { STOP_WAITING, WAIT_CANCELED };

enum class WatchResponseType : uint8_t {
  KEY_UPDATED,
  KEY_CREATED,
  KEY_DELETED,
  KEY_CALLBACK_REGISTERED,
  KEY_APPENDED,
};

// Separate thread that is only launched on master
class TCPStoreMasterDaemon : public BackgroundThread {
 public:
  explicit TCPStoreMasterDaemon(Socket&& storeListenSocket);

  ~TCPStoreMasterDaemon() override;

 private:
  void run();
  void queryFds(std::vector<struct pollfd>& fds);
  void query(int socket);
  void clearSocketWaitState(int socket);

  // The master runs on a single thread so only
  // one handler can be executed at a time
  void setHandler(int socket);
  void compareSetHandler(int socket);
  void addHandler(int socket);
  void getHandler(int socket) const;
  void checkHandler(int socket) const;
  void getNumKeysHandler(int socket) const;
  void deleteHandler(int socket);
  void waitHandler(int socket);
  void watchHandler(int socket);
  void appendHandler(int socket);
  void multiGetHandler(int socket);
  void multiSetHandler(int socket);
  void cancelWaitHandler(int socket);

  bool checkKeys(const std::vector<std::string>& keys) const;
  // Helper function to alerts waiting workers, used in setHandler, getHandler
  void wakeupWaitingClients(const std::string& key);
  // Helper function used when the key is changed
  // used in setHandler, addHandler, getHandler, deleteHandler
  void sendKeyUpdatesToClients(
      const std::string& key,
      const enum WatchResponseType& type,
      const std::vector<uint8_t>& oldData,
      const std::vector<uint8_t>& newData);
  void doSet(const std::string& key, const std::vector<uint8_t>& newData);

  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // From key -> the list of sockets waiting on the key
  std::unordered_map<std::string, std::vector<int>> waitingSockets_;
  // From socket -> number of keys awaited
  std::unordered_map<int, size_t> keysAwaited_;
  // From key -> the list of sockets watching the key
  std::unordered_map<std::string, std::vector<int>> watchedSockets_;
};

// Simply start the daemon thread
TCPStoreMasterDaemon::TCPStoreMasterDaemon(Socket&& storeListenSocket)
    : BackgroundThread{std::move(storeListenSocket)} {
  daemonThread_ = std::thread{&TCPStoreMasterDaemon::run, this};
}

TCPStoreMasterDaemon::~TCPStoreMasterDaemon() {
  dispose();
}

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
  if (qt == QueryType::SET) {
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

  } else if (qt == QueryType::WATCH_KEY) {
    watchHandler(socket);
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

void TCPStoreMasterDaemon::sendKeyUpdatesToClients(
    const std::string& key,
    const enum WatchResponseType& type,
    const std::vector<uint8_t>& oldData,
    const std::vector<uint8_t>& newData) {
  for (int socket : watchedSockets_[key]) {
    tcputil::sendValue<WatchResponseType>(socket, type);
    tcputil::sendString(socket, key, true);
    tcputil::sendVector<uint8_t>(socket, oldData);
    tcputil::sendVector<uint8_t>(socket, newData);
  }
}

void TCPStoreMasterDaemon::doSet(
    const std::string& key,
    const std::vector<uint8_t>& newData) {
  std::vector<uint8_t> oldData;
  bool newKey = true;
  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    oldData = it->second;
    newKey = false;
  }
  tcpStore_[key] = newData;
  // On "set", wake up all clients that have been waiting
  wakeupWaitingClients(key);
  // Send key update to all watching clients
  newKey ? sendKeyUpdatesToClients(
               key, WatchResponseType::KEY_CREATED, oldData, newData)
         : sendKeyUpdatesToClients(
               key, WatchResponseType::KEY_UPDATED, oldData, newData);
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

      // Send key update to all watching clients
      sendKeyUpdatesToClients(
          key, WatchResponseType::KEY_CREATED, currentValue, newValue);
      tcputil::sendVector<uint8_t>(socket, newValue);
    } else {
      // TODO: This code path is not ideal as we are "lying" to the caller in
      // case the key does not exist. We should come up with a working solution.
      tcputil::sendVector<uint8_t>(socket, currentValue);
    }
  } else {
    if (pos->second == currentValue) {
      pos->second = std::move(newValue);

      // Send key update to all watching clients
      sendKeyUpdatesToClients(
          key, WatchResponseType::KEY_UPDATED, currentValue, pos->second);
    }
    tcputil::sendVector<uint8_t>(socket, pos->second);
  }
}

void TCPStoreMasterDaemon::addHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  int64_t addVal = tcputil::recvValue<int64_t>(socket);

  bool newKey = true;
  std::vector<uint8_t> oldData;
  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    oldData = it->second;
    auto buf = reinterpret_cast<const char*>(it->second.data());
    auto len = it->second.size();
    addVal += std::stoll(std::string(buf, len));
    newKey = false;
  }
  auto addValStr = std::to_string(addVal);
  std::vector<uint8_t> newData =
      std::vector<uint8_t>(addValStr.begin(), addValStr.end());
  tcpStore_[key] = newData;
  // Now send the new value
  tcputil::sendValue<int64_t>(socket, addVal);
  // On "add", wake up all clients that have been waiting
  wakeupWaitingClients(key);
  // Send key update to all watching clients
  newKey ? sendKeyUpdatesToClients(
               key, WatchResponseType::KEY_CREATED, oldData, newData)
         : sendKeyUpdatesToClients(
               key, WatchResponseType::KEY_UPDATED, oldData, newData);
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
  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    std::vector<uint8_t> oldData = it->second;
    // Send key update to all watching clients
    std::vector<uint8_t> newData;
    sendKeyUpdatesToClients(
        key, WatchResponseType::KEY_DELETED, oldData, newData);
  }
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

void TCPStoreMasterDaemon::watchHandler(int socket) {
  std::string key = tcputil::recvString(socket);

  // Record the socket to respond to when the key is updated
  watchedSockets_[key].push_back(socket);

  // Send update to TCPStoreWorkerDaemon on client
  tcputil::sendValue<WatchResponseType>(
      socket, WatchResponseType::KEY_CALLBACK_REGISTERED);
}

void TCPStoreMasterDaemon::appendHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  std::vector<uint8_t> newData = tcputil::recvVector<uint8_t>(socket);
  bool newKey = true;
  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    it->second.insert(it->second.end(), newData.begin(), newData.end());
    newKey = false;
  } else {
    tcpStore_[key] = newData;
  }
  // we should not have clients waiting if we're appending, so it's all fine
  wakeupWaitingClients(key);
  // Send key update to all watching clients
  std::vector<uint8_t> oldData;
  newKey ? sendKeyUpdatesToClients(
               key, WatchResponseType::KEY_CREATED, oldData, newData)
         : sendKeyUpdatesToClients(
               key, WatchResponseType::KEY_APPENDED, oldData, newData);
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
        throw std::system_error(
            ECONNABORTED,
            std::system_category(),
            "Unexpected poll revent on the master's listening socket: " +
                std::to_string(fds[0].revents));
      }
      Socket socket = storeListenSocket_.accept();
      int rawSocket = socket.handle();
      sockets_.emplace_back(std::move(socket));
      tcputil::addPollfd(fds, rawSocket, POLLIN);
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
        throw std::system_error(
            ECONNABORTED,
            std::system_category(),
            "Unexpected poll revent on the master's listening socket: " +
                std::to_string(fds[0].revents));
      }
      Socket socket = storeListenSocket_.accept();
      int rawSocket = socket.handle();
      sockets_.emplace_back(std::move(socket));
      tcputil::addPollfd(fds, rawSocket, POLLIN);
    }

    // The pipe receives an event which tells us to shutdown the daemon
    if (fds[1].revents != 0) {
      // The main thread will write a byte to the pipe then close it before
      // joining the background thread
      if (fds[1].revents & ~(POLLIN | POLLHUP)) {
        throw std::system_error(
            ECONNABORTED,
            std::system_category(),
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

// Separate thread that is launched on all instances (including master)
// Right now only handles callbacks registered from watchKey()
class TCPStoreWorkerDaemon : public BackgroundThread {
 public:
  explicit TCPStoreWorkerDaemon(Socket&& listenSocket);
  ~TCPStoreWorkerDaemon() override;
  // Set the callback to run key change
  void setCallback(std::string key, WatchKeyCallback cb);
  void waitForCallbackRegistration() {
    // Block until callback has been registered successfully
    std::unique_lock<std::mutex> callbackRegistrationLock(
        callbackRegistrationMutex_);
    callbackRegisteredCV_.wait(
        callbackRegistrationLock, [&] { return callbackRegisteredData_; });

    // Reset payload for next callback
    callbackRegisteredData_ = false;
  }
  void setCallbackRegistered() {
    {
      std::unique_lock<std::mutex> callbackRegistrationLock(
          callbackRegistrationMutex_);
      callbackRegisteredData_ = true;
    }
    callbackRegisteredCV_.notify_one();
  }

 private:
  void run();
  void callbackHandler(int socket);
  // List of callbacks map each watched key
  std::unordered_map<std::string, WatchKeyCallback> keyToCallbacks_{};
  std::mutex keyToCallbacksMutex_{};
  std::mutex callbackRegistrationMutex_{};
  std::condition_variable callbackRegisteredCV_{};
  bool callbackRegisteredData_ = false;
};

// TCPStoreListener class methods
TCPStoreWorkerDaemon::TCPStoreWorkerDaemon(Socket&& listenSocket)
    : BackgroundThread{std::move(listenSocket)} {
  daemonThread_ = std::thread{&TCPStoreWorkerDaemon::run, this};
}

TCPStoreWorkerDaemon::~TCPStoreWorkerDaemon() {
  dispose();
}

void TCPStoreWorkerDaemon::setCallback(
    std::string key,
    WatchKeyCallback callback) {
  const std::lock_guard<std::mutex> lock(keyToCallbacksMutex_);
  keyToCallbacks_[key] = callback;
}

// Runs all the callbacks that the worker has registered
void TCPStoreWorkerDaemon::callbackHandler(int socket) {
  auto watchResponse = tcputil::recvValue<WatchResponseType>(socket);
  if (watchResponse == WatchResponseType::KEY_CALLBACK_REGISTERED) {
    // Notify the waiting "watchKey" operation to return
    setCallbackRegistered();
    return;
  }
  std::string key = tcputil::recvString(socket);
  std::vector<uint8_t> currentValueVec = tcputil::recvVector<uint8_t>(socket);
  std::vector<uint8_t> newValueVec = tcputil::recvVector<uint8_t>(socket);
  c10::optional<std::string> currentValue;
  if (watchResponse == WatchResponseType::KEY_CREATED) {
    assert(currentValueVec.empty());
    currentValue = c10::nullopt;
  } else {
    currentValue = std::string(currentValueVec.begin(), currentValueVec.end());
  }
  c10::optional<std::string> newValue;
  if (watchResponse == WatchResponseType::KEY_DELETED) {
    assert(newValueVec.empty());
    newValue = c10::nullopt;
  } else {
    newValue = std::string(newValueVec.begin(), newValueVec.end());
  }
  const std::lock_guard<std::mutex> lock(keyToCallbacksMutex_);
  keyToCallbacks_.at(key)(currentValue, newValue);
}

#ifdef _WIN32
void TCPStoreWorkerDaemon::run() {
  std::vector<struct pollfd> fds;
  tcputil::addPollfd(fds, storeListenSocket_.handle(), POLLIN);

  while (true) {
    // Check control and exit early if triggered
    int res;
    SYSCHECK_ERR_RETURN_NEG1(
        res = WSAPoll(fds.data(), fds.size(), checkTimeout_.count()))
    if (res == 0) {
      auto rvPoll = WaitForSingleObject(ghStopEvent_, 0);
      if (rvPoll != WAIT_TIMEOUT) {
        break;
      }
      continue;
    }

    // if connection is closed gracefully by master, peeked data will return 0
    char data;
    int ret = recv(fds[0].fd, &data, 1, MSG_PEEK);
    if (ret == 0) {
      auto rvData = WaitForSingleObject(ghStopEvent_, 0);
      if (rvData != WAIT_TIMEOUT) {
        break;
      }
      continue;
    }

    // valid request, perform callback logic
    callbackHandler(fds[0].fd);
  }
}
#else
void TCPStoreWorkerDaemon::run() {
  std::vector<struct pollfd> fds;
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
  tcputil::addPollfd(fds, storeListenSocket_.handle(), POLLIN);

  while (true) {
    SYSCHECK_ERR_RETURN_NEG1(::poll(fds.data(), fds.size(), -1));

    // Check control and exit early if triggered
    // The pipe receives an event which tells us to shutdown the listener thread
    if (fds[0].revents != 0) {
      // The main thread will write a byte to the pipe then close it before
      // joining the background thread
      if (fds[0].revents & ~(POLLIN | POLLHUP)) {
        throw std::system_error(
            ECONNABORTED,
            std::system_category(),
            "Unexpected poll revent on the control pipe's reading fd: " +
                std::to_string(fds[0].revents));
      }
      break;
    }

    // if connection is closed gracefully by master, peeked data will return 0
    char data = 0;
    int ret = recv(fds[1].fd, &data, 1, MSG_PEEK);
    if (ret == 0) {
      continue;
    }

    // valid request, perform callback logic
    callbackHandler(fds[1].fd);
  }
}
#endif

} // namespace

// Manages the lifecycle of a server daemon.
class TCPServer {
 public:
  static std::shared_ptr<TCPServer> start(const TCPStoreOptions& opts);

  std::uint16_t port() const noexcept {
    return port_;
  }

  explicit TCPServer(
      std::uint16_t port,
      std::unique_ptr<TCPStoreMasterDaemon>&& daemon)
      : port_{port}, daemon_{std::move(daemon)} {}

 private:
  std::uint16_t port_;
  std::unique_ptr<TCPStoreMasterDaemon> daemon_;

  // We store weak references to all TCPServers for which the caller requested
  // multi-tenancy.
  static std::unordered_map<std::uint16_t, std::weak_ptr<TCPServer>>
      cachedServers_;

  static std::mutex cache_mutex_;
};

std::unordered_map<std::uint16_t, std::weak_ptr<TCPServer>>
    TCPServer::cachedServers_{};

std::mutex TCPServer::cache_mutex_{};

std::shared_ptr<TCPServer> TCPServer::start(const TCPStoreOptions& opts) {
  auto startCore = [&opts]() {
    Socket socket = opts.masterListenFd.has_value()
        ? Socket::listenFromFd(*opts.masterListenFd, opts.port)
        : Socket::listen(opts.port);

    std::uint16_t port = socket.port();

    auto daemon = std::make_unique<TCPStoreMasterDaemon>(std::move(socket));

    return std::make_shared<TCPServer>(port, std::move(daemon));
  };

  std::shared_ptr<TCPServer> server{};

  if (opts.multiTenant) {
    std::lock_guard<std::mutex> guard{cache_mutex_};

    // If the caller is okay with a multi-tenant store, first check if we
    // already have a TCPServer running on the specified port.
    if (opts.port > 0) {
      auto pos = cachedServers_.find(opts.port);
      if (pos != cachedServers_.end()) {
        server = pos->second.lock();
        if (server != nullptr) {
          return server;
        }

        // Looks like the TCPStore has been disposed, make sure that we release
        // the control block.
        cachedServers_.erase(pos);
      }
    }

    server = startCore();

    cachedServers_.emplace(server->port(), server);
  } else {
    server = startCore();
  }

  return server;
}

class TCPClient {
 public:
  static std::unique_ptr<TCPClient> connect(
      const SocketAddress& addr,
      const TCPStoreOptions& opts);

  void sendRaw(uint8_t* data, size_t lenght) {
    tcputil::sendBytes(socket_.handle(), data, lenght);
  }

  std::vector<std::uint8_t> receiveBits() {
    return tcputil::recvVector<std::uint8_t>(socket_.handle());
  }

  template <typename T>
  T receiveValue() {
    return tcputil::recvValue<T>(socket_.handle());
  }
  template <typename T>
  bool receiveValueWithTimeout(T& t, std::chrono::milliseconds timeout) {
    if (!socket_.waitForInput(timeout))
      return false;
    t = tcputil::recvValue<T>(socket_.handle());
    return true;
  }
  void setTimeout(std::chrono::milliseconds value);

  explicit TCPClient(Socket&& socket) : socket_{std::move(socket)} {}

 private:
  Socket socket_;
};

std::unique_ptr<TCPClient> TCPClient::connect(
    const SocketAddress& addr,
    const TCPStoreOptions& opts) {
  auto timeout = std::chrono::duration_cast<std::chrono::seconds>(opts.timeout);
  Socket socket = Socket::connect(
      addr.host, addr.port, SocketOptions{}.connect_timeout(timeout));

  return std::make_unique<TCPClient>(std::move(socket));
}

void TCPClient::setTimeout(std::chrono::milliseconds value) {
  if (value == std::chrono::milliseconds::zero()) {
    return;
  }

#ifdef _WIN32
  struct timeval timeoutTV = {
      static_cast<long>(value.count() / 1000),
      static_cast<long>((value.count() % 1000) * 1000)};
#else
  struct timeval timeoutTV = {
      .tv_sec = value.count() / 1000,
      .tv_usec = static_cast<suseconds_t>((value.count() % 1000) * 1000),
  };
#endif
  SYSCHECK_ERR_RETURN_NEG1(::setsockopt(
      socket_.handle(),
      SOL_SOCKET,
      SO_RCVTIMEO,
      reinterpret_cast<char*>(&timeoutTV),
      sizeof(timeoutTV)));
}

class TCPCallbackClient {
 public:
  static std::unique_ptr<TCPCallbackClient> connect(
      const SocketAddress& addr,
      const TCPStoreOptions& opts);

  void setCallback(const std::string& key, WatchKeyCallback callback);

  explicit TCPCallbackClient(
      int rawSocket,
      std::unique_ptr<TCPStoreWorkerDaemon>&& daemon)
      : rawSocket_{rawSocket}, daemon_{std::move(daemon)} {}

 private:
  int rawSocket_;
  std::unique_ptr<TCPStoreWorkerDaemon> daemon_;
  std::mutex mutex_;
};

std::unique_ptr<TCPCallbackClient> TCPCallbackClient::connect(
    const SocketAddress& addr,
    const TCPStoreOptions& opts) {
  auto timeout = std::chrono::duration_cast<std::chrono::seconds>(opts.timeout);
  Socket socket = Socket::connect(
      addr.host, addr.port, SocketOptions{}.connect_timeout(timeout));

  int rawSocket = socket.handle();

  auto daemon = std::make_unique<TCPStoreWorkerDaemon>(std::move(socket));

  return std::make_unique<TCPCallbackClient>(rawSocket, std::move(daemon));
}

void TCPCallbackClient::setCallback(
    const std::string& key,
    WatchKeyCallback callback) {
  std::lock_guard<std::mutex> guard{mutex_};

  daemon_->setCallback(key, callback);

  tcputil::sendValue<QueryType>(rawSocket_, QueryType::WATCH_KEY);

  tcputil::sendString(rawSocket_, key);

  daemon_->waitForCallbackRegistration();
}

class SendBuffer {
  // ethernet mtu 1500 - 40 (ip v6 header) - 20 (tcp header)
  const size_t FLUSH_WATERMARK = 1440;
  std::vector<uint8_t> buffer;
  detail::TCPClient& client;

  void maybeFlush() {
    if (buffer.size() >= FLUSH_WATERMARK) {
      flush();
    }
  }

 public:
  SendBuffer(detail::TCPClient& client, detail::QueryType cmd)
      : client(client) {
    buffer.reserve(32); // enough for most commands
    buffer.push_back((uint8_t)cmd);
  }

  void appendString(const std::string& str) {
    appendValue<uint64_t>(str.size());
    buffer.insert(buffer.end(), str.begin(), str.end());
    maybeFlush();
  }

  void appendBytes(const std::vector<uint8_t>& vec) {
    appendValue<uint64_t>(vec.size());
    buffer.insert(buffer.end(), vec.begin(), vec.end());
    maybeFlush();
  }

  template <typename T>
  void appendValue(T value) {
    uint8_t* begin = (uint8_t*)&value;
    buffer.insert(buffer.end(), begin, begin + sizeof(T));
    maybeFlush();
  }

  void flush() {
    if (buffer.size() > 0) {
      client.sendRaw(buffer.data(), buffer.size());
      buffer.clear();
    }
  }
};

} // namespace detail

using detail::Socket;

// TCPStore class methods
TCPStore::TCPStore(
    const std::string& masterAddr,
    std::uint16_t masterPort,
    c10::optional<int> numWorkers,
    bool isServer,
    const std::chrono::milliseconds& timeout,
    bool waitWorkers)
    : TCPStore{
          masterAddr,
          TCPStoreOptions{
              masterPort,
              isServer,
              numWorkers ? c10::optional<std::size_t>(*numWorkers)
                         : c10::nullopt,
              waitWorkers,
              timeout}} {}

TCPStore::TCPStore(std::string host, const TCPStoreOptions& opts)
    : Store{opts.timeout},
      addr_{std::move(host)},
      numWorkers_{opts.numWorkers} {
  Socket::initialize();

  if (opts.isServer) {
    server_ = detail::TCPServer::start(opts);
    // server successfully started
    C10D_DEBUG("The server has started on port = {}.", server_->port());

    addr_.port = server_->port();
  } else {
    addr_.port = opts.port;
  }

  client_ = detail::TCPClient::connect(addr_, opts);
  // TCP connection established
  C10D_DEBUG("TCP client connected to host {}:{}", addr_.host, addr_.port);

  if (opts.waitWorkers) {
    waitForWorkers();
  }

  callbackClient_ = detail::TCPCallbackClient::connect(addr_, opts);
  // TCP CallbackClient connection established
  C10D_DEBUG(
      "TCP callback client connected to host {}:{}", addr_.host, addr_.port);
}

TCPStore::~TCPStore() = default;

void TCPStore::waitForWorkers() {
  if (numWorkers_ == c10::nullopt) {
    return;
  }

  incrementValueBy(initKey_, 1);

  // Let server block until all workers have completed, this ensures that
  // the server daemon thread is always running until the very end
  if (server_) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      // TODO: Any chance to make this cleaner?
      std::vector<uint8_t> value = doGet(initKey_);
      auto buf = reinterpret_cast<const char*>(value.data());
      auto len = value.size();
      int numWorkersCompleted = std::stoi(std::string(buf, len));
      if (numWorkersCompleted >= static_cast<int>(*numWorkers_)) {
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (timeout_ != kNoTimeout && elapsed > timeout_) {
        break;
      }
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<uint8_t> TCPStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::COMPARE_SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(expectedValue);
  buffer.appendBytes(desiredValue);
  buffer.flush();

  return client_->receiveBits();
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  return doGet(keyPrefix_ + key);
}

std::vector<uint8_t> TCPStore::doGet(const std::string& key) {
  doWait(key, timeout_);
  detail::SendBuffer buffer(*client_, detail::QueryType::GET);
  buffer.appendString(key);
  buffer.flush();

  return client_->receiveBits();
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  return incrementValueBy(keyPrefix_ + key, value);
}

bool TCPStore::deleteKey(const std::string& key) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::DELETE_KEY);
  buffer.appendString(keyPrefix_ + key);
  buffer.flush();

  auto numDeleted = client_->receiveValue<std::int64_t>();
  return numDeleted == 1;
}

void TCPStore::watchKey(const std::string& key, WatchKeyCallback callback) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  callbackClient_->setCallback(keyPrefix_ + key, callback);
}

int64_t TCPStore::incrementValueBy(const std::string& key, int64_t delta) {
  detail::SendBuffer buff(*client_, detail::QueryType::ADD);
  buff.appendString(key);
  buff.appendValue<std::int64_t>(delta);
  buff.flush();

  return client_->receiveValue<std::int64_t>();
}

int64_t TCPStore::getNumKeys() {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::GETNUMKEYS);
  buffer.flush();

  return client_->receiveValue<std::int64_t>();
}

bool TCPStore::check(const std::vector<std::string>& keys) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::CHECK);
  buffer.appendValue(keys.size());

  for (const std::string& key : keys) {
    buffer.appendString(keyPrefix_ + key);
  }
  buffer.flush();

  auto response = client_->receiveValue<detail::CheckResponseType>();
  if (response == detail::CheckResponseType::READY) {
    return true;
  }
  if (response == detail::CheckResponseType::NOT_READY) {
    return false;
  }
  TORCH_CHECK(false, "ready or not_ready response expected");
}

void TCPStore::wait(const std::vector<std::string>& keys) {
  wait(keys, timeout_);
}

void TCPStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  std::vector<std::string> prefixedKeys{};
  prefixedKeys.reserve(keys.size());
  for (const std::string& key : keys) {
    prefixedKeys.emplace_back(keyPrefix_ + key);
  }

  doWait(prefixedKeys, timeout);
}

void TCPStore::doWait(
    c10::ArrayRef<std::string> keys,
    std::chrono::milliseconds timeout) {
  {
    detail::SendBuffer buffer(*client_, detail::QueryType::WAIT);
    buffer.appendValue(keys.size());
    for (const std::string& key : keys) {
      buffer.appendString(key);
    }
    buffer.flush();
  }

  detail::WaitResponseType response;
  if (client_->receiveValueWithTimeout<detail::WaitResponseType>(
          response, timeout)) {
    if (response != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK(false, "Stop_waiting response is expected");
    }
    return;
  }
  // this is the cancel wait timeout, once here we expect the server to respond
  // in a timely fashion
  {
    detail::SendBuffer buffer(*client_, detail::QueryType::CANCEL_WAIT);
    buffer.flush();
  }

  response = client_->receiveValue<detail::WaitResponseType>();
  // this can happen if the server responds before we cancel, just ignore it
  if (response != detail::WaitResponseType::WAIT_CANCELED) {
    if (response != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK(false, "Stop_waiting response is expected");
    }

    response = client_->receiveValue<detail::WaitResponseType>(); // ignore
    if (response != detail::WaitResponseType::WAIT_CANCELED) {
      TORCH_CHECK(false, "wait_canceled response is expected");
    }
  }
  TORCH_CHECK(false, "Socket Timeout");
}

void TCPStore::append(
    const std::string& key,
    const std::vector<uint8_t>& data) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::APPEND);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<std::vector<uint8_t>> TCPStore::multiGet(
    const std::vector<std::string>& keys) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  std::vector<std::string> prefixedKeys;
  prefixedKeys.reserve(keys.size());
  for (const std::string& key : keys) {
    prefixedKeys.emplace_back(keyPrefix_ + key);
  }
  doWait(prefixedKeys, timeout_);

  detail::SendBuffer buffer(*client_, detail::QueryType::MULTI_GET);
  buffer.appendValue(keys.size());
  for (auto& key : prefixedKeys) {
    buffer.appendString(key);
  }
  buffer.flush();

  std::vector<std::vector<uint8_t>> result;
  result.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    result.emplace_back(client_->receiveBits());
  }
  return result;
}

void TCPStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  TORCH_CHECK(
      keys.size() == values.size(),
      "multiSet keys and values vectors must be of same size");
  const std::lock_guard<std::mutex> lock(activeOpLock_);

  detail::SendBuffer buffer(*client_, detail::QueryType::MULTI_SET);
  buffer.appendValue<std::int64_t>(keys.size());
  for (auto i : c10::irange(keys.size())) {
    buffer.appendString(keyPrefix_ + keys[i]);
    buffer.appendBytes(values[i]);
  }
  buffer.flush();
}

bool TCPStore::hasExtendedApi() const {
  return true;
}

} // namespace c10d
