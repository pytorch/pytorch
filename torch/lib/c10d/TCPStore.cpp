#include <c10d/TCPStore.hpp>

#ifdef _WIN32
#include <io.h>
#include <winsock2.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

#include <fcntl.h>
#include <algorithm>
#include <system_error>

namespace c10d {

namespace {

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
};

enum class CheckResponseType : uint8_t { READY, NOT_READY };

enum class WaitResponseType : uint8_t { STOP_WAITING };

} // anonymous namespace

// Background thread parent class methods
BackgroundThread::BackgroundThread(int storeListenSocket)
    : storeListenSocket_(storeListenSocket) {
  // Signal instance destruction to the daemon thread.
  initStopSignal();
}

BackgroundThread::~BackgroundThread() {
  // Stop the run
  stop();
  // Join the thread
  join();
  // Close unclosed sockets
  for (auto socket : sockets_) {
    if (socket != -1) {
      tcputil::closeSocket(socket);
    }
  }
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
    throw std::runtime_error(
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
    throw std::runtime_error(
        "Failed to create the control pipe to start the "
        "BackgroundThread run");
  }
}

void BackgroundThread::closeStopSignal() {
  for (auto fd : controlPipeFd_) {
    if (fd != -1) {
      ::close(fd);
    }
  }
}

void BackgroundThread::stop() {
  if (controlPipeFd_[1] != -1) {
    // close the write end of the pipe
    ::close(controlPipeFd_[1]);
    controlPipeFd_[1] = -1;
  }
}
#endif

// TCPStoreListener class methods
TCPStoreWorkerDaemon::TCPStoreWorkerDaemon(int listenSocket)
    : BackgroundThread(listenSocket) {
  daemonThread_ = std::thread(&TCPStoreWorkerDaemon::run, this);
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
  tcputil::addPollfd(fds, storeListenSocket_, POLLIN);

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
  tcputil::addPollfd(fds, controlPipeFd_[0], POLLHUP);
  tcputil::addPollfd(fds, storeListenSocket_, POLLIN);

  while (true) {
    SYSCHECK_ERR_RETURN_NEG1(::poll(fds.data(), fds.size(), -1));

    // Check control and exit early if triggered
    // The pipe receives an event which tells us to shutdown the listener thread
    if (fds[0].revents != 0) {
      // Will be POLLUP when the pipe is closed
      if (fds[0].revents ^ POLLHUP) {
        throw std::system_error(
            ECONNABORTED,
            std::system_category(),
            "Unexpected poll revent on the control pipe's reading fd: " +
                std::to_string(fds[0].revents));
      }
      break;
    }

    // if connection is closed gracefully by master, peeked data will return 0
    char data;
    int ret = recv(fds[1].fd, &data, 1, MSG_PEEK);
    if (ret == 0) {
      continue;
    }

    // valid request, perform callback logic
    callbackHandler(fds[1].fd);
  }
}
#endif

// TCPStoreMasterDaemon class methods
// Simply start the daemon thread
TCPStoreMasterDaemon::TCPStoreMasterDaemon(int storeListenSocket)
    : BackgroundThread(storeListenSocket) {
  daemonThread_ = std::thread(&TCPStoreMasterDaemon::run, this);
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
      tcputil::closeSocket(fds[fdIdx].fd);

      // Remove all the tracking state of the close FD
      for (auto it = waitingSockets_.begin(); it != waitingSockets_.end();) {
        for (auto vecIt = it->second.begin(); vecIt != it->second.end();) {
          if (*vecIt == fds[fdIdx].fd) {
            vecIt = it->second.erase(vecIt);
          } else {
            ++vecIt;
          }
        }
        if (it->second.size() == 0) {
          it = waitingSockets_.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = keysAwaited_.begin(); it != keysAwaited_.end();) {
        if (it->first == fds[fdIdx].fd) {
          it = keysAwaited_.erase(it);
        } else {
          ++it;
        }
      }
      fds.erase(fds.begin() + fdIdx);
      sockets_.erase(sockets_.begin() + fdIdx - CONNECT_SOCKET_OFFSET);
      --fdIdx;
      continue;
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

  } else {
    throw std::runtime_error("Unexpected query type");
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
    std::vector<uint8_t>& oldData,
    std::vector<uint8_t>& newData) {
  for (int socket : watchedSockets_[key]) {
    tcputil::sendValue<WatchResponseType>(socket, type);
    tcputil::sendString(socket, key, true);
    tcputil::sendVector<uint8_t>(socket, oldData);
    tcputil::sendVector<uint8_t>(socket, newData);
  }
}

void TCPStoreMasterDaemon::setHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  std::vector<uint8_t> newData = tcputil::recvVector<uint8_t>(socket);
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
  SizeType nargs;
  tcputil::recvBytes<SizeType>(socket, &nargs, 1);
  std::vector<std::string> keys(nargs);
  for (size_t i = 0; i < nargs; i++) {
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
  SizeType nargs;
  tcputil::recvBytes<SizeType>(socket, &nargs, 1);
  std::vector<std::string> keys(nargs);
  for (size_t i = 0; i < nargs; i++) {
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

bool TCPStoreMasterDaemon::checkKeys(
    const std::vector<std::string>& keys) const {
  return std::all_of(keys.begin(), keys.end(), [this](const std::string& s) {
    return tcpStore_.count(s) > 0;
  });
}

#ifdef _WIN32
void TCPStoreMasterDaemon::run() {
  std::vector<struct pollfd> fds;
  tcputil::addPollfd(fds, storeListenSocket_, POLLIN);

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (size_t i = 0; i < sockets_.size(); i++) {
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
      int sockFd = std::get<0>(tcputil::accept(storeListenSocket_));
      sockets_.push_back(sockFd);
      tcputil::addPollfd(fds, sockFd, POLLIN);
    }
    queryFds(fds);
  }
}
#else

void TCPStoreMasterDaemon::run() {
  std::vector<struct pollfd> fds;
  tcputil::addPollfd(fds, storeListenSocket_, POLLIN);
  // Push the read end of the pipe to signal the stopping of the daemon run
  tcputil::addPollfd(fds, controlPipeFd_[0], POLLHUP);

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (size_t i = 0; i < sockets_.size(); i++) {
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
      int sockFd = std::get<0>(tcputil::accept(storeListenSocket_));
      sockets_.push_back(sockFd);
      tcputil::addPollfd(fds, sockFd, POLLIN);
    }

    // The pipe receives an event which tells us to shutdown the daemon
    if (fds[1].revents != 0) {
      // Will be POLLUP when the pipe is closed
      if (fds[1].revents ^ POLLHUP) {
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

// TCPStore class methods
TCPStore::TCPStore(
    const std::string& masterAddr,
    PortType masterPort,
    c10::optional<int> numWorkers,
    bool isServer,
    const std::chrono::milliseconds& timeout,
    bool waitWorkers)
    : Store(timeout),
      isServer_(isServer),
      tcpStoreAddr_(masterAddr),
      tcpStorePort_(masterPort),
      numWorkers_(numWorkers),
      initKey_("init/"),
      regularPrefix_("/") {
  tcputil::socketInitialize();
  if (isServer_) {
    // Opening up the listening socket
    std::tie(masterListenSocket_, tcpStorePort_) = tcputil::listen(masterPort);
  }
  try {
    if (isServer_) {
      // Now start the daemon
      tcpStoreMasterDaemon_ =
          std::make_unique<TCPStoreMasterDaemon>(masterListenSocket_);
    }
    // Connect to the daemon
    storeSocket_ = tcputil::connect(
        tcpStoreAddr_, tcpStorePort_, /* wait= */ true, timeout_);
    if (numWorkers.value_or(-1) >= 0 && waitWorkers) {
      waitForWorkers();
    }

    // socket to handle requests from server
    listenSocket_ = tcputil::connect(
        tcpStoreAddr_, tcpStorePort_, /* wait= */ true, timeout_);
    tcpStoreWorkerDaemon_ =
        std::make_unique<TCPStoreWorkerDaemon>(listenSocket_);
  } catch (const std::exception&) {
    if (isServer_) {
      tcpStoreMasterDaemon_ = nullptr;
      tcputil::closeSocket(masterListenSocket_);
    }
    tcpStoreWorkerDaemon_ = nullptr;
    if (listenSocket_ != -1) {
      tcputil::closeSocket(listenSocket_);
    }
    if (storeSocket_ != -1) {
      tcputil::closeSocket(storeSocket_);
    }
    throw;
  }
}

TCPStore::~TCPStore() {
  if (isServer_) {
    // Store daemon should end because of closed connection.
    // daemon destructor should join the thread
    tcpStoreMasterDaemon_ = nullptr;
    tcputil::closeSocket(masterListenSocket_);
  }
  tcpStoreWorkerDaemon_ = nullptr;
  tcputil::closeSocket(listenSocket_);
  tcputil::closeSocket(storeSocket_);
}

void TCPStore::waitForWorkers() {
  addHelper_(initKey_, 1);
  // Let server block until all workers have completed, this ensures that
  // the server daemon thread is always running until the very end
  if (isServer_) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      std::vector<uint8_t> value = getHelper_(initKey_);
      auto buf = reinterpret_cast<const char*>(value.data());
      auto len = value.size();
      int numWorkersCompleted = std::stoi(std::string(buf, len));
      if (numWorkersCompleted >= numWorkers_.value_or(-1)) {
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
  std::string regKey = regularPrefix_ + key;
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::SET);
  tcputil::sendString(storeSocket_, regKey, true);
  tcputil::sendVector<uint8_t>(storeSocket_, data);
}

std::vector<uint8_t> TCPStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  std::string regKey = regularPrefix_ + key;
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::COMPARE_SET);
  tcputil::sendString(storeSocket_, regKey, true);
  tcputil::sendVector<uint8_t>(storeSocket_, expectedValue);
  tcputil::sendVector<uint8_t>(storeSocket_, desiredValue);
  return tcputil::recvVector<uint8_t>(storeSocket_);
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  std::string regKey = regularPrefix_ + key;
  return getHelper_(regKey);
}

std::vector<uint8_t> TCPStore::getHelper_(const std::string& key) {
  waitHelper_({key}, timeout_);
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::GET);
  tcputil::sendString(storeSocket_, key);
  return tcputil::recvVector<uint8_t>(storeSocket_);
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  std::string regKey = regularPrefix_ + key;
  return addHelper_(regKey, value);
}

bool TCPStore::deleteKey(const std::string& key) {
  std::string regKey = regularPrefix_ + key;
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::DELETE_KEY);
  tcputil::sendString(storeSocket_, regKey);
  auto numDeleted = tcputil::recvValue<int64_t>(storeSocket_);
  return (numDeleted == 1);
}

void TCPStore::watchKey(const std::string& key, WatchKeyCallback callback) {
  // Only allow one thread to perform watchKey() at a time
  const std::lock_guard<std::mutex> watchKeyLock(watchKeyMutex_);

  // Register callback with TCPStoreMasterDaemon to call TCPStoreWorkerDaemon on
  // key change
  std::string regKey = regularPrefix_ + key;
  tcpStoreWorkerDaemon_->setCallback(regKey, callback);
  tcputil::sendValue<QueryType>(listenSocket_, QueryType::WATCH_KEY);
  tcputil::sendString(listenSocket_, regKey);

  // Block until callback has been registered successfully
  tcpStoreWorkerDaemon_->waitForCallbackRegistration();
}

int64_t TCPStore::addHelper_(const std::string& key, int64_t value) {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::ADD);
  tcputil::sendString(storeSocket_, key, true);
  tcputil::sendValue<int64_t>(storeSocket_, value);
  return tcputil::recvValue<int64_t>(storeSocket_);
}

int64_t TCPStore::getNumKeys() {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::GETNUMKEYS);
  return tcputil::recvValue<int64_t>(storeSocket_);
}

bool TCPStore::check(const std::vector<std::string>& keys) {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::CHECK);
  SizeType nkeys = keys.size();
  tcputil::sendBytes<SizeType>(storeSocket_, &nkeys, 1, (nkeys > 0));
  for (size_t i = 0; i < nkeys; i++) {
    std::string regKey = regularPrefix_ + keys[i];
    tcputil::sendString(storeSocket_, regKey, (i != (nkeys - 1)));
  }
  auto checkResponse = tcputil::recvValue<CheckResponseType>(storeSocket_);
  if (checkResponse == CheckResponseType::READY) {
    return true;
  } else if (checkResponse == CheckResponseType::NOT_READY) {
    return false;
  } else {
    throw std::runtime_error("ready or not_ready response expected");
  }
}

void TCPStore::wait(const std::vector<std::string>& keys) {
  wait(keys, timeout_);
}

void TCPStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  std::vector<std::string> regKeys;
  regKeys.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    regKeys[i] = regularPrefix_ + keys[i];
  }
  waitHelper_(regKeys, timeout);
}

void TCPStore::waitHelper_(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  // Set the socket timeout if there is a wait timeout
  if (timeout != kNoTimeout) {
#ifdef _WIN32
    struct timeval timeoutTV = {
        timeout.count() / 1000, (timeout.count() % 1000) * 1000};
#else
    struct timeval timeoutTV = {
        .tv_sec = timeout.count() / 1000,
        .tv_usec = (timeout.count() % 1000) * 1000};
#endif
    SYSCHECK_ERR_RETURN_NEG1(::setsockopt(
        storeSocket_,
        SOL_SOCKET,
        SO_RCVTIMEO,
        reinterpret_cast<char*>(&timeoutTV),
        sizeof(timeoutTV)));
  }
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::WAIT);
  SizeType nkeys = keys.size();
  tcputil::sendBytes<SizeType>(storeSocket_, &nkeys, 1, (nkeys > 0));
  for (size_t i = 0; i < nkeys; i++) {
    tcputil::sendString(storeSocket_, keys[i], (i != (nkeys - 1)));
  }
  auto waitResponse = tcputil::recvValue<WaitResponseType>(storeSocket_);
  if (waitResponse != WaitResponseType::STOP_WAITING) {
    throw std::runtime_error("Stop_waiting response is expected");
  }
}

const std::string& TCPStore::getHost() const noexcept {
  return tcpStoreAddr_;
}

PortType TCPStore::getPort() const noexcept {
  return tcpStorePort_;
}

} // namespace c10d
