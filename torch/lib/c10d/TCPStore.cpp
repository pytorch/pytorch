#include "TCPStore.hpp"

#include <poll.h>

#include <unistd.h>
#include <algorithm>
#include <system_error>

namespace c10d {

namespace {

enum class QueryType : uint8_t { SET, GET, ADD, CHECK, WAIT };

enum class CheckResponseType : uint8_t { READY, NOT_READY };

enum class WaitResponseType : uint8_t { STOP_WAITING };

} // anonymous namespace

// TCPStoreDaemon class methods
// Simply start the daemon thread
TCPStoreDaemon::TCPStoreDaemon(int storeListenSocket)
    : storeListenSocket_(storeListenSocket) {
  // Use control pipe to signal instance destruction to the daemon thread.
  if (pipe(controlPipeFd_.data()) == -1) {
    throw std::runtime_error(
        "Failed to create the control pipe to start the "
        "TCPStoreDaemon run");
  }
  daemonThread_ = std::thread(&TCPStoreDaemon::run, this);
}

TCPStoreDaemon::~TCPStoreDaemon() {
  // Stop the run
  stop();
  // Join the thread
  join();
  // Close unclosed sockets
  for (auto socket : sockets_) {
    if (socket != -1) {
      ::close(socket);
    }
  }
  // Now close the rest control pipe
  for (auto fd : controlPipeFd_) {
    if (fd != -1) {
      ::close(fd);
    }
  }
}

void TCPStoreDaemon::join() {
  daemonThread_.join();
}

void TCPStoreDaemon::run() {
  std::vector<struct pollfd> fds;
  fds.push_back({.fd = storeListenSocket_, .events = POLLIN});
  // Push the read end of the pipe to signal the stopping of the daemon run
  fds.push_back({.fd = controlPipeFd_[0], .events = POLLHUP});

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (size_t i = 0; i < sockets_.size(); i++) {
      fds[i].revents = 0;
    }

    SYSCHECK(::poll(fds.data(), fds.size(), -1));

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
      fds.push_back({.fd = sockFd, .events = POLLIN});
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
    // Skipping the fds[0] and fds[1],
    // fds[0] is master's listening socket
    // fds[1] is control pipe's reading fd
    for (size_t fdIdx = 2; fdIdx < fds.size(); ++fdIdx) {
      if (fds[fdIdx].revents == 0) {
        continue;
      }

      if (fds[fdIdx].revents ^ POLLIN) {
        throw std::system_error(
            ECONNABORTED,
            std::system_category(),
            "Unexpected poll revent: " + std::to_string(fds[fdIdx].revents) +
                " on socket: " + std::to_string(fds[fdIdx].fd));
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
        ::close(fds[fdIdx].fd);

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
        sockets_.erase(sockets_.begin() + fdIdx - 2);
        --fdIdx;
        continue;
      }
    }
  }
}

void TCPStoreDaemon::stop() {
  if (controlPipeFd_[1] != -1) {
    // close the write end of the pipe
    ::close(controlPipeFd_[1]);
    controlPipeFd_[1] = -1;
  }
}

// query communicates with the worker. The format
// of the query is as follows:
// type of query | size of arg1 | arg1 | size of arg2 | arg2 | ...
// or, in the case of wait
// type of query | number of args | size of arg1 | arg1 | ...
void TCPStoreDaemon::query(int socket) {
  QueryType qt;
  tcputil::recvBytes<QueryType>(socket, &qt, 1);

  if (qt == QueryType::SET) {
    setHandler(socket);

  } else if (qt == QueryType::ADD) {
    addHandler(socket);

  } else if (qt == QueryType::GET) {
    getHandler(socket);

  } else if (qt == QueryType::CHECK) {
    checkHandler(socket);

  } else if (qt == QueryType::WAIT) {
    waitHandler(socket);

  } else {
    throw std::runtime_error("Unexpected query type");
  }
}

void TCPStoreDaemon::wakeupWaitingClients(const std::string& key) {
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

void TCPStoreDaemon::setHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  tcpStore_[key] = tcputil::recvVector<uint8_t>(socket);
  // On "set", wake up all clients that have been waiting
  wakeupWaitingClients(key);
}

void TCPStoreDaemon::addHandler(int socket) {
  std::string key = tcputil::recvString(socket);
  int64_t addVal = tcputil::recvValue<int64_t>(socket);

  if (tcpStore_.find(key) != tcpStore_.end()) {
    auto buf = reinterpret_cast<const char*>(tcpStore_[key].data());
    auto len = tcpStore_[key].size();
    addVal += std::stoll(std::string(buf, len));
  }
  auto addValStr = std::to_string(addVal);
  tcpStore_[key] = std::vector<uint8_t>(addValStr.begin(), addValStr.end());
  // Now send the new value
  tcputil::sendValue<int64_t>(socket, addVal);
  // On "add", wake up all clients that have been waiting
  wakeupWaitingClients(key);
}

void TCPStoreDaemon::getHandler(int socket) const {
  std::string key = tcputil::recvString(socket);
  auto data = tcpStore_.at(key);
  tcputil::sendVector<uint8_t>(socket, data);
}

void TCPStoreDaemon::checkHandler(int socket) const {
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

void TCPStoreDaemon::waitHandler(int socket) {
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
    for (auto& key : keys) {
      waitingSockets_[key].push_back(socket);
    }
    keysAwaited_[socket] = keys.size();
  }
}

bool TCPStoreDaemon::checkKeys(const std::vector<std::string>& keys) const {
  return std::all_of(keys.begin(), keys.end(), [this](const std::string& s) {
    return tcpStore_.count(s) > 0;
  });
}

// TCPStore class methods
TCPStore::TCPStore(
    const std::string& masterAddr,
    PortType masterPort,
    bool isServer)
    : isServer_(isServer),
      tcpStoreAddr_(masterAddr),
      tcpStorePort_(masterPort) {
  if (isServer_) {
    // Opening up the listening socket
    std::tie(masterListenSocket_, std::ignore) = tcputil::listen(masterPort);
    // Now start the daemon
    tcpStoreDaemon_ = std::unique_ptr<TCPStoreDaemon>(
        new TCPStoreDaemon(masterListenSocket_));
  }
  // Connect to the daemon
  storeSocket_ = tcputil::connect(tcpStoreAddr_, tcpStorePort_);
}

TCPStore::~TCPStore() {
  ::close(storeSocket_);
  if (isServer_) {
    // Store daemon should end because of closed connection.
    // daemon destructor should join the thread
    tcpStoreDaemon_.reset(nullptr);
    ::close(masterListenSocket_);
  }
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::SET);
  tcputil::sendString(storeSocket_, key, true);
  tcputil::sendVector<uint8_t>(storeSocket_, data);
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  wait({key});
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::GET);
  tcputil::sendString(storeSocket_, key);
  return tcputil::recvVector<uint8_t>(storeSocket_);
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::ADD);
  tcputil::sendString(storeSocket_, key, true);
  tcputil::sendValue<int64_t>(storeSocket_, value);
  return tcputil::recvValue<int64_t>(storeSocket_);
}

bool TCPStore::check(const std::vector<std::string>& keys) {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::CHECK);
  SizeType nkeys = keys.size();
  tcputil::sendBytes<SizeType>(storeSocket_, &nkeys, 1, (nkeys > 0));
  for (size_t i = 0; i < nkeys; i++) {
    tcputil::sendString(storeSocket_, keys[i], (i != (nkeys - 1)));
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
  // Set the socket timeout if there is a wait timeout
  if (timeout != kNoTimeout) {
    struct timeval timeoutTV = {.tv_sec = timeout.count() / 1000,
                                .tv_usec = (timeout.count() % 1000) * 1000};
    SYSCHECK(::setsockopt(
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

} // namespace c10d
