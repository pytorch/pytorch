#include "TcpStore.hpp"

#include <poll.h>
#include <system_error>
#include <unistd.h>


namespace c10d {


namespace {

enum class QueryType : std::uint8_t {
  SET,
  GET,
  ADD,
  CHECK,
  STOP_WAITING,
  KEEP_WAITING
};

} // anonymous namespace


// TcpStoreDaemon class methods

// Simply start the daemon thread
TcpStoreDaemon::TcpStoreDaemon(int storeListenSocket) :
  storeListenSocket_(storeListenSocket)
{
  daemonThread_ = std::thread(&TcpStoreDaemon::run, this);
}

TcpStoreDaemon::~TcpStoreDaemon() {
  for (auto socket : sockets_) {
    if (socket != -1) {
      ::close(socket);
    }
  }
  // Join the thread
  join();
}

void TcpStoreDaemon::join() {
  daemonThread_.join();
}

void TcpStoreDaemon::run() {

  std::vector<struct pollfd> fds;
  fds.push_back({ .fd = storeListenSocket_, .events = POLLIN });

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (size_t i = 0; i < sockets_.size(); i++) {
      fds[i].revents = 0;
    }

    SYSCHECK(::poll(fds.data(), fds.size(), -1));

    if (fds[0].revents != 0) {
      if (fds[0].revents ^ POLLIN) {
        throw std::system_error(ECONNABORTED, std::system_category());
      }
      int sockFd = std::get<0>(tcputil::accept(storeListenSocket_));
      sockets_.push_back(sockFd);
      keysAwaited_.push_back(0);
      fds.push_back({ .fd = sockFd, .events = POLLIN });
    }
    for (size_t rank = 0; rank < sockets_.size(); rank++) {
      if (fds[rank + 1].revents == 0) {
        continue;
      }

      if (fds[rank + 1].revents ^ POLLIN) {
        throw std::system_error(ECONNABORTED, std::system_category());
      }
      try {
        query(rank);
      } catch (std::exception& ex) {
        /**
         * There was an error when processing query. Probably an exception
         * occurred in recv/send what would indicate that socket on the other
         * side has been closed. If the closing was due to normal exit, then the
         * store should exit too. Otherwise, if it was different exception,
         * other processes will get an exception once they try to use the store.
         */
        finished = true;
        break;
      }
    }
  }
}

void TcpStoreDaemon::wakeUpWaitingRanks(const std::string& key) {
  auto toWake = waiting_.find(key);
  if (toWake != waiting_.end()) {
    for (int proc : toWake->second) {
      if (--keysAwaited_[proc] == 0) {
        tcputil::sendValue<QueryType>(sockets_[proc],
                                      QueryType::STOP_WAITING);
      }
    }
    waiting_.erase(toWake);
  }
}

/**
 * query communicates with the worker. The format
 * of the query is as follows:
 * type of query | size of arg1 | arg1 | size of arg2 | arg2 | ...
 * or, in the case of wait
 * type of query | number of args | size of arg1 | arg1 | ...
 */
void TcpStoreDaemon::query(RankType rank) {

  int socket = sockets_[rank];
  QueryType qt;
  tcputil::recvBytes<QueryType>(socket, &qt, 1);

  if (qt == QueryType::SET) {
    std::string key = tcputil::recvString(socket);
    tcpStore_[key] = tcputil::recvVector<uint8_t>(socket);
    // On "set", wake up all of the processes that wait
    // for keys already in the store
    wakeUpWaitingRanks(key);

  } else if (qt == QueryType::ADD) {
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
    // On "add", wake up all of the processes that wait
    // for keys already in the store
    wakeUpWaitingRanks(key);

  } else if (qt == QueryType::GET) {
    std::string key = tcputil::recvString(socket);
    auto data = tcpStore_.at(key);
    tcputil::sendVector<uint8_t>(socket, data);

  } else if (qt == QueryType::CHECK) {
    SizeType nargs;
    tcputil::recvBytes<SizeType>(socket, &nargs, 1);
    std::vector<std::string> keys(nargs);
    for (size_t i = 0; i < nargs; i++) {
      keys[i] = tcputil::recvString(socket);
    }
    // Now we have received all the keys
    if (checkAndUpdate(keys)) {
      tcputil::sendValue<QueryType>(socket, QueryType::STOP_WAITING);
    } else {
      for (auto& key : keys) {
        waiting_[key].push_back(rank);
      }
      keysAwaited_[rank] = keys.size();
      tcputil::sendValue<QueryType>(socket, QueryType::KEEP_WAITING);
    }
  } else {
    throw std::runtime_error("expected a query type");
  }
}

bool TcpStoreDaemon::checkAndUpdate(std::vector<std::string>& keys) const {
  bool ret = true;
  for (auto it = keys.begin(); it != keys.end();) {
    if (tcpStore_.count(*it) == 0) {
      ret = false;
      it++;
    } else {
      it = keys.erase(it);
    }
  }
  return ret;
}

// TcpStore class methods

TcpStore::TcpStore(const std::string& masterAddr,
                   PortType masterPort,
                   bool isServer)
 : isServer_(isServer)
 , tcpStoreAddr_(masterAddr)
 , tcpStorePort_(masterPort)

{
  if (isServer_) {
    // Openning up the listening socket
    std::tie(masterListenSocket_, std::ignore) = tcputil::listen(masterPort);
    // Now start the daemon
    tcpStoreDaemon_ = std::unique_ptr<TcpStoreDaemon>(
        new TcpStoreDaemon(masterListenSocket_)
    );
  }
  // Connect to the daemon
  storeSocket_ = tcputil::connect(tcpStoreAddr_, tcpStorePort_);
}

TcpStore::~TcpStore() {
  ::close(storeSocket_);
  if (isServer_) {
    ::close(masterListenSocket_);
    /**
     * Store daemon should end because of closed connection.
     * daemon destructor should join the thread
     */
    tcpStoreDaemon_.reset(nullptr);
  }
}

void TcpStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::SET);
  tcputil::sendString(storeSocket_, key, true);
  tcputil::sendVector<uint8_t>(storeSocket_, data);
}

std::vector<uint8_t> TcpStore::get(const std::string& key) {
  wait({key});
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::GET);
  tcputil::sendString(storeSocket_, key);
  return tcputil::recvVector<uint8_t>(storeSocket_);
}

int64_t TcpStore::add(const std::string& key, int64_t value) {
  tcputil::sendValue<QueryType>(storeSocket_, QueryType::ADD);
  tcputil::sendString(storeSocket_, key, true);
  tcputil::sendValue<int64_t>(storeSocket_, value);
  return tcputil::recvValue<int64_t>(storeSocket_);
}

bool TcpStore::check(const std::vector<std::string>& keys) {

  tcputil::sendValue<QueryType>(storeSocket_, QueryType::CHECK);
  SizeType nkeys = keys.size();
  tcputil::sendBytes<SizeType>(storeSocket_, &nkeys, 1, (nkeys > 0));
  for (size_t i = 0; i < nkeys; i++) {
    tcputil::sendString(storeSocket_, keys[i], (i != (nkeys - 1)));
  }
  auto checkResponse = tcputil::recvValue<QueryType>(storeSocket_);
  if (checkResponse == QueryType::STOP_WAITING) {
    return true;
  } else if (checkResponse == QueryType::KEEP_WAITING) {
    return false;
  } else {
    throw std::runtime_error("stop_waiting or keep_waiting response expected");
  }
}

void TcpStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {

  const auto start = std::chrono::steady_clock::now();
  while (!check(keys)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      throw std::runtime_error("Wait timeout");
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace c10d
