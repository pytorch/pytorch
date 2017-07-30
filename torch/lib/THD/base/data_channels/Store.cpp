#include "Store.hpp"
#include "../ChannelUtils.hpp"

#include <poll.h>
#include <system_error>
#include <unistd.h>

namespace thd {

namespace {

enum class QueryType : std::uint8_t {
  SET,
  GET,
  WAIT,
  STOP_WAITING
};

} // anonymous namespace

Store::StoreDeamon::StoreDeamon(int listen_socket)
 : _listen_socket(listen_socket)
 , _keys_awaited()
 , _sockets()
{
  _deamon = std::thread(&Store::StoreDeamon::deamon, this);
}

Store::StoreDeamon::~StoreDeamon()
{
  ::close(_listen_socket);
  for (auto socket : _sockets) {
    if (socket != -1)
      ::close(socket);
  }
}

void Store::StoreDeamon::join() {
  _deamon.join();
}

void Store::StoreDeamon::deamon() {
  std::vector<struct pollfd> fds;
  fds.push_back({ .fd = _listen_socket, .events = POLLIN });

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (std::size_t i = 0; i < _sockets.size(); i++) {
      fds[i].revents = 0;
    }

    SYSCHECK(::poll(fds.data(), fds.size(), -1));
    if (fds[0].revents != 0) {
      if (fds[0].revents ^ POLLIN)
        throw std::system_error(ECONNABORTED, std::system_category());

      int sock_fd = std::get<0>(accept(_listen_socket));
      _sockets.push_back(sock_fd);
      _keys_awaited.push_back(0);
      fds.push_back({ .fd = sock_fd, .events = POLLIN });
    }
    for (std::size_t rank = 0; rank < _sockets.size(); rank++) {
      if (fds[rank + 1].revents == 0)
        continue;

      if (fds[rank + 1].revents ^ POLLIN)
        throw std::system_error(ECONNABORTED, std::system_category());

      try {
        query(rank);
      } catch (...) {
        // There was an error when processing query. Probably an exception occurred in
        // recv/send what would indicate that socket on the other side has been closed.
        // If the closing was due to normal exit, then the store should exit too.
        // Otherwise, if it was different exception, other processes will get
        // an exception once they try to use the store.
        finished = true;
        break;
      }
    }
  }
}

/*
 * query communicates with the worker. The format
 * of the query is as follows:
 * type of query | size of arg1 | arg1 | size of arg2 | arg2 | ...
 * or, in the case of wait
 * type of query | number of args | size of arg1 | arg1 | ...
 */
void Store::StoreDeamon::query(rank_type rank) {
  int socket = _sockets[rank];
  QueryType qt;
  recv_bytes<QueryType>(socket, &qt, 1);
  if (qt == QueryType::SET) {
    std::string key = recv_string(socket);
    _store[key] = recv_vector<char>(socket);
    // On "set", wake up all of the processes that wait
    // for keys already in the store
    auto to_wake = _waiting.find(key);
    if (to_wake != _waiting.end()) {
      for (int proc : to_wake->second) {
        if (--_keys_awaited[proc] == 0)
          send_value<QueryType>(_sockets[proc], QueryType::STOP_WAITING);
      }
      _waiting.erase(to_wake);
    }
  } else if (qt == QueryType::GET) {
    std::string key = recv_string(socket);
    std::vector<char> data = _store.at(key);
    send_vector(socket, data);
  } else if (qt == QueryType::WAIT) {
    size_type nargs;
    recv_bytes<size_type>(socket, &nargs, 1);
    std::vector<std::string> keys(nargs);
    for (std::size_t i = 0; i < nargs; i++) {
      keys[i] = recv_string(socket);
    }
    if (checkAndUpdate(keys)) {
      send_value<QueryType>(socket, QueryType::STOP_WAITING);
    } else {
      for (auto& key : keys) {
        _waiting[key].push_back(rank);
      }
      _keys_awaited[rank] = keys.size();
    }
  } else {
    throw std::runtime_error("expected a query type");
  }
}

bool Store::StoreDeamon::checkAndUpdate(std::vector<std::string>& keys) const {
  bool ret = true;
  for (auto it = keys.begin(); it != keys.end();) {
    if (_store.count(*it) == 0) {
      ret = false;
      it++;
    } else {
      it = keys.erase(it);
    }
  }
  return ret;
}



Store::Store(const std::string& addr,
             port_type port, int listen_socket)
 : _store_addr(addr)
 , _store_port(port)
 , _socket(-1)
 , _store_thread(nullptr)
{
  if (listen_socket != Store::CLIENT_ONLY) {
    _store_thread = std::unique_ptr<StoreDeamon>(
      new StoreDeamon(listen_socket)
    );
  }

  _socket = connect(_store_addr, _store_port);
}

Store::~Store() {
  ::close(_socket);

  // Store deamon should end because of closed connection.
  if (_store_thread) {
    _store_thread->join();
  }
}

void Store::set(const std::string& key, const std::vector<char>& data) {
  send_value<QueryType>(_socket, QueryType::SET);
  send_string(_socket, key, true);
  send_vector<char>(_socket, data);
}

std::vector<char> Store::get(const std::string& key) {
  wait({key});
  send_value<QueryType>(_socket, QueryType::GET);
  send_string(_socket, key);
  return recv_vector<char>(_socket);
}

void Store::wait(const std::vector<std::string>& keys) {
  send_value<QueryType>(_socket, QueryType::WAIT);
  size_type nkeys = keys.size();
  send_bytes<size_type>(_socket, &nkeys, 1, (nkeys > 0));
  for (std::size_t i = 0; i < nkeys; i++) {
    send_string(_socket, keys[i], (i != (nkeys - 1)));
  }
  // after sending the query, wait for a 'stop_waiting' response
  QueryType qr;
  recv_bytes<QueryType>(_socket, &qr, 1);
  if (qr != QueryType::STOP_WAITING)
    throw std::runtime_error("stop_waiting response expected");
}

} // namespace thd
