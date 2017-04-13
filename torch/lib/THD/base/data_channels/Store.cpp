#include "Store.hpp"
#include "../ChannelUtils.hpp"

#include <poll.h>
#include <system_error>
#include <unistd.h>

namespace thd {

namespace {

using store_type = std::unordered_map<std::string, std::vector<char>>;

void store_thread_daemon();
bool query(rank_type rank, const std::vector<int>& sockets);
bool checkAndUpdate(const store_type& store,
                    std::vector<std::string>& keys);
void wake_up(int socket);

enum class QueryType : uint8_t {
  SET,
  GET,
  WAIT,
  FINISH,
  STOP_WAITING
};

} // anonymous namespace

Store::Store()
    : _rank(load_rank_env()) {
  // Master runs the store_thread
  if (_rank == 0) {
    _store_thread = std::thread(store_thread_daemon);
    std::tie(_store_port, std::ignore) = load_master_env();
    _store_addr = "localhost";
  } else {
    std::tie(_store_addr, _store_port) = load_worker_env();
  }
  _socket = connect(_store_addr, _store_port);
}

Store::~Store() {
  // the 0 process has to stop the daemon
  if (_rank == 0) {
    send_value<QueryType>(_socket, QueryType::FINISH);
    _store_thread.join();
  }
  SYSCHECK(::close(_socket));
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
  for (int i = 0; i < nkeys; i++) {
    send_string(_socket, keys[i], (i != (nkeys - 1)));
  }
  // after sending the query, wait for a 'stop_waiting' reponse
  QueryType qr;
  recv_bytes<QueryType>(_socket, &qr, 1);
  if (qr != QueryType::STOP_WAITING)
    throw std::runtime_error("stop_waiting response expected");
}

namespace {

void store_thread_daemon() {
  int ret;
  port_type port;
  rank_type world_size;
  int socket;
  std::tie(port, world_size) = load_master_env();
  std::vector<int> sockets(world_size);
  std::tie(socket, std::ignore) = listen(port);
  // accept WORLD_SIZE connections
  for (int i = 0; i < world_size; i++) {
    int p_socket;
    std::tie(p_socket, std::ignore) = accept(socket);
    sockets[i] = p_socket;
  }

  SYSCHECK(::close(socket));

  // listen for requests
  struct pollfd fds[world_size];
  for (int i = 0; i < world_size; i++) {
    fds[i].fd = sockets[i];
    fds[i].events = POLLIN;   // we only read
  }

  // receive the queries
  bool finished = false;
  while (!finished) {
    for (int i = 0; i < world_size; i++) fds[i].revents = 0;
    SYSCHECK(::poll(fds, world_size, -1));
    for (int i = 0; i < world_size; i++) {
      if (fds[i].revents == 0)
        continue;
      if (fds[i].revents ^ POLLIN)
        throw std::system_error(ECONNABORTED, std::system_category());
      finished = query(i, sockets);
      if (finished)
        break;
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
bool query(rank_type rank, const std::vector<int>& sockets) {
  static store_type store_;
  static std::unordered_map<std::string, std::vector<rank_type>> waiting;
  static std::vector<int> keys_awaited(sockets.size(), 0);
  int socket = sockets[rank];
  QueryType qt;
  recv_bytes<QueryType>(socket, &qt, 1);
  if (qt == QueryType::SET) {
    std::string key = recv_string(socket);
    store_[key] = recv_vector<char>(socket);
    // On "set", wake up all of the processes that wait
    // for keys already in the store
    auto to_wake = waiting.find(key);
    if (to_wake != waiting.end()) {
      for (int proc : to_wake->second) {
        if (--keys_awaited[proc] == 0)
          wake_up(sockets[proc]);
      }
      waiting.erase(to_wake);
    }
    return false;
  } else if (qt == QueryType::GET) {
    std::string key = recv_string(socket);
    std::vector<char> data = store_.at(key);
    send_vector(socket, data);
    return false;
  } else if (qt == QueryType::WAIT) {
    size_type nargs;
    recv_bytes<size_type>(socket, &nargs, 1);
    std::vector<std::string> keys(nargs);
    for (int i = 0; i < nargs; i++) {
      keys[i] = recv_string(socket);
    }
    if (checkAndUpdate(store_, keys)) {
      wake_up(socket);
    } else {
      for (auto& key : keys) {
        waiting[key].push_back(rank);
      }
      keys_awaited[rank] = keys.size();
    }
    return false;
  } else if (qt == QueryType::FINISH) {
    return true;
  } else {
    throw std::runtime_error("expected a query type");
  }
}

bool checkAndUpdate(const store_type& store,
                           std::vector<std::string>& keys) {
  bool ret = true;
  for (auto it = keys.begin(); it != keys.end(); it++) {
    if (store.find(*it) == store.end())
      ret = false;
    else
      it = keys.erase(it);
  }
  return ret;
}

void wake_up(int socket) {
  send_value<QueryType>(socket, QueryType::STOP_WAITING);
}

} // anonymous namespace 

} // namespace thd
