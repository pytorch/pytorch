#include "Store.hpp"
#include "../ChannelUtils.hpp"

#include <poll.h>
#include <system_error>

namespace thd {

using store_type = std::unordered_map<std::string, std::vector<char>>;

static void store_thread_daemon();
static bool query(rank_type rank, const std::vector<int>& sockets);
static bool checkAndUpdate(const store_type& store,
                           std::vector<std::string>& keys);
static void wake_up(int socket);

enum class query_type : uint8_t {
  set,
  get,
  wait,
  finish,
  stop_waiting
};

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
    send_value<query_type>(_socket, query_type::finish);
    _store_thread.join();
  }
}

void Store::set(const std::string& key, const std::vector<char>& data) {
  send_value<query_type>(_socket, query_type::set);
  send_string(_socket, key, true);
  send_vector<char>(_socket, data);
}

std::vector<char> Store::get(const std::string& key) {
  wait({key});
  send_value<query_type>(_socket, query_type::get);
  send_string(_socket, key);
  return recv_vector<char>(_socket);
}

void Store::wait(const std::vector<std::string>& keys) {
  send_value<query_type>(_socket, query_type::wait);
  size_type nkeys = keys.size();
  send_bytes<size_type>(_socket, &nkeys, 1, (nkeys > 0));
  for (int i = 0; i < nkeys; i++) {
    send_string(_socket, keys[i], (i != (nkeys - 1)));
  }
  // after sending the query, wait for a 'stop_waiting' reponse
  query_type qr;
  recv_bytes<query_type>(_socket, &qr, 1);
  if (qr != query_type::stop_waiting)
    throw std::runtime_error("stop_waiting response expected");
}

static void store_thread_daemon() {
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
static bool query(rank_type rank, const std::vector<int>& sockets) {
  static store_type store_;
  static std::vector<std::vector<std::string>> awaited_keys(sockets.size());
  int socket = sockets[rank];
  query_type qt;
  recv_bytes<query_type>(socket, &qt, 1);
  if (qt == query_type::set) {
    std::string key = recv_string(socket);
    store_[key] = recv_vector<char>(socket);
    // On "set" wake up all of the processes that wait
    // for keys already in the store
    for (std::size_t i = 0; i < sockets.size(); i++) {
      auto& p_socket = sockets[i];
      auto& p_keys = awaited_keys[i];
      if (checkAndUpdate(store_, p_keys)) {
        wake_up(p_socket);
        awaited_keys[i].clear();
      }
    }
    return false;
  } else if (qt == query_type::get) {
    std::string key = recv_string(socket);
    std::vector<char> data = store_.at(key);
    send_vector(socket, data);
    return false;
  } else if (qt == query_type::wait) {
    size_type nargs;
    recv_bytes<size_type>(socket, &nargs, 1);
    std::vector<std::string> keys(nargs);
    for (int i = 0; i < nargs; i++) {
      keys[i] = recv_string(socket);
    }
    if (checkAndUpdate(store_, keys)) {
      wake_up(socket);
    } else {
      awaited_keys[socket] = std::move(keys);
    }
    return false;
  } else if (qt == query_type::finish) {
    return true;
  } else {
    throw std::runtime_error("expected a query type");
  }
}

static bool checkAndUpdate(const store_type& store,
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

static void wake_up(int socket) {
  send_value<query_type>(socket, query_type::stop_waiting);
}

} // namespace thd
