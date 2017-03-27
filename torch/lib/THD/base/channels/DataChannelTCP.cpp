#include "DataChannelTCP.hpp"
#include "DataChannelUtils.hpp"

#include <arpa/inet.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <netdb.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>


#define SYSCHECK(expr) { \
  errno = 0; (expr);     \
  if (errno != 0) throw std::system_error(errno, std::system_category()); \
}

namespace thd {
namespace {

constexpr int MASTER_RANK = 0;
constexpr int LISTEN_QUEUE_SIZE = 64;

template<typename T>
void send_bytes(int socket, const T* buffer, std::size_t length)
{
  std::size_t bytes_to_send = sizeof(T) * length;
  if (bytes_to_send == 0)
    return;

  auto bytes = reinterpret_cast<const std::uint8_t*>(buffer);
  std::uint8_t *current_bytes = const_cast<std::uint8_t*>(bytes);

  while (bytes_to_send > 0) {
    ssize_t bytes_sent;
    SYSCHECK(bytes_sent = ::send(socket, current_bytes, bytes_to_send, 0))
    if (bytes_sent == 0)
      throw std::system_error(EBADMSG, std::system_category());

    bytes_to_send -= bytes_sent;
    current_bytes += bytes_sent;
  }
}


template<typename T>
void recv_bytes(int socket, T* buffer, std::size_t length)
{
  std::size_t bytes_to_receive = sizeof(T) * length;
  if (bytes_to_receive == 0)
    return;

  auto bytes = reinterpret_cast<std::uint8_t*>(buffer);
  std::uint8_t *current_bytes = bytes;

  while (bytes_to_receive > 0) {
    ssize_t bytes_received;
    SYSCHECK(bytes_received = ::recv(socket, current_bytes, bytes_to_receive, 0))
    if (bytes_received == 0)
      throw std::system_error(EBADMSG, std::system_category());

    bytes_to_receive -= bytes_received;
    current_bytes += bytes_received;
  }
}


inline bool validatePort(int port) {
  return (port > 0 && port < 65536);
}


inline int log2ceil(std::uint32_t value) {
  int dim = 0;
#if defined(__GNUC__)
  if (value <= 1)
    return 0;
  dim = 32 - __builtin_clz(value - 1);
#else
  for (int size = 1; size < value; ++dim, size <<= 1) /* empty */;
#endif // defined(__GNUC__)
  return dim;
}

} // namespace


DataChannelTCP::RequestTCP::RequestTCP(QueueWorker::Request&& request)
  : _request(std::move(request)) {
}


DataChannelTCP::RequestTCP::~RequestTCP() {}


bool DataChannelTCP::RequestTCP::isCompleted() {
  return _request.isCompleted();
}


void DataChannelTCP::RequestTCP::wait() {
  _request.wait();
}


DataChannelTCP::DataChannelTCP()
  : DataChannelTCP(-1)
{}


DataChannelTCP::DataChannelTCP(int timeout)
  : _socket(-1)
  , _port(0)
  , _timeout(timeout)
  , _poll_events(nullptr)
{
  auto rank_env = std::getenv(RANK_ENV);
  if (!rank_env)
    throw std::domain_error("env variable not found: " + std::string(RANK_ENV));

  _rank = std::stoi(rank_env);
  if (_rank == MASTER_RANK) { // MASTER
    auto master_port_env = std::getenv(MASTER_PORT_ENV);
    if (!master_port_env)
      throw std::domain_error("env variable not found: " + std::string(MASTER_PORT_ENV));

    _port = std::stoul(master_port_env);
    if (!validatePort(_port))
      throw std::domain_error("invalid listen port number");

    auto num_proceses_env = std::getenv(WORLD_SIZE_ENV);
    if (!num_proceses_env)
      throw std::domain_error("env variable not found: " + std::string(WORLD_SIZE_ENV));

    int processes_number = std::stoul(num_proceses_env);
    if (processes_number == 0)
      throw std::domain_error("invalid " + std::string(WORLD_SIZE_ENV) + " env variable");

    _processes.resize(processes_number);
    _processes[_rank] = {
      .rank = static_cast<std::uint32_t>(_rank),
      .address = "",
      .port = 0,
      .socket = -1,
    };
  } else { // WORKER
    auto master_addr_env = std::getenv(MASTER_ADDR_ENV);
    if (!master_addr_env)
      throw std::domain_error("env variable not found: " + std::string(MASTER_ADDR_ENV));

    std::string full_address = std::string(master_addr_env);
    auto found_pos = full_address.rfind(":");
    if (found_pos == std::string::npos)
      throw std::domain_error("invalid master address, usage: IP:PORT | HOSTNAME:PORT");

    std::string str_port = full_address.substr(found_pos + 1);
    int port = std::stoul(str_port);
    if (!validatePort(port))
      throw std::domain_error("invalid master port number");

    // add master
    _processes.resize(MASTER_RANK + 1);
    _processes[MASTER_RANK] = {
      .rank = MASTER_RANK,
      .address = full_address.substr(0, found_pos),
      .port = static_cast<std::uint16_t>(port),
      .socket = -1,
    };
  }
}


DataChannelTCP::~DataChannelTCP()
{
  if (_socket != -1)
    ::close(_socket);

  for (const auto& process : _processes) {
    if ((process.rank != _rank) && (process.socket != -1))
      ::close(process.socket);
  }
}


void DataChannelTCP::listen(std::uint16_t port = 0) {
  struct addrinfo hints, *res = NULL;

  memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_PASSIVE;
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP

  // `getaddrinfo` will sort addresses according to RFC 3484 and can be tweeked
  // by editing `/etc/gai.conf`. so there is no need to manual sorting
  // or protcol preference.
  int err = getaddrinfo(NULL, std::to_string(port).data(), &hints, &res);
  if (err != 0 || !res) {
    throw std::invalid_argument("cannot find host to listen on: " + std::string(gai_strerror(err)));
  }

  std::shared_ptr<struct addrinfo> addresses(res, [](struct addrinfo* p) {
    ::freeaddrinfo(p);
  });

  struct addrinfo *next_addr = addresses.get();
  while (true) {
    try {
      SYSCHECK(_socket = ::socket(next_addr->ai_family, next_addr->ai_socktype, next_addr->ai_protocol))

      int optval = 1;
      SYSCHECK(::setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)))
      SYSCHECK(::bind(_socket, next_addr->ai_addr, next_addr->ai_addrlen))
      SYSCHECK(::listen(_socket, LISTEN_QUEUE_SIZE))
      break;
    } catch (const std::system_error& e) {
      ::close(_socket);
      next_addr = next_addr->ai_next;

      // we have tried all addresses but could not establish listening on any of them
      if (!next_addr) {
        throw e;
      }
    }
  }

  // get listen port
  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  SYSCHECK(::getsockname(_socket, reinterpret_cast<struct sockaddr*>(&addr), &addr_len))
  _port = ntohs(addr.sin_port);
}


int DataChannelTCP::connect(const std::string& address, std::uint16_t port,
                            int wait = true) const {
  struct addrinfo hints, *res = NULL;

  memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_NUMERICSERV; // specifies that port (service) is numeric
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP

  // `getaddrinfo` will sort addresses according to RFC 3484 and can be tweeked
  // by editing `/etc/gai.conf`. so there is no need to manual sorting
  // or protcol preference.
  int err = ::getaddrinfo(address.data(), std::to_string(port).data(), &hints, &res);
  if (err != 0 || !res) {
    throw std::invalid_argument("host not found: " + std::string(gai_strerror(err)));
  }

  std::shared_ptr<struct addrinfo> addresses(res, [](struct addrinfo* p) {
    ::freeaddrinfo(p);
  });

  struct addrinfo *next_addr = addresses.get();
  int socket;
  while (true) {
    try {
      SYSCHECK(socket = ::socket(next_addr->ai_family, next_addr->ai_socktype, next_addr->ai_protocol))
      SYSCHECK(::connect(socket, next_addr->ai_addr, next_addr->ai_addrlen))
      break;
    } catch (const std::system_error& e) {
      // if `connect` fails, the state of the socket is unspecified.
      // we should close the socket and create a new one before attempting to reconnect.
      ::close(socket);

      if (!wait || (errno != ECONNREFUSED)) {
        // we need to move to next address because this was not available
        // to connect or to create socket
        next_addr = next_addr->ai_next;

        // we have tried all addresses but could not connect to any of them
        if (!next_addr) {
          throw e;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }

  return socket;
}


std::tuple<int, std::string> DataChannelTCP::accept() const {
  // poll on listen socket, it allows to make timeout
  std::unique_ptr<struct pollfd[]> events(new struct pollfd[1]);
  events[0] = {.fd = _socket, .events = POLLIN};

  int res;
  SYSCHECK(res = ::poll(events.get(), 1, _timeout))
  if (res == 0) {
    throw std::runtime_error("waiting for processes to connect has timed out");
  } else {
    if (!(events[0].revents & POLLIN))
      throw std::system_error(ECONNABORTED, std::system_category());
  }

  int socket;
  SYSCHECK(socket = ::accept(_socket, NULL, NULL))

  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  char address[INET6_ADDRSTRLEN + 1];

  SYSCHECK(::getpeername(socket, reinterpret_cast<struct sockaddr*>(&addr), &addr_len))

  if (addr.ss_family == AF_INET) {
    struct sockaddr_in *s = reinterpret_cast<struct sockaddr_in*>(&addr);
    SYSCHECK(::inet_ntop(AF_INET, &(s->sin_addr), address, INET_ADDRSTRLEN))
    address[INET_ADDRSTRLEN] = '\0';
  } else {
    struct sockaddr_in6 *s = reinterpret_cast<struct sockaddr_in6*>(&addr);
    SYSCHECK(::inet_ntop(AF_INET6, &(s->sin6_addr), address, INET6_ADDRSTRLEN))
    address[INET6_ADDRSTRLEN] = '\0';
  }

  return std::make_tuple(socket, std::string(address));
}


bool DataChannelTCP::initWorker() {
  auto& master = _processes[MASTER_RANK];
  master.socket = connect(master.address, master.port);
  int master_socket = master.socket;

  listen();

  std::uint32_t p_rank = (std::uint32_t)_rank;
  std::uint16_t p_port = (std::uint16_t)_port;
  send_bytes<std::uint32_t>(master_socket, &p_rank, 1);
  send_bytes<std::uint16_t>(master_socket, &p_port, 1); // send listening port to master

  std::uint32_t processes_number;
  recv_bytes<std::uint32_t>(master_socket, &processes_number, 1);
  _processes.resize(processes_number);

  // get all metadata of other processes in network
  processes_number--; // exclude master
  while (processes_number > 0) {
    std::uint32_t p_rank, p_address_len;
    std::uint16_t p_port;

    recv_bytes<std::uint32_t>(master_socket, &p_rank, 1); // get process rank
    recv_bytes<std::uint32_t>(master_socket, &p_address_len, 1); // get process address length

    // get process address
    std::unique_ptr<char[]> tmp_address(new char[p_address_len + 1]);
    recv_bytes<char>(master_socket, tmp_address.get(), p_address_len);

    recv_bytes<std::uint16_t>(master_socket, &p_port, 1); // get process port

    _processes[p_rank] = {
      .rank = p_rank,
      .address = std::string(tmp_address.get(), p_address_len),
      .port = p_port,
      .socket = -1,
    };

    processes_number--;
  }

  // make network connection with other processes
  for (auto& process : _processes) {
    if ((process.rank == _rank) || (process.rank == MASTER_RANK)) continue;

    // it is to prevent accept-connect deadlock
    if (process.rank < _rank) {
      process.socket = connect(process.address, process.port);
    } else {
      auto accept_state = accept();
      process.socket = std::get<0>(accept_state);
    }
  }

  // close socket for listening, we will not use it anymore
  ::close(_socket);
  _socket = -1;

  return true;
}


bool DataChannelTCP::initMaster() {
  listen(_port);

  // wait for all workers to connect
  int workers = _processes.size() - 1;
  while (workers > 0) {
    auto accept_state = accept();
    int socket = std::get<0>(accept_state);
    std::string p_address = std::get<1>(accept_state);

    std::uint32_t p_rank;
    std::uint16_t p_port;
    recv_bytes<std::uint32_t>(socket, &p_rank, 1);
    recv_bytes<std::uint16_t>(socket, &p_port, 1);

    if (p_rank >= _processes.size()) {
      throw std::out_of_range(
        "worker's rank(" + std::to_string(p_rank) + ") is out"
        "of range: [0, " + std::to_string(_processes.size() - 1) + "]"
      );
    }

    if (_processes[p_rank].rank == p_rank) {
      throw std::logic_error(
        "two processes (" + _processes[p_rank].address + ", " + p_address + ") "
        "reported a rank of " + std::to_string(p_rank)
      );
    }

    _processes[p_rank] = {
      .rank = p_rank,
      .address = p_address,
      .port = p_port,
      .socket = socket,
    };

    workers--;
  }

  // send informations about processes to all workers
  for (const auto& worker : _processes) {
    if (worker.rank == _rank) continue;

    std::uint32_t processes_number = _processes.size();
    send_bytes<std::uint32_t>(worker.socket, &processes_number, 1);

    for (auto& process : _processes) {
      if (process.rank == _rank) continue;

      std::uint32_t proc_address_length = process.address.size();
      send_bytes<std::uint32_t>(worker.socket, &process.rank, 1);
      send_bytes<std::uint32_t>(worker.socket, &proc_address_length, 1);
      send_bytes<char>(worker.socket, process.address.data(), proc_address_length);
      send_bytes<std::uint16_t>(worker.socket, &(process.port), 1);
    }
  }

  // close socket for listening, we will not use it anymore
  ::close(_socket);
  _socket = -1;

  return true;
}


bool DataChannelTCP::init() {
  bool ok = (_rank == MASTER_RANK ? initMaster() : initWorker());
  if (ok) {
    std::vector<int> ranks;
    ranks.reserve(_processes.size());
    for (size_t rank = 0; rank < _processes.size(); ++rank)
      ranks.push_back(rank);

    _groups.insert({
      THDGroupWORLD,
      DataChannel::Group(ranks, _processes.size() - 1)
    });
  }

  return ok;
}


int DataChannelTCP::getRank() {
  return _rank;
}


int DataChannelTCP::getNumProcesses() {
  return _processes.size();
}


void DataChannelTCP::allGather(std::vector<thpp::Tensor*>& output,
                               thpp::Tensor& input, THDGroup group_id) {
  /*
   * Since all-gather is semantically equivalent to gather followed by
   * broadcast we use those functions to implement all-gather function.
   *
   * Even though we use first rank from group as point of gather and broadcast
   * we should not see any bottlenecks here.
   */

  const auto& group = _groups.at(group_id);
  bool exists;
  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  if (output.size() != group.size())
    throw std::logic_error("allGather: number of output tensors and group size does not match");

  for (auto out_tensor : output)
    assertTensorEqual(*out_tensor, input, "allGather");

  auto main_rank = group.mustGetGlobalRank(0);
  gather(output, input, main_rank, group_id);

  for (std::size_t i = 0; i < group.size(); ++i)
    broadcast(*(output.at(i)), main_rank, group_id);
}


void DataChannelTCP::gather(std::vector<thpp::Tensor*>& output,
                            thpp::Tensor& input, int dst_rank, THDGroup group_id) {
  const auto& group = _groups.at(group_id);
  bool exists;

  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  // assert if dst_rank exists in group
  group.mustGetGroupRank(dst_rank);
  if (_rank != dst_rank) {
    send(input, dst_rank);
  } else {
    if (output.size() != group.size())
      throw std::logic_error("gather: number of output tensors and group size does not match");

    for (auto out_tensor : output)
      assertTensorEqual(*out_tensor, input, "gather");

    for (std::size_t i = 0; i < group.size(); ++i) {
      // TODO: change it to some kind of helper
      auto global_rank = group.mustGetGlobalRank(i);
      if (_rank != global_rank) {
        receive(*(output.at(i)), global_rank);
      } else {
        memcpy(output.at(i)->data(), input.data(), input.numel() * input.elementSize());
      }
    }
  }
}


void DataChannelTCP::scatter(std::vector<thpp::Tensor*>& input,
                             thpp::Tensor& output, int src_rank,
                             THDGroup group_id) {
  const auto& group = _groups.at(group_id);
  bool exists;

  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  // assert if src_rank exists in group
  group.mustGetGroupRank(src_rank);
  if (_rank != src_rank) {
    receive(output, src_rank);
  } else {
    if (input.size() != group.size())
      throw std::logic_error("scatter: number of input tensors and group size does not match");

    for (auto in_tensor : input)
      assertTensorEqual(*in_tensor, output, "scatter");

    for (std::size_t i = 0; i < group.size(); ++i) {
      // TODO: change it to some kind of helper
      auto global_rank = group.mustGetGlobalRank(i);
      if (_rank != global_rank) {
        send(*(input.at(i)), global_rank);
      } else {
        memcpy(output.data(), input.at(i)->data(), output.numel() * output.elementSize());
      }
    }
  }
}


void DataChannelTCP::allReduce(thpp::Tensor& data, THDReduceOp operation,
                               THDGroup group_id) {
  /*
   * Since an all-reduce operation is semantically equivalent to an
   * all-to-one reduction followed by a one-to-all broadcast, the asymptotically
   * optimal algorithms for these two operations can be used to construct
   * a similar algorithm for the all-reduce operation.
   *
   * Even though we use first rank from group as point of broadcast and aggregation
   * we should not see any bottlenecks here.
   */

  const auto& group = _groups.at(group_id);
  bool exists;

  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  auto main_rank = group.mustGetGlobalRank(0);
  reduce(data, operation, main_rank, group_id);
  broadcast(data, main_rank, group_id);
}


void DataChannelTCP::reduce(thpp::Tensor& data, THDReduceOp operation,
                            int dst_rank, THDGroup group_id) {
  /*
   * Idea of this algorithm is similar to broadcast but with reversed
   * order and direction of communication.
   */

  const auto& group = _groups.at(group_id);
  unsigned int group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  unsigned int group_dst_rank = group.mustGetGroupRank(dst_rank);
  int dim = log2ceil(group.size());
  int virtual_rank = ((group.size() - group_dst_rank) + group_rank) % group.size();
  long long mask = 0;
  auto result_tensor = data.clone();

  for (int k = 0; k <= dim - 1; mask ^= (1 << k), ++k) {
    if ((virtual_rank & mask) == 0) {
      int partner = virtual_rank ^ (1 << k); // partner has opposite bit `k`
      if (partner >= group.size())
        continue;

      partner = group.mustGetGlobalRank((partner + group_dst_rank) % group.size());
      if ((virtual_rank & (1 << k)) != 0) {
        send(*result_tensor, partner);
      } else {
        receive(data, partner);
        _reduce(*result_tensor, data, operation);
      }
    }
  }

  if (_rank == dst_rank)
    std::memcpy(data.data(), result_tensor->data(), data.elementSize() * data.numel());

  delete result_tensor;
}


void DataChannelTCP::broadcast(thpp::Tensor& data, int src_rank,
                               THDGroup group_id) {
  /*
   * General idea of this algorithm is to send data in `d` dimensional
   * hypercube where vertices are nodes (processes) and edges are
   * network connections which can be used to transfer data.
   *
   * Since hypercube algorithm works for case when broadcasting rank is 0
   * we have to create `virtual_rank` which converts regular ranks to
   * virtual ones where `virtual_rank` for `src_rank` is 0.
   */

  const auto& group = _groups.at(group_id);
  unsigned int group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  unsigned int group_src_rank = group.mustGetGroupRank(src_rank);
  int dim = log2ceil(group.size());
  int virtual_rank = ((group.size() - group_src_rank) + group_rank) % group.size();
  long long mask = (1 << dim) - 1;

  for (int k = dim - 1; k >= 0; --k) {
    mask ^= (1 << k); // clear bit `k`
    if ((virtual_rank & mask) == 0) {
      int partner = virtual_rank ^ (1 << k); // partner has opposite bit `k`
      if (partner >= group.size())
        continue;

      partner = group.mustGetGlobalRank((partner + group_src_rank) % group.size());
      if ((virtual_rank & (1 << k)) == 0) {
        send(data, partner);
      } else {
        receive(data, partner);
      }
    }
  }
}


void DataChannelTCP::send(const Scalar& data, int dst_rank) {
  auto request = _send_worker.push([this, &data, dst_rank]{
    this->_send(data, dst_rank);
  });
  request.wait();
}


void DataChannelTCP::send(thpp::Tensor& data, int dst_rank) {
  auto request = _send_worker.push([this, &data, dst_rank]{
    this->_send(data, dst_rank);
  });
  request.wait();
}


void DataChannelTCP::receive(Scalar& data, int src_rank) {
  auto request = _receive_worker.push([this, &data, src_rank]{
    this->_receive(data, src_rank);
  });
  request.wait();
}


void DataChannelTCP::receive(thpp::Tensor& data) {
  auto request = _receive_worker.push([this, &data]{
    if (!this->_poll_events) {
      // cache poll events array, it will be reused in another `receive` calls
      this->_poll_events.reset(new struct pollfd[this->_processes.size()]);
      for (size_t rank = 0; rank < this->_processes.size(); ++rank) {
        this->_poll_events[rank] = {
          .fd = this->_processes[rank].socket,
          .events = POLLIN
        };
      }
    }

    // cleanup
    for (size_t rank = 0; rank < this->_processes.size(); ++rank) {
      this->_poll_events[rank].revents = 0;
    }

    SYSCHECK(::poll(this->_poll_events.get(), this->_processes.size(), -1)) // infinite timeout
    for (size_t rank = 0; rank < this->_processes.size(); ++rank) {
      if (this->_poll_events[rank].revents == 0)
        continue;

      if (!(this->_poll_events[rank].revents & POLLIN))
        throw std::system_error(ECONNABORTED, std::system_category());

      this->_receive(data, rank);
      break;
    }
  });

  request.wait();
}


void DataChannelTCP::receive(thpp::Tensor& data, int src_rank) {
  auto request = _receive_worker.push([this, &data, src_rank]{
    this->_receive(data, src_rank);
  });
  request.wait();
}


DataChannelTCP::RequestTCP* DataChannelTCP::isend(thpp::Tensor& data,
                                                  int dst_rank) {
  std::shared_ptr<thpp::Tensor> copy_tensor(data.clone_shallow());
  auto request = _send_worker.push([this, copy_tensor, dst_rank]{
    this->_send(*copy_tensor, dst_rank);
  });
  return new DataChannelTCP::RequestTCP(std::move(request));
}


DataChannelTCP::RequestTCP* DataChannelTCP::ireceive(thpp::Tensor& data,
                                                     int src_rank) {
  std::shared_ptr<thpp::Tensor> copy_tensor(data.clone_shallow());
  auto request = _receive_worker.push([this, copy_tensor, src_rank]{
    this->_receive(*copy_tensor, src_rank);
  });
  return new DataChannelTCP::RequestTCP(std::move(request));
}


void DataChannelTCP::barrier(THDGroup group_id) {
  /*
   * Barrier is implementation of Bruck algorithm. All processes send to
   * other processes with rank (i + 2^k) and recv from process with rank (i - 2^k)
   * with wrap-around. Since we cannot do recv and send at the same time
   * we do recv asynchronously (thread), send byte and then wait for recv to complete.
   */

  const auto& group = _groups.at(group_id);
  unsigned int group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  std::uint8_t byte = 1;
  for (int distance = 1; distance < group.size(); distance <<= 1) {
    int recv_partner = (group_rank + group.size() - distance) % group.size();
    const auto& recv_process = _processes.at(group.mustGetGlobalRank(recv_partner));
    auto recv_request = _receive_worker.push([&recv_process, &byte]{
      recv_bytes<std::uint8_t>(recv_process.socket, &byte, 1);
    });

    int send_partner = (group_rank + distance) % group.size();
    const auto& send_process = _processes.at(group.mustGetGlobalRank(send_partner));
    auto send_request = _send_worker.push([&send_process, &byte]{
      send_bytes<std::uint8_t>(send_process.socket, &byte, 1);
    });

    send_request.wait();
    recv_request.wait();
  }
}


THDGroup DataChannelTCP::newGroup(const std::vector<int>& ranks) {
  auto new_group = DataChannel::Group(ranks, _processes.size() - 1);
  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());

  _groups.insert({new_group_id, new_group});
  return new_group_id;
}


void DataChannelTCP::_send(const Scalar& data, int dst_rank) {
  /*
   * We have to check if dst_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  if (dst_rank < 0)
    throw std::out_of_range("destination rank is invalid (< 0)");

  const auto& process_dst = _processes.at(dst_rank);
  if (process_dst.rank == _rank)
    throw std::logic_error("cannot send scalar to process with same rank");

  // send size of scalar in bytes
  std::uint64_t scalar_bytes = data.elementSize();
  send_bytes<std::uint64_t>(process_dst.socket, &scalar_bytes, 1);

  // send data (bytes)
  send_bytes<std::uint8_t>(
    process_dst.socket,
    reinterpret_cast<const std::uint8_t*>(data.data()),
    scalar_bytes
  );
}


void DataChannelTCP::_send(thpp::Tensor& data, int dst_rank) {
  /*
   * We have to check if dst_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  if (dst_rank < 0)
    throw std::out_of_range("destination rank is invalid (< 0)");

  const auto& process_dst = _processes.at(dst_rank);
  if (process_dst.rank == _rank)
    throw std::logic_error("cannot send tensor to process with same rank");

  if (!data.isContiguous())
    throw std::logic_error("tensor to send is not contiguous");

  // send size of tensor data in bytes
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  send_bytes<std::uint64_t>(process_dst.socket, &tensor_bytes, 1);

  // send data (bytes)
  send_bytes<std::uint8_t>(
    process_dst.socket,
    reinterpret_cast<const std::uint8_t*>(data.data()),
    tensor_bytes
  );
}


void DataChannelTCP::_receive(Scalar& data, int src_rank) {
  /*
   * We have to check if src_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  if (src_rank < 0)
    throw std::out_of_range("source rank is invalid (< 0)");

  const auto& process_src = _processes.at(src_rank);
  if (process_src.rank == _rank)
    throw std::logic_error("cannot receive scalar from process with same rank");

  // get size of scalar in bytes
  std::uint64_t scalar_bytes;
  recv_bytes<std::uint64_t>(process_src.socket, &scalar_bytes, 1);

  // recv data (bytes)
  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[scalar_bytes]);
  recv_bytes<std::uint8_t>(process_src.socket, bytes.get(), scalar_bytes);

  std::uint64_t actual_scalar_bytes = data.elementSize();
  if (actual_scalar_bytes != scalar_bytes)
    throw std::logic_error("scalar sizes do not match");

  std::memcpy(data.data(), bytes.get(), scalar_bytes);
}


void DataChannelTCP::_receive(thpp::Tensor& data, int src_rank) {
  /*
   * We have to check if src_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  if (src_rank < 0)
    throw std::out_of_range("source rank is invalid (< 0)");

  const auto& process_src = _processes.at(src_rank);
  if (process_src.rank == _rank)
    throw std::logic_error("cannot receive tensor from process with same rank");

  if (!data.isContiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  // get size of tensor data in bytes
  std::uint64_t tensor_bytes;
  recv_bytes<std::uint64_t>(process_src.socket, &tensor_bytes, 1);

  // recv data (bytes)
  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
  recv_bytes<std::uint8_t>(process_src.socket, bytes.get(), tensor_bytes);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes)
    throw std::logic_error("tensor sizes do not match");

  std::memcpy(data.data(), bytes.get(), tensor_bytes);
}


void DataChannelTCP::_reduce(thpp::Tensor& result, thpp::Tensor& data,
                             THDReduceOp operation) const {
  assertTensorEqual(result, data, "reduce");

  thpp::Type tensor_type = data.type();
  switch(tensor_type) {
    case thpp::Type::CHAR:   _reduce<char>(result, data, operation); break;
    case thpp::Type::FLOAT:  _reduce<float>(result, data, operation); break;
    case thpp::Type::DOUBLE: _reduce<double>(result, data, operation); break;
    case thpp::Type::SHORT:  _reduce<short>(result, data, operation); break;
    case thpp::Type::USHORT: _reduce<unsigned short>(result, data, operation); break;
    case thpp::Type::INT:    _reduce<int>(result, data, operation); break;
    case thpp::Type::UINT:   _reduce<unsigned int>(result, data, operation); break;
    case thpp::Type::LONG:   _reduce<long>(result, data, operation); break;
    case thpp::Type::ULONG:  _reduce<unsigned long>(result, data, operation); break;
    case thpp::Type::LONG_LONG:  _reduce<long long>(result, data, operation); break;
    case thpp::Type::ULONG_LONG: _reduce<unsigned long long>(result, data, operation); break;
    default:
      throw std::logic_error("unsupported tensor type in reduce");
  }
}


template<typename T>
void DataChannelTCP::_reduce(thpp::Tensor& result, thpp::Tensor& data,
                             THDReduceOp operation) const {
  assertTensorEqual(result, data, "reduce");

  auto result_data = reinterpret_cast<T*>(result.data());
  auto new_data = reinterpret_cast<T*>(data.data());

  for (std::size_t i = 0; i < data.numel(); ++i) {
    if (operation == THDReduceOp::THDReduceMIN) {
      result_data[i] = std::min(result_data[i], new_data[i]);
    } else if (operation == THDReduceOp::THDReduceMAX) {
      result_data[i] = std::max(result_data[i], new_data[i]);
    } else if (operation == THDReduceOp::THDReduceSUM) {
      result_data[i] += new_data[i];
    } else if (operation == THDReduceOp::THDReducePRODUCT) {
      result_data[i] *= new_data[i];
    } else {
      throw std::logic_error("unsupported reduce operation");
    }
  }
}

} // namespace thd


#undef SYSCHECK
