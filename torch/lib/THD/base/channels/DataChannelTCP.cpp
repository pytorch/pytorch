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


DataChannelTCP::DataChannelTCP()
  : DataChannelTCP(-1)
{}


DataChannelTCP::DataChannelTCP(int timeout)
  : _socket(0)
  , _port(0)
  , _timeout(timeout)
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
  ::close(_socket);

  for (const auto& process : _processes) {
    if ((process.rank != _rank) && (process.socket != -1))
      ::close(process.socket);
  }
}


void DataChannelTCP::listen(std::uint16_t port = 0) {
  SYSCHECK(_socket = ::socket(PF_INET, SOCK_STREAM, 0))

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);

  memset(&addr, 0, addr_len);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;

  int optval = 1;
  SYSCHECK(::setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)))
  SYSCHECK(::bind(_socket, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)))
  SYSCHECK(::listen(_socket, LISTEN_QUEUE_SIZE))
  SYSCHECK(::getsockname(_socket, reinterpret_cast<struct sockaddr*>(&addr), &addr_len))

  _port = ntohs(addr.sin_port);
}


int DataChannelTCP::connect(const std::string& address, std::uint16_t port,
                            int wait = true) const {
  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);

  memset(&addr, 0, addr_len);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  struct addrinfo *res;

  // get address by host or IP
  int err = ::getaddrinfo(address.data(), NULL, NULL, &res);
  if (err == 0) {
    std::memcpy(
      &(addr.sin_addr),
      &(reinterpret_cast<struct sockaddr_in*>(res->ai_addr)->sin_addr),
      sizeof(struct in_addr)
    );
    ::freeaddrinfo(res);
  } else {
    SYSCHECK(err = ::inet_pton(AF_INET, address.data(), &(addr.sin_addr)))
    if (err == 0)
      throw std::invalid_argument("invalid IP address");
  }

  int socket;
  while (true) {
    try {
      /*
       * If connect() fails, the state of the socket is unspecified.
       * We should close the socket and create a new one before attempting to reconnect.
       */
      SYSCHECK(socket = ::socket(AF_INET, SOCK_STREAM, 0))
      SYSCHECK(::connect(socket, reinterpret_cast<const struct sockaddr*>(&addr), addr_len))
      break;
    } catch (const std::system_error& e) {
      ::close(socket);
      if (!wait || (errno != ECONNREFUSED))
        throw e;

      std::this_thread::sleep_for(std::chrono::seconds(1));
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

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  std::memset(&addr, 0, sizeof(addr));

  int socket;
  SYSCHECK(socket = ::accept(_socket, reinterpret_cast<struct sockaddr*>(&addr), &addr_len))

  char address[INET_ADDRSTRLEN + 1];
  SYSCHECK(::inet_ntop(AF_INET, &(addr.sin_addr), address, INET_ADDRSTRLEN))
  address[INET_ADDRSTRLEN] = '\0';

  return std::make_tuple(socket, std::string(address));
}


bool DataChannelTCP::initWorker() {
  auto& master = _processes[MASTER_RANK];
  master.socket = connect(master.address, master.port);

  listen();

  std::uint32_t p_rank = (std::uint32_t)_rank;
  std::uint16_t p_port = (std::uint16_t)_port;
  send_bytes<std::uint32_t>(master.socket, &p_rank, 1);
  send_bytes<std::uint16_t>(master.socket, &p_port, 1); // send listening port to master

  std::uint32_t processes_number;
  recv_bytes<std::uint32_t>(master.socket, &processes_number, 1);
  _processes.resize(processes_number);

  // get all metadata of other processes in network
  processes_number--; // exclude master
  while (processes_number > 0) {
    std::uint32_t p_rank, p_address_len;
    std::uint16_t p_port;

    recv_bytes<std::uint32_t>(master.socket, &p_rank, 1); // get process rank
    recv_bytes<std::uint32_t>(master.socket, &p_address_len, 1); // get process address length

    // get process address
    std::unique_ptr<char[]> tmp_address(new char[p_address_len + 1]);
    recv_bytes<char>(master.socket, tmp_address.get(), p_address_len);

    recv_bytes<std::uint16_t>(master.socket, &p_port, 1); // get process port

    _processes[p_rank] = {
      .rank = p_rank,
      .address = std::string(tmp_address.get(), p_address_len),
      .port = p_port,
      .socket = 0,
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


void DataChannelTCP::allGather(std::vector<Tensor*>& output, Tensor& input, THDGroup group_id) {
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


void DataChannelTCP::gather(std::vector<Tensor*>& output, Tensor& input, int dst_rank, THDGroup group_id) {
  const auto& group = _groups.at(group_id);
  bool exists;

  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  if (_rank != dst_rank) {
    send(input, dst_rank);
  } else {
    if (output.size() != group.size())
      throw std::logic_error("gather: number of output tensors and group size does not match");

    for (auto out_tensor : output)
      assertTensorEqual(*out_tensor, input, "gather");

    for (std::size_t i = 0; i < group.size(); ++i) {
      // TODO: change it to some kind of helper
      auto group_rank = group.mustGetGlobalRank(i);
      if (_rank != group_rank) {
        receive(*(output.at(i)), group_rank);
      } else {
        memcpy(output.at(i)->data(), input.data(), input.numel() * input.elementSize());
      }
    }
  }
}


void DataChannelTCP::scatter(std::vector<Tensor*>& input, Tensor& output, int src_rank, THDGroup group_id) {
  const auto& group = _groups.at(group_id);
  bool exists;

  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  if (_rank != src_rank) {
    receive(output, src_rank);
  } else {
    if (input.size() != group.size())
      throw std::logic_error("scatter: number of input tensors and group size does not match");

    for (auto in_tensor : input)
      assertTensorEqual(*in_tensor, output, "scatter");

    for (std::size_t i = 0; i < group.size(); ++i) {
      // TODO: change it to some kind of helper
      auto group_rank = group.mustGetGlobalRank(i);
      if (_rank != group_rank) {
        send(*(input.at(i)), group_rank);
      } else {
        memcpy(output.data(), input.at(i)->data(), output.numel() * output.elementSize());
      }
    }
  }
}


void DataChannelTCP::allReduce(Tensor& data, THDReduceOp operation, THDGroup group_id) {
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


void DataChannelTCP::reduce(Tensor& data, THDReduceOp operation, int dst_rank, THDGroup group_id) {
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
        reduce_(*result_tensor, data, operation);
      }
    }
  }

  if (_rank == dst_rank)
    std::memcpy(data.data(), result_tensor->data(), data.elementSize() * data.numel());

  delete result_tensor;
}


void DataChannelTCP::broadcast(Tensor& data, int src_rank, THDGroup group_id) {
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


void DataChannelTCP::send(Tensor& data, int dst_rank) {
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


void DataChannelTCP::receive(Tensor& data, int src_rank) {
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


void DataChannelTCP::reduce_(Tensor& result, Tensor& data, THDReduceOp operation) const {
  assertTensorEqual(result, data, "reduce");

  Type tensor_type = data.type();
  switch(tensor_type) {
    case Type::CHAR:   reduce_<char>(result, data, operation); break;
    case Type::FLOAT:  reduce_<float>(result, data, operation); break;
    case Type::DOUBLE: reduce_<double>(result, data, operation); break;
    case Type::SHORT:  reduce_<short>(result, data, operation); break;
    case Type::USHORT: reduce_<unsigned short>(result, data, operation); break;
    case Type::INT:    reduce_<int>(result, data, operation); break;
    case Type::UINT:   reduce_<unsigned int>(result, data, operation); break;
    case Type::LONG:   reduce_<long>(result, data, operation); break;
    case Type::ULONG:  reduce_<unsigned long>(result, data, operation); break;
    case Type::LONG_LONG:  reduce_<long long>(result, data, operation); break;
    case Type::ULONG_LONG: reduce_<unsigned long long>(result, data, operation); break;
    default:
      throw std::logic_error("unsupported tensor type in reduce");
  }
}


template<typename T>
void DataChannelTCP::reduce_(Tensor& result, Tensor& data, THDReduceOp operation) const {
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
    // start aka asynchronous recv
    auto async_recv = std::async(std::launch::async, [&recv_process, &byte]() {
      recv_bytes<std::uint8_t>(recv_process.socket, &byte, 1);
    });


    int send_partner = (group_rank + distance) % group.size();
    const auto& send_process = _processes.at(group.mustGetGlobalRank(send_partner));
    send_bytes<std::uint8_t>(send_process.socket, &byte, 1);

    // if future is not valid before `wait`, it can result in undefined behaviour
    if (!async_recv.valid())
      throw std::future_error(std::future_errc::no_state);

    async_recv.wait(); // wait for recv to complete
  }
}


THDGroup DataChannelTCP::newGroup(const std::vector<int>& ranks) {
  auto new_group = DataChannel::Group(ranks, _processes.size() - 1);
  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());

  _groups.insert({new_group_id, new_group});
  return new_group_id;
}

} // namespace thd


#undef SYSCHECK
