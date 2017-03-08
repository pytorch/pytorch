#include "DataChannelTCP.hpp"
#include "../ChannelUtils.hpp"

#include <sys/poll.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>


namespace thd {
namespace {

constexpr int MASTER_RANK = 0;


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
  _rank = load_rank_env();

  if (_rank == MASTER_RANK) { // MASTER
    std::uint32_t processes_number;
    std::tie(_port, processes_number) = load_master_env();

    _processes.resize(processes_number);
    _processes[_rank] = {
      .rank = _rank,
      .address = "",
      .port = 0,
      .socket = -1,
    };
  } else { // WORKER
    std::string address;
    std::uint16_t port;
    std::tie(address, port) = load_worker_env();

    // add master
    _processes.resize(1);
    _processes[MASTER_RANK] = {
      .rank = MASTER_RANK,
      .address = address,
      .port = port,
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


bool DataChannelTCP::initWorker() {
  auto& master = _processes[MASTER_RANK];
  master.socket = connect(master.address, master.port);
  int master_socket = master.socket;

  std::tie(_socket, _port) = listen();

  send_bytes<std::uint32_t>(master_socket, &_rank, 1);
  send_bytes<std::uint16_t>(master_socket, &_port, 1); // send listening port to master

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

  /*
   * Firstly we are connecting to workers with rank lower than our rank,
   * then we accepting connections from other wokers with higher rank.
   *
   * This prevents from deadlocks where everyone is accepting or everyone is
   * trying to connect.
   */

  for (std::uint32_t r = 1; r < _rank; ++r) {
    auto& process = _processes[r];
    process.socket = connect(process.address, process.port);

    // send rank to tell to the accepting process who we are
    std::uint32_t p_rank = static_cast<std::uint32_t>(_rank);
    send_bytes<std::uint32_t>(process.socket, &p_rank, 1);
  }

  for (std::uint32_t i = _rank + 1; i < _processes.size(); ++i) {
    int socket;
    std::tie(socket, std::ignore) = accept(_socket, _timeout);

    // get rank of process we have just accepted
    std::uint32_t p_rank;
    recv_bytes<std::uint32_t>(socket, &p_rank, 1);

    _processes[p_rank].socket = socket;
  }

  // close socket for listening, we will not use it anymore
  ::close(_socket);
  _socket = -1;

  return true;
}


bool DataChannelTCP::initMaster() {
  std::tie(_socket, std::ignore) = listen(_port);

  // wait for all workers to connect
  int workers = _processes.size() - 1;
  while (workers > 0) {
    std::string p_address;
    int p_socket;
    std::tie(p_socket, p_address) = accept(_socket, _timeout);

    std::uint32_t p_rank;
    std::uint16_t p_port;
    recv_bytes<std::uint32_t>(p_socket, &p_rank, 1);
    recv_bytes<std::uint16_t>(p_socket, &p_port, 1);

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
      .socket = p_socket,
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
