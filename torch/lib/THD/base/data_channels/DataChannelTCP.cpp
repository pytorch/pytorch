#include "DataChannelTCP.hpp"

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

constexpr rank_type MASTER_RANK = 0;

inline std::uint32_t log2ceil(std::uint32_t value) {
  std::uint32_t dim = 0;
#if defined(__GNUC__)
  if (value <= 1)
    return 0;
  dim = 32 - __builtin_clz(value - 1);
#else
  for (std::uint32_t size = 1; size < value; ++dim, size <<= 1) /* empty */;
#endif // defined(__GNUC__)
  return dim;
}

// Finds nearest power-of-two less than or equal to `value`.
template<typename T>
inline std::uint64_t pow2(T value) {
  std::uint64_t pof2 = 1;
  while (pof2 <= value) { pof2 <<= 1; }
  pof2 >>= 1;
  return pof2;
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


DataChannelTCP::DataChannelTCP(InitMethod::Config config)
  : DataChannelTCP(config, -1)
{}


DataChannelTCP::DataChannelTCP(InitMethod::Config config, int timeout)
  : _socket(-1)
  , _port(0)
  , _timeout(timeout)
  , _poll_events(nullptr)
{
  _rank = config.rank;

  if (_rank == MASTER_RANK) { // MASTER
    _socket = config.master.listen_socket;
    _port = config.master.listen_port;

    _processes.resize(config.master.world_size);
    _processes[_rank] = {
      .rank = _rank,
      .address = "",
      .port = 0,
      .socket = -1,
    };
  } else { // WORKER
    std::string address = config.worker.address;
    port_type port = config.worker.listen_port;

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

  std::tie(_socket, std::ignore, _port) = listen();

  send_bytes<rank_type>(master_socket, &_rank, 1, true);
  send_bytes<port_type>(master_socket, &_port, 1); // send listening port to master

  rank_type processes_number;
  recv_bytes<rank_type>(master_socket, &processes_number, 1);
  _processes.resize(processes_number);

  // get all metadata of other processes in network
  processes_number--; // exclude master
  while (processes_number > 0) {
    std::uint32_t p_address_len;
    rank_type p_rank;
    port_type p_port;

    recv_bytes<rank_type>(master_socket, &p_rank, 1); // get process rank
    recv_bytes<std::uint32_t>(master_socket, &p_address_len, 1); // get process address length

    // get process address
    std::unique_ptr<char[]> tmp_address(new char[p_address_len + 1]);
    recv_bytes<char>(master_socket, tmp_address.get(), p_address_len);

    recv_bytes<port_type>(master_socket, &p_port, 1); // get process port

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

  for (rank_type r = 1; r < _rank; ++r) {
    auto& process = _processes[r];
    process.socket = connect(process.address, process.port);

    // send rank to tell to the accepting process who we are
    send_bytes<rank_type>(process.socket, &_rank, 1);
  }

  for (rank_type i = _rank + 1; i < _processes.size(); ++i) {
    int socket;
    std::tie(socket, std::ignore) = accept(_socket, _timeout);

    // get rank of process we have just accepted
    rank_type p_rank;
    recv_bytes<rank_type>(socket, &p_rank, 1);

    _processes[p_rank].socket = socket;
  }

  // close socket for listening, we will not use it anymore
  ::close(_socket);
  _socket = -1;

  return true;
}


bool DataChannelTCP::initMaster() {
  // wait for all workers to connect
  std::size_t workers = _processes.size() - 1;
  while (workers > 0) {
    std::string p_address;
    int p_socket;
    std::tie(p_socket, p_address) = accept(_socket, _timeout);

    rank_type p_rank;
    port_type p_port;
    recv_bytes<rank_type>(p_socket, &p_rank, 1);
    recv_bytes<port_type>(p_socket, &p_port, 1);

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

    rank_type processes_number = _processes.size();
    send_bytes<rank_type>(worker.socket, &processes_number, 1, true);

    for (auto& process : _processes) {
      if (process.rank == _rank) continue;

      std::uint32_t proc_address_length = process.address.size();
      send_bytes<rank_type>(worker.socket, &process.rank, 1, true);
      send_bytes<std::uint32_t>(worker.socket, &proc_address_length, 1, true);
      send_bytes<char>(worker.socket, process.address.data(), proc_address_length, true);
      send_bytes<port_type>(worker.socket, &process.port, 1);
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
    std::vector<rank_type> ranks;
    ranks.reserve(_processes.size());
    for (rank_type rank = 0; rank < _processes.size(); ++rank)
      ranks.push_back(rank);

    _groups.insert({
      THDGroupWORLD,
      DataChannel::Group(ranks, _processes.size() - 1)
    });
  }

  return ok;
}


rank_type DataChannelTCP::getRank() {
  return _rank;
}


rank_type DataChannelTCP::getNumProcesses() {
  return _processes.size();
}


void DataChannelTCP::allGather(std::vector<thpp::Tensor*>& output,
                               thpp::Tensor& input, THDGroup group_id) {
  /*
   * Allgather algorithm is simple ring algorithm. This algorithm perfroms
   * well on large data (> 512 KB) and generalize well on large group of nodes.
   * More about efficiency can be found here:
   *   > http://www.mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf (section 4.1)
   *
   * TODO: implement Bruck / recursive doubling algorithms to make allGather
   * efficient also for small data (< 512 KB).
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;
  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  if (output.size() != group.size())
    throw std::logic_error("allGather: number of output tensors and group size does not match");

  for (auto out_tensor : output)
    assertSameSizeAndType(*out_tensor, input, "allGather");

  rank_type left = (group.size() + group_rank - 1) % group.size();
  rank_type right = (group_rank + 1) % group.size();

  memcpy(output[group_rank]->data(), input.data(), input.elementSize() * input.numel());

  auto j = group_rank, jnext = left;
  for (rank_type i = 0; i < group.size(); ++i) {
    auto send_request = isend(*(output[j]), group.mustGetGlobalRank(right));
    receive(*(output[jnext]), group.mustGetGlobalRank(left));
    send_request->wait();

    j = jnext;
    jnext = (group.size() + jnext - 1) % group.size();
  }
}


void DataChannelTCP::gather(std::vector<thpp::Tensor*>& output,
                            thpp::Tensor& input, rank_type dst_rank, THDGroup group_id) {
  std::lock_guard<std::mutex> lock(_mutex);

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
      assertSameSizeAndType(*out_tensor, input, "gather");

    for (rank_type i = 0; i < group.size(); ++i) {
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
                             thpp::Tensor& output, rank_type src_rank,
                             THDGroup group_id) {
  std::lock_guard<std::mutex> lock(_mutex);

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
      assertSameSizeAndType(*in_tensor, output, "scatter");

    for (rank_type i = 0; i < group.size(); ++i) {
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
   * Allreduce implementation is recursive doubling algorithm. It is good
   * algorithm for small sizes of message but other (theoratically better)
   * implementations could not be addapted because of non-commutative
   * operations on tensors (operation cannot be commutative because this could
   * introduce different numerical errors on different workers).
   *
   * More about efficiency can be found here:
   *   > http://www.mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf (section 4.5)
   *
   * Implementation is based on:
   *   > https://github.com/pmodels/mpich/blob/master/src/mpi/coll/allreduce.c
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  auto tmp_tensor = std::unique_ptr<thpp::Tensor>(data.clone());

  auto pof2 = pow2(group.size());
  int rem = group.size() - pof2;
  int newrank = 0;

  if (group_rank < 2 * rem) {
    if (group_rank % 2 == 0) {
      send(data, group.mustGetGlobalRank(group_rank + 1));
      newrank = -1;
    } else {
      receive(*tmp_tensor, group.mustGetGlobalRank(group_rank - 1));
      _reduce(data, *tmp_tensor, operation);
      newrank = group_rank / 2;
    }
  } else {
    newrank = group_rank - rem;
  }

  if (newrank != -1) {
    int mask = 0x1;
    while (mask < pof2) {
      int newdst = newrank ^ mask;
      int dst = (newdst < rem) ? (newdst * 2 + 1) : (newdst + rem);

      auto dst_global_rank = group.mustGetGlobalRank(dst);
      auto send_request = isend(data, dst_global_rank);
      receive(*tmp_tensor, dst_global_rank);
      send_request->wait();

      if (dst < group_rank) {
        _reduce(data, *tmp_tensor, operation);
      } else {
        _reduce(*tmp_tensor, data, operation);
        std::memcpy(data.data(), tmp_tensor->data(), tensor_bytes);
      }

      mask <<= 1;
    }
  }

  if (group_rank < 2 * rem) {
    if (group_rank % 2) {
      send(data, group.mustGetGlobalRank(group_rank - 1));
    } else {
      receive(data, group.mustGetGlobalRank(group_rank + 1));
    }
  }
}


void DataChannelTCP::reduce(thpp::Tensor& data, THDReduceOp operation,
                            rank_type dst_rank, THDGroup group_id) {
  /*
   * Idea of this algorithm is similar to broadcast but with reversed
   * order and direction of communication.
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  auto group_dst_rank = group.mustGetGroupRank(dst_rank);
  int dim = log2ceil(group.size());
  rank_type virtual_rank = (group_rank + group.size() - group_dst_rank) % group.size();
  long long mask = 0;
  auto result_tensor = std::unique_ptr<thpp::Tensor>(data.clone());

  for (int k = 0; k <= dim - 1; mask ^= (1 << k), ++k) {
    if ((virtual_rank & mask) == 0) {
      rank_type partner = virtual_rank ^ (1 << k); // partner has opposite bit `k`
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
}


void DataChannelTCP::broadcast(thpp::Tensor& data, rank_type src_rank,
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

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  auto group_src_rank = group.mustGetGroupRank(src_rank);
  int dim = log2ceil(group.size());
  rank_type virtual_rank = (group_rank + group.size() - group_src_rank) % group.size();
  long long mask = (1 << dim) - 1;

  for (int k = dim - 1; k >= 0; --k) {
    mask ^= (1 << k); // clear bit `k`
    if ((virtual_rank & mask) == 0) {
      rank_type partner = virtual_rank ^ (1 << k); // partner has opposite bit `k`
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


void DataChannelTCP::send(const Scalar& data, rank_type dst_rank) {
  auto request = _send_worker.push([this, &data, dst_rank]{
    this->_send(data, dst_rank);
  });
  request.wait();
}


void DataChannelTCP::send(thpp::Tensor& data, rank_type dst_rank) {
  auto request = _send_worker.push([this, &data, dst_rank]{
    this->_send(data, dst_rank);
  });
  request.wait();
}


void DataChannelTCP::receive(Scalar& data, rank_type src_rank) {
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
      for (std::size_t rank = 0; rank < this->_processes.size(); ++rank) {
        this->_poll_events[rank] = {
          .fd = this->_processes[rank].socket,
          .events = POLLIN
        };
      }
    }

    // cleanup
    for (std::size_t rank = 0; rank < this->_processes.size(); ++rank) {
      this->_poll_events[rank].revents = 0;
    }

    SYSCHECK(::poll(this->_poll_events.get(), this->_processes.size(), -1)) // infinite timeout
    for (std::size_t rank = 0; rank < this->_processes.size(); ++rank) {
      if (this->_poll_events[rank].revents == 0)
        continue;

      if (this->_poll_events[rank].revents ^ POLLIN)
        throw std::system_error(ECONNABORTED, std::system_category());

      this->_receive(data, rank);
      break;
    }
  });

  request.wait();
}


void DataChannelTCP::receive(thpp::Tensor& data, rank_type src_rank) {
  auto request = _receive_worker.push([this, &data, src_rank]{
    this->_receive(data, src_rank);
  });
  request.wait();
}


DataChannelTCP::RequestTCP* DataChannelTCP::isend(thpp::Tensor& data,
                                                  rank_type dst_rank) {
  std::shared_ptr<thpp::Tensor> copy_tensor(data.clone_shallow());
  auto request = _send_worker.push([this, copy_tensor, dst_rank]{
    this->_send(*copy_tensor, dst_rank);
  });
  return new DataChannelTCP::RequestTCP(std::move(request));
}


DataChannelTCP::RequestTCP* DataChannelTCP::ireceive(thpp::Tensor& data,
                                                     rank_type src_rank) {
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

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  std::uint8_t byte = 1;
  for (rank_type distance = 1; distance < group.size(); distance <<= 1) {
    rank_type recv_partner = (group_rank + group.size() - distance) % group.size();
    const auto& recv_process = _processes.at(group.mustGetGlobalRank(recv_partner));
    auto recv_request = _receive_worker.push([&recv_process, &byte]{
      recv_bytes<std::uint8_t>(recv_process.socket, &byte, 1);
    });

    rank_type send_partner = (group_rank + distance) % group.size();
    const auto& send_process = _processes.at(group.mustGetGlobalRank(send_partner));
    auto send_request = _send_worker.push([&send_process, &byte]{
      send_bytes<std::uint8_t>(send_process.socket, &byte, 1);
    });

    send_request.wait();
    recv_request.wait();
  }
}


THDGroup DataChannelTCP::newGroup(const std::vector<rank_type>& ranks) {
  auto new_group = DataChannel::Group(ranks, _processes.size() - 1);
  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());

  _groups.insert({new_group_id, new_group});
  return new_group_id;
}


void DataChannelTCP::_send(const Scalar& data, rank_type dst_rank) {
  /*
   * We have to check if dst_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  const auto& process_dst = _processes.at(dst_rank);
  if (process_dst.rank == _rank)
    throw std::logic_error("cannot send scalar to process with same rank");

  // send size of scalar in bytes
  std::uint64_t scalar_bytes = data.elementSize();
  send_bytes<std::uint64_t>(process_dst.socket, &scalar_bytes, 1, true);

  // send data (bytes)
  send_bytes<std::uint8_t>(
    process_dst.socket,
    reinterpret_cast<const std::uint8_t*>(data.data()),
    scalar_bytes
  );
}


void DataChannelTCP::_send(thpp::Tensor& data, rank_type dst_rank) {
  /*
   * We have to check if dst_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  const auto& process_dst = _processes.at(dst_rank);
  if (process_dst.rank == _rank)
    throw std::logic_error("cannot send tensor to process with same rank");

  if (!data.isContiguous())
    throw std::logic_error("tensor to send is not contiguous");

  // send size of tensor data in bytes
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  send_bytes<std::uint64_t>(process_dst.socket, &tensor_bytes, 1, true);

  // send data (bytes)
  send_bytes<std::uint8_t>(
    process_dst.socket,
    reinterpret_cast<const std::uint8_t*>(data.data()),
    tensor_bytes
  );
}


void DataChannelTCP::_receive(Scalar& data, rank_type src_rank) {
  /*
   * We have to check if src_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  const auto& process_src = _processes.at(src_rank);
  if (process_src.rank == _rank)
    throw std::logic_error("cannot receive scalar from process with same rank");

  // get size of scalar in bytes
  std::uint64_t scalar_bytes;
  recv_bytes<std::uint64_t>(process_src.socket, &scalar_bytes, 1);

  std::uint64_t actual_scalar_bytes = data.elementSize();
  if (actual_scalar_bytes == scalar_bytes) {
    recv_bytes<std::uint8_t>(
      process_src.socket,
      reinterpret_cast<std::uint8_t*>(data.data()),
      scalar_bytes
    );
  } else {
    // remove invalid data from recv buffer
    std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[scalar_bytes]);
    recv_bytes<std::uint8_t>(process_src.socket, bytes.get(), scalar_bytes);
    throw std::logic_error("scalar sizes do not match");
  }
}


void DataChannelTCP::_receive(thpp::Tensor& data, rank_type src_rank) {
  /*
   * We have to check if src_rank is positive to properly use `.at` function in vector.
   * Not checking that can result in int overflow and strange errors.
   */

  const auto& process_src = _processes.at(src_rank);
  if (process_src.rank == _rank)
    throw std::logic_error("cannot receive tensor from process with same rank");

  if (!data.isContiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  // get size of tensor data in bytes
  std::uint64_t tensor_bytes;
  recv_bytes<std::uint64_t>(process_src.socket, &tensor_bytes, 1);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes == tensor_bytes) {
    recv_bytes<std::uint8_t>(
      process_src.socket,
      reinterpret_cast<std::uint8_t*>(data.data()),
      tensor_bytes
    );
  } else {
    // remove invalid data from recv buffer
    std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
    recv_bytes<std::uint8_t>(process_src.socket, bytes.get(), tensor_bytes);
    throw std::logic_error("tensor sizes do not match");
  }
}

void DataChannelTCP::_reduce(thpp::Tensor& result, thpp::Tensor& data,
                             THDReduceOp operation) const {
  assertSameSizeAndType(result, data, "reduce");

  if (operation == THDReduceOp::THDReduceMIN) {
    result.cmin(result, data);
  } else if (operation == THDReduceOp::THDReduceMAX) {
    result.cmax(result, data);
  } else if (operation == THDReduceOp::THDReduceSUM) {
    result.cadd(result, data);
  } else if (operation == THDReduceOp::THDReducePRODUCT) {
    result.cmul(result, data);
  } else {
    throw std::logic_error("unsupported reduce operation");
  }
}

} // namespace thd
