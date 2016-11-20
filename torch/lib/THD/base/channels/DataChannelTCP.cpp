#include "DataChannelTCP.hpp"

#include <arpa/inet.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <netdb.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <stdexcept>
#include <thread>


#define SYSCHECK(expr) { \
  errno = 0; (expr);     \
  if (errno != 0) throw std::system_error(errno, std::system_category()); \
}

namespace thd {
namespace {

constexpr int MASTER_RANK = 0;
constexpr int LISTEN_QUEUE_SIZE = 64;

template<typename T>
void send_bytes(int socket, const T* buffer, size_t length)
{
  size_t bytes_to_send = sizeof(T) * length;
  if (bytes_to_send == 0)
    return;

  auto bytes = reinterpret_cast<const uint8_t*>(buffer);
  uint8_t *current_bytes = const_cast<uint8_t*>(bytes);

  while (bytes_to_send > 0) {
    ssize_t bytes_sent;
    SYSCHECK(bytes_sent = ::send(socket, current_bytes, bytes_to_send, 0))
    if (bytes_sent == 0) {
      throw std::system_error(EBADMSG, std::system_category());
    }

    bytes_to_send -= bytes_sent;
    current_bytes += bytes_sent;
  }
}


template<typename T>
void recv_bytes(int socket, T* buffer, size_t length)
{
  size_t bytes_to_receive = sizeof(T) * length;
  if (bytes_to_receive == 0)
    return;

  auto bytes = reinterpret_cast<uint8_t*>(buffer);
  uint8_t *current_bytes = bytes;

  while (bytes_to_receive > 0) {
    ssize_t bytes_received;
    SYSCHECK(bytes_received = ::recv(socket, current_bytes, bytes_to_receive, 0))
    if (bytes_received == 0) {
      throw std::system_error(EBADMSG, std::system_category());
    }

    bytes_to_receive -= bytes_received;
    current_bytes += bytes_received;
  }
}


inline bool validatePort(int port) {
  return (port > 0 && port < 65536);
}

} // namespace


DataChannelTCP::DataChannelTCP()
  : DataChannelTCP(-1)
{}


DataChannelTCP::DataChannelTCP(int timeout)
  : m_socket(0)
  , m_port(0)
  , m_timeout(timeout)
{
  auto rank_env = std::getenv(RANK_ENV);
  if (!rank_env) {
    throw std::domain_error("env variable not found: " + std::string(RANK_ENV));
  }

  m_rank = std::stoi(rank_env);
  if (m_rank == MASTER_RANK) { // MASTER
    auto master_port_env = std::getenv(MASTER_PORT_ENV);
    if (!master_port_env) {
      throw std::domain_error("env variable not found: " + std::string(MASTER_PORT_ENV));
    }

    m_port = std::stoul(master_port_env);
    if (!validatePort(m_port)) {
      throw std::domain_error("invalid listen port number");
    }

    auto num_proceses_env = std::getenv(WORLD_SIZE_ENV);
    if (!num_proceses_env) {
      throw std::domain_error("env variable not found: " + std::string(WORLD_SIZE_ENV));
    }

    int processes_number = std::stoul(num_proceses_env);
    if (processes_number == 0) {
      throw std::domain_error("invalid " + std::string(WORLD_SIZE_ENV) + " env variable");
    }

    m_processes.resize(processes_number);
    m_processes[m_rank] = {
      .rank = static_cast<uint32_t>(m_rank),
      .address = "",
      .port = 0,
      .socket = -1,
    };
  } else { // WORKER
    auto master_addr_env = std::getenv(MASTER_ADDR_ENV);
    if (!master_addr_env) {
      throw std::domain_error("env variable not found: " + std::string(MASTER_ADDR_ENV));
    }

    std::string full_address = std::string(master_addr_env);
    auto found_pos = full_address.rfind(":");
    if (found_pos == std::string::npos) {
      throw std::domain_error("invalid master address, usage: IP:PORT | HOSTNAME:PORT");
    }

    std::string str_port = full_address.substr(found_pos + 1);
    int port = std::stoul(str_port);
    if (!validatePort(port)) {
      throw std::domain_error("invalid master port number");
    }

    // add master
    m_processes.resize(MASTER_RANK + 1);
    m_processes[MASTER_RANK] = {
      .rank = MASTER_RANK,
      .address = full_address.substr(0, found_pos),
      .port = static_cast<uint16_t>(port),
      .socket = -1,
    };
  }
}


DataChannelTCP::~DataChannelTCP()
{
  ::close(m_socket);

  for (const auto& process : m_processes) {
    if ((process.rank != m_rank) && (process.socket != -1))
      ::close(process.socket);
  }
}


void DataChannelTCP::listen(uint16_t port = 0) {
  SYSCHECK(m_socket = ::socket(PF_INET, SOCK_STREAM, 0))

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);

  memset(&addr, 0, addr_len);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;

  int optval = 1;
  SYSCHECK(::setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)))
  SYSCHECK(::bind(m_socket, (struct sockaddr*)&addr, sizeof(addr)))
  SYSCHECK(::listen(m_socket, LISTEN_QUEUE_SIZE))
  SYSCHECK(::getsockname(m_socket, (struct sockaddr*)&addr, &addr_len))

  m_port = ntohs(addr.sin_port);
}


int DataChannelTCP::connect(const std::string& address, uint16_t port,
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
    std::memcpy(&(addr.sin_addr), &((struct sockaddr_in*)res->ai_addr)->sin_addr, sizeof(struct in_addr));
    ::freeaddrinfo(res);
  } else {
    SYSCHECK(err = ::inet_pton(AF_INET, address.data(), &(addr.sin_addr)))
    if (err == 0) {
      throw std::invalid_argument("invalid IP address");
    }
  }

  int socket;
  while (true) {
    try {
      /*
       * If connect() fails, the state of the socket is unspecified.
       * We should close the socket and create a new one before attempting to reconnect.
       */
      SYSCHECK(socket = ::socket(AF_INET, SOCK_STREAM, 0))
      SYSCHECK(::connect(socket, (const struct sockaddr*)&addr, addr_len))
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
  events[0] = {.fd = m_socket, .events = POLLIN};

  int res;
  SYSCHECK(res = ::poll(events.get(), 1, m_timeout))
  if (res == 0) {
    throw std::runtime_error("waiting for processes to connect has timed out");
  } else {
    if (!(events[0].revents & POLLIN)) {
      throw std::system_error(ECONNABORTED, std::system_category());
    }
  }

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  std::memset(&addr, 0, sizeof(addr));

  int socket;
  SYSCHECK(socket = ::accept(m_socket, (struct sockaddr*)&addr, &addr_len))

  char address[INET_ADDRSTRLEN + 1];
  SYSCHECK(::inet_ntop(AF_INET, &(addr.sin_addr), address, INET_ADDRSTRLEN))
  address[INET_ADDRSTRLEN] = '\0';

  return std::make_tuple(socket, std::string(address));
}


bool DataChannelTCP::initWorker() {
  auto& master = m_processes[MASTER_RANK];
  master.socket = connect(master.address, master.port);

  listen();

  uint32_t p_rank = (uint32_t)m_rank;
  uint16_t p_port = (uint16_t)m_port;
  send_bytes<uint32_t>(master.socket, &p_rank, 1);
  send_bytes<uint16_t>(master.socket, &p_port, 1); // send listening port to master

  uint32_t processes_number;
  recv_bytes<uint32_t>(master.socket, &processes_number, 1);
  m_processes.resize(processes_number);

  // get all metadata of other processes in network
  processes_number--; // exclude master
  while (processes_number > 0) {
    uint32_t p_rank, p_address_len;
    uint16_t p_port;

    recv_bytes<uint32_t>(master.socket, &p_rank, 1); // get process rank
    recv_bytes<uint32_t>(master.socket, &p_address_len, 1); // get process address length

    // get process address
    std::unique_ptr<char[]> tmp_address(new char[p_address_len + 1]);
    recv_bytes<char>(master.socket, tmp_address.get(), p_address_len);

    recv_bytes<uint16_t>(master.socket, &p_port, 1); // get process port

    m_processes[p_rank] = {
      .rank = p_rank,
      .address = std::string(tmp_address.get(), p_address_len),
      .port = p_port,
      .socket = 0,
    };

    processes_number--;
  }

  // make network connection with other processes
  for (auto& process : m_processes) {
    if ((process.rank == m_rank) || (process.rank == MASTER_RANK)) continue;

    // it is to prevent accept-connect deadlock
    if (process.rank < m_rank) {
      process.socket = connect(process.address, process.port);
    } else {
      auto accept_state = accept();
      process.socket = std::get<0>(accept_state);
    }
  }

  return true;
}


bool DataChannelTCP::initMaster() {
  listen(m_port);

  // wait for all workers to connect
  int workers = m_processes.size() - 1;
  while (workers > 0) {
    auto accept_state = accept();
    int socket = std::get<0>(accept_state);
    std::string p_address = std::get<1>(accept_state);

    uint32_t p_rank;
    uint16_t p_port;
    recv_bytes<uint32_t>(socket, &p_rank, 1);
    recv_bytes<uint16_t>(socket, &p_port, 1);

    if (p_rank >= m_processes.size()) {
      throw std::out_of_range(
        "worker's rank(" + std::to_string(p_rank) + ") is out"
        "of range: [0, " + std::to_string(m_processes.size() - 1) + "]"
      );
    }

    if (m_processes[p_rank].rank == p_rank) {
      throw std::logic_error(
        "two processes (" + m_processes[p_rank].address + ", " + p_address + ") "
        "reported a rank of " + std::to_string(p_rank)
      );
    }

    m_processes[p_rank] = {
      .rank = p_rank,
      .address = p_address,
      .port = p_port,
      .socket = socket,
    };

    workers--;
  }

  // send informations about processes to all workers
  for (const auto& worker : m_processes) {
    if (worker.rank == m_rank) continue;

    uint32_t processes_number = m_processes.size();
    send_bytes<uint32_t>(worker.socket, &processes_number, 1);

    for (auto& process : m_processes) {
      if (process.rank == m_rank) continue;

      uint32_t proc_address_length = process.address.size();
      send_bytes<uint32_t>(worker.socket, &process.rank, 1);
      send_bytes<uint32_t>(worker.socket, &proc_address_length, 1);
      send_bytes<char>(worker.socket, process.address.data(), proc_address_length);
      send_bytes<uint16_t>(worker.socket, &(process.port), 1);
    }
  }

  return true;
}


bool DataChannelTCP::init() {
  if (m_rank == MASTER_RANK) {
    return initMaster();
  }

  return initWorker();
}


int DataChannelTCP::getRank() const {
  return m_rank;
}


int DataChannelTCP::getNumProcesses() const {
  return m_processes.size();
}


void DataChannelTCP::allReduce(Tensor& data) {
  // TODO: implement
}


void DataChannelTCP::reduce(Tensor& data, int dst_rank) {
  // TODO: implement
}


void DataChannelTCP::broadcast(Tensor& data, int src_rank) {
  if (src_rank != m_rank) {
    receive(data, src_rank);
  } else {
    /*
     * NOTE: This can be inefficient because send can block entire broadcast.
     * There can be used poll or select.
     */
    for (const auto& process : m_processes) {
      if (process.rank != m_rank)
        send(data, process.rank);
    }
  }
}


void DataChannelTCP::send(Tensor& data, int dst_rank) {
  const auto& process_dst = m_processes[dst_rank];
  if (process_dst.rank == m_rank) {
    throw std::logic_error("cannot send tensor to process with same rank");
  }

  if (!data.isContiguous()) {
    throw std::logic_error("tensor to send is not contiguous");
  }

  // send size of tensor data in bytes
  uint64_t tensor_bytes = data.elementSize() * data.numel();
  send_bytes<uint64_t>(process_dst.socket, &tensor_bytes, 1);

  // send data (bytes)
  send_bytes<uint8_t>(
    process_dst.socket,
    reinterpret_cast<const uint8_t*>(data.data()),
    tensor_bytes
  );
}


void DataChannelTCP::receive(Tensor& data, int src_rank) {
  const auto& process_src = m_processes[src_rank];
  if (process_src.rank == m_rank) {
    throw std::logic_error("cannot receive tensor from process with same rank");
  }

  if (!data.isContiguous()) {
    throw std::logic_error("tensor to receive is not contiguous");
  }

  // get size of tensor data in bytes
  uint64_t tensor_bytes;
  recv_bytes<uint64_t>(process_src.socket, &tensor_bytes, 1);

  // recv data (bytes)
  std::unique_ptr<uint8_t[]> bytes(new uint8_t[tensor_bytes]);
  recv_bytes<uint8_t>(process_src.socket, bytes.get(), tensor_bytes);

  uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes) {
    throw std::logic_error("tensor sizes does not match");
  }

  std::memcpy(data.data(), bytes.get(), tensor_bytes);
}

} // namespace thd


#undef SYSCHECK
