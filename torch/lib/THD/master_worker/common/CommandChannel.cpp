#include "CommandChannel.hpp"
#include "Functions.hpp"
#include "../../base/ChannelEnvVars.hpp"
#include "../../base/ChannelUtils.hpp"

#include <unistd.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace thd {
namespace {

void sendMessage(int socket, std::unique_ptr<rpc::RPCMessage> msg) {
  auto& bytes = msg.get()->bytes();
  std::uint64_t msg_length = static_cast<std::uint64_t>(bytes.length());

  send_bytes<std::uint64_t>(socket, &msg_length, 1, true);
  send_bytes<std::uint8_t>(
    socket,
    reinterpret_cast<const std::uint8_t*>(bytes.data()),
    msg_length
  );
}

std::unique_ptr<rpc::RPCMessage> receiveMessage(int socket) {
  std::uint64_t msg_length;
  recv_bytes<std::uint64_t>(socket, &msg_length, 1);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[msg_length]);
  recv_bytes<std::uint8_t>(socket, bytes.get(), msg_length);

  return std::unique_ptr<rpc::RPCMessage>(
    new rpc::RPCMessage(reinterpret_cast<char*>(bytes.get()), msg_length)
  );
}

} // anonymous namespace

MasterCommandChannel::MasterCommandChannel()
  : _rank(0)
  , _poll_events(nullptr)
  , _error_pipe(-1)
  , _error(nullptr)
{
  rank_type world_size;
  std::tie(_port, world_size) = load_master_env();

  _sockets.assign(world_size, -1);
  _mutexes = std::vector<std::mutex>(world_size);
}

MasterCommandChannel::~MasterCommandChannel() {
  if (_error_thread.joinable()) {
    if (::write(_error_pipe, "exit", 4) != 4) {
      std::cerr << "Failed to notify error thread" << std::endl;
    }
    _error_thread.join();

    ::close(_error_pipe);
  }

  auto world_size = _sockets.size();
  for (std::size_t i = 0; i < world_size; ++i) {
    auto socket = _sockets[i];
    if (socket == -1) continue;
    try {
      sendMessage(rpc::packMessage(Functions::exit), i);
    } catch(...) {}
    ::close(socket);
  }

}

bool MasterCommandChannel::init() {
  std::tie(_sockets[0], std::ignore, std::ignore) = listen(_port);

  int socket;
  rank_type rank;
  for (std::size_t i = 1; i < _sockets.size(); ++i) {
    std::tie(socket, std::ignore) = accept(_sockets[0]);
    recv_bytes<rank_type>(socket, &rank, 1);
    _sockets.at(rank) = socket;
  }

  /* Sending confirm byte is to test connection and make barrier for workers.
   * It allows to block connected workers until all remaining workers connect.
   */
  for (std::size_t i = 1; i < _sockets.size(); ++i) {
    std::uint8_t confirm_byte = 1;
    send_bytes<std::uint8_t>(_sockets[i], &confirm_byte, 1);
  }

   // close listen socket
  ::close(_sockets[0]);

  int fd[2];
  SYSCHECK(::pipe(fd));
  _sockets[0] = fd[0];
  _error_pipe = fd[1];
  _error_thread = std::thread(&MasterCommandChannel::errorHandler, this);
  return true;
}

void MasterCommandChannel::errorHandler() {
  while (true) {
    auto error = recvError();
    if (std::get<0>(error) == 0) {
      return;
    }

    _error.reset(new std::string(
      "error (rank " + std::to_string(std::get<0>(error)) + "): " + std::get<1>(error)
    ));
  }
}

void MasterCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank) {
  // Throw error received from a worker.
  if (_error) {
    throw std::runtime_error(*_error);
  }

  if ((rank <= 0) || (rank >= _sockets.size())) {
    throw std::domain_error("sendMessage received invalid rank as parameter");
  }

  std::lock_guard<std::mutex> guard(_mutexes[rank]);
  ::thd::sendMessage(_sockets[rank], std::move(msg));
}

std::tuple<rank_type, std::string> MasterCommandChannel::recvError() {
  if (!_poll_events) {
    // cache poll events array, it will be reused in another `receiveError` calls
    _poll_events.reset(new struct pollfd[_sockets.size()]);
    for (std::size_t rank = 0; rank < _sockets.size(); ++rank) {
      _poll_events[rank] = {
        .fd = _sockets[rank],
        .events = POLLIN
      };
    }
  }

  for (std::size_t rank = 0; rank < _sockets.size(); ++rank) {
    _poll_events[rank].revents = 0;
  }

  SYSCHECK(::poll(_poll_events.get(), _sockets.size(), -1))
  for (std::size_t rank = 0; rank < _sockets.size(); ++rank) {
    if (this->_poll_events[rank].revents == 0)
      continue;

    if (rank == 0) { // we are notified by master to end
      return std::make_tuple(0, "");
    }

    if (_poll_events[rank].revents ^ POLLIN) {
      _poll_events[rank].fd = -1; // mark worker as ignored
      return std::make_tuple(rank, "connection with worker has been closed");
    }

    try {
      // receive error
      std::uint64_t error_length;
      recv_bytes<std::uint64_t>(_poll_events[rank].fd, &error_length, 1);

      std::unique_ptr<char[]> error(new char[error_length]);
      recv_bytes<char>(_poll_events[rank].fd, error.get(), error_length);
      return std::make_tuple(rank, std::string(error.get(), error_length));
    } catch (const std::exception& e) {
      return std::make_tuple(rank, "recv: " + std::string(e.what()));
    }
  }

  // We did not receive error from any worker despite being notified.
  return std::make_tuple(0, "failed to receive error from worker");
}


WorkerCommandChannel::WorkerCommandChannel()
  : _socket(-1)
{
  _rank = load_rank_env();
  std::tie(_master_addr, _master_port) = load_worker_env();
}

WorkerCommandChannel::~WorkerCommandChannel() {
  if (_socket != -1)
    ::close(_socket);
}

bool WorkerCommandChannel::init() {
  _socket = connect(_master_addr, _master_port);
  send_bytes<rank_type>(_socket, &_rank, 1); // send rank

  std::uint8_t confirm_byte;
  recv_bytes<std::uint8_t>(_socket, &confirm_byte, 1);
  return true;
}

std::unique_ptr<rpc::RPCMessage> WorkerCommandChannel::recvMessage() {
  return ::thd::receiveMessage(_socket);
}

void WorkerCommandChannel::sendError(const std::string& error) {
  std::uint64_t error_length = static_cast<std::uint64_t>(error.size());
  send_bytes<std::uint64_t>(_socket, &error_length, 1, true);
  send_bytes<char>(_socket, error.data(), error_length);
}

} // namespace thd
