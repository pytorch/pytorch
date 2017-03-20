#include "CommandChannel.hpp"
#include "../../base/ChannelEnvVars.hpp"
#include "../../base/ChannelUtils.hpp"

#include "../master/State.hpp"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

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
{
  rank_type world_size;
  std::tie(_port, world_size) = load_master_env();

  _connections.resize(world_size);
}

MasterCommandChannel::~MasterCommandChannel() {}

bool MasterCommandChannel::init() {
  std::tie(_connections[0].command_socket, std::ignore) = listen(_port);

  int socket;
  rank_type rank;
  for (std::size_t i = 1; i < _connections.size(); ++i) {
    std::tie(socket, std::ignore) = accept(_connections[0].command_socket);
    recv_bytes<rank_type>(socket, &rank, 1);
    _connections.at(rank).command_socket = socket;
  }

  /* Sending confirm byte is to test connection and make barrier for workers.
   * It allows to block connected workers until all remaining workers connect.
   */
  for (std::size_t i = 1; i < _connections.size(); ++i) {
    std::uint8_t confirm_byte = 1;
    send_bytes<std::uint8_t>(_connections[i].command_socket, &confirm_byte, 1);
  }


  for (std::size_t i = 1; i < _connections.size(); ++i) {
    std::tie(socket, std::ignore) = accept(_connections[0].command_socket);
    recv_bytes<rank_type>(socket, &rank, 1);
    _connections.at(rank).error_socket = socket;
  }

  /* Send confirm byte on `error_socket`.
   */
  for (std::size_t i = 1; i < _connections.size(); ++i) {
    std::uint8_t confirm_byte = 1;
    send_bytes<std::uint8_t>(_connections[i].error_socket, &confirm_byte, 1);
  }

   // close listen socket
  _connections[0].close();
  return true;
}

void MasterCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank) {
  if (thd::master::THDState::s_error != "") {
    throw std::runtime_error(thd::master::THDState::s_error);
  }

  if ((rank <= 0) || (rank >= _connections.size())) {
    throw std::domain_error("sendMessage received invalid rank as parameter");
  }

  ::thd::sendMessage(_connections[rank].command_socket, std::move(msg));
}

std::unique_ptr<rpc::RPCMessage> MasterCommandChannel::recvMessage(int rank) {
  if ((rank <= 0) || (rank >= _connections.size())) {
    throw std::domain_error("recvMessage received invalid rank as parameter");
  }

  return ::thd::receiveMessage(_connections[rank].command_socket);
}

std::tuple<rank_type, std::string> MasterCommandChannel::recvError() {
  if (!_poll_events) {
    // cache poll events array, it will be reused in another `receiveError` calls
    _poll_events.reset(new struct pollfd[_connections.size()]);
    for (std::size_t rank = 0; rank < _connections.size(); ++rank) {
      _poll_events[rank] = {
        .fd = _connections[rank].error_socket,
        .events = POLLIN
      };
    }
  }

  // cleanup
  for (std::size_t rank = 0; rank < _connections.size(); ++rank) {
    _poll_events[rank].revents = 0;
  }

  SYSCHECK(::poll(_poll_events.get(), _connections.size(), -1)) // infinite timeout
  for (std::size_t rank = 0; rank < _connections.size(); ++rank) {
    if (this->_poll_events[rank].revents == 0)
      continue;

    /* If `POLLIN` is not set or there is different flag set it means that
     * something is wrong.
     */
    if (_poll_events[rank].revents ^ POLLIN) {
      _poll_events[rank].fd = -1; // mark worker as ignored
      return std::make_tuple(rank, "Connection with worker has been closed");
    }

    // receive error
    std::uint64_t error_length;
    recv_bytes<std::uint64_t>(_poll_events[rank].fd, &error_length, 1);

    std::unique_ptr<char[]> error(new char[error_length + 1]);
    recv_bytes<char>(_poll_events[rank].fd, error.get(), error_length);
    return std::make_tuple(rank, std::string(error.get(), error_length));
  }

  throw std::runtime_error("Failed to receive error from worker");
}


WorkerCommandChannel::WorkerCommandChannel()
  : _connection()
{
  _rank = load_rank_env();
  std::tie(_master_addr, _master_port) = load_worker_env();
}

WorkerCommandChannel::~WorkerCommandChannel() {}

bool WorkerCommandChannel::init() {
  std::uint8_t confirm_byte;

  _connection.command_socket = connect(_master_addr, _master_port);
  send_bytes<rank_type>(_connection.command_socket, &_rank, 1); // send rank
  recv_bytes<std::uint8_t>(_connection.command_socket, &confirm_byte, 1);

  _connection.error_socket = connect(_master_addr, _master_port);
  send_bytes<rank_type>(_connection.error_socket, &_rank, 1);
  recv_bytes<std::uint8_t>(_connection.error_socket, &confirm_byte, 1);

  return true;
}

std::unique_ptr<rpc::RPCMessage> WorkerCommandChannel::recvMessage() {
  return ::thd::receiveMessage(_connection.command_socket);
}

void WorkerCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg) {
  ::thd::sendMessage(_connection.command_socket, std::move(msg));
}

void WorkerCommandChannel::sendError(const std::string& error) {
  std::uint64_t error_length = static_cast<std::uint64_t>(error.size());
  send_bytes<std::uint64_t>(_connection.error_socket, &error_length, 1, true);
  send_bytes<char>(_connection.error_socket, error.data(), error_length);
}

} // namespace thd
