#include "CommandChannel.hpp"
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
{
  rank_type world_size;
  std::tie(_port, world_size) = load_master_env();

  _sockets.resize(world_size);
  std::fill(_sockets.begin(), _sockets.end(), -1);
}

MasterCommandChannel::~MasterCommandChannel() {
  for (auto socket : _sockets) {
    if (socket != -1)
      ::close(socket);
  }
}

bool MasterCommandChannel::init() {
  std::tie(_sockets[0], std::ignore) = listen(_port);

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
  _sockets[0] = -1;
  return true;
}

void MasterCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank) {
  if ((rank <= 0) || (rank >= _sockets.size())) {
    throw std::domain_error("sendMessage received invalid rank as parameter");
  }

  ::thd::sendMessage(_sockets[rank], std::move(msg));
}

std::unique_ptr<rpc::RPCMessage> MasterCommandChannel::recvMessage(int rank) {
  if ((rank <= 0) || (rank >= _sockets.size())) {
    throw std::domain_error("recvMessage received invalid rank as parameter");
  }

  return receiveMessage(_sockets[rank]);
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

void WorkerCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg) {
  ::thd::sendMessage(_socket, std::move(msg));
}

std::unique_ptr<rpc::RPCMessage> WorkerCommandChannel::recvMessage() {
  return receiveMessage(_socket);
}

} // namespace thd
