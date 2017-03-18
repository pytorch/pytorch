#pragma once

#include "RPC.hpp"

#include <unistd.h>
#include <sys/poll.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace thd {
namespace command_channel {

struct Connection {
  Connection() : command_socket(-1), error_socket(-1) {}
  ~Connection() {
    close();
  }

  void close() {
    if (command_socket != -1)
      ::close(command_socket);
    command_socket = -1;

    if (error_socket != -1)
      ::close(error_socket);
    error_socket = -1;
  }

  int command_socket;
  int error_socket;
};

} // namespace command_channel

struct MasterCommandChannel {
  MasterCommandChannel();
  ~MasterCommandChannel();

  bool init();

  std::unique_ptr<rpc::RPCMessage> recvMessage(int rank);
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank);
  std::tuple<rank_type, std::string> recvError();

private:
  rank_type _rank;
  std::vector<command_channel::Connection> _connections;
  std::unique_ptr<struct pollfd[]> _poll_events;

  port_type _port;
};

struct WorkerCommandChannel {
  WorkerCommandChannel();
  ~WorkerCommandChannel();

  bool init();

  std::unique_ptr<rpc::RPCMessage> recvMessage();
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg);
  void sendError(const std::string& error);

private:
  rank_type _rank;
  command_channel::Connection _connection;

  std::string _master_addr;
  port_type _master_port;
};

} // namespace thd
