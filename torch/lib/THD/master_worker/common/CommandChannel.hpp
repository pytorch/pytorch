#pragma once

#include "RPC.hpp"

#include <memory>
#include <string>
#include <vector>

#include <zmq.hpp>

namespace thd {

struct MasterCommandChannel {
  MasterCommandChannel();
  ~MasterCommandChannel();

  std::unique_ptr<rpc::RPCMessage> recvMessage(int rank);
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank);

private:
  zmq::context_t _context;
  const int _rank;
  int _world_size; // MUST be declared before sockets.
  std::vector<zmq::socket_t> _pull_sockets;
  std::vector<zmq::socket_t> _push_sockets;
  std::vector<std::string> _pull_endpoints;
  std::vector<std::string> _push_endpoints;
};

struct WorkerCommandChannel {
  WorkerCommandChannel(int rank);
  ~WorkerCommandChannel();

  std::unique_ptr<rpc::RPCMessage> recvMessage();
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg);

private:
  zmq::context_t _context; // MUST be declared before sockets.
  zmq::socket_t _pull_socket;
  zmq::socket_t _push_socket;
  std::string _pull_endpoint;
  std::string _push_endpoint;
  int _rank;
};

} // namespace thd
