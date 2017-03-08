#pragma once

#include "RPC.hpp"

#include <memory>
#include <string>
#include <vector>

namespace thd {

struct MasterCommandChannel {
  MasterCommandChannel();
  ~MasterCommandChannel();

  bool init();

  std::unique_ptr<rpc::RPCMessage> recvMessage(int rank);
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank);

private:
  rank_type _rank;
  std::vector<int> _sockets;

  port_type _port;
};

struct WorkerCommandChannel {
  WorkerCommandChannel();
  ~WorkerCommandChannel();

  bool init();

  std::unique_ptr<rpc::RPCMessage> recvMessage();
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg);

private:
  rank_type _rank;
  int _socket;

  std::string _master_addr;
  port_type _master_port;
};

} // namespace thd
