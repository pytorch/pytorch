#pragma once

#include "RPC.hpp"

#include <sys/poll.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace thd {

struct MasterCommandChannel {
  MasterCommandChannel();
  ~MasterCommandChannel();

  bool init();

  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank);
  std::tuple<rank_type, std::string> recvError();

private:
  rank_type _rank;
  std::vector<int> _sockets;
  std::unique_ptr<struct pollfd[]> _poll_events;

  port_type _port;
};

struct WorkerCommandChannel {
  WorkerCommandChannel();
  ~WorkerCommandChannel();

  bool init();

  std::unique_ptr<rpc::RPCMessage> recvMessage();
  void sendError(const std::string& error);

private:
  rank_type _rank;
  int _socket;

  std::string _master_addr;
  port_type _master_port;
};

} // namespace thd
