#pragma once

#include "../../../master_worker/common/RPC.hpp"

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
  std::uint32_t _rank;
  std::vector<int> _sockets;

  std::uint16_t _port;
};

struct WorkerCommandChannel {
  WorkerCommandChannel();
  ~WorkerCommandChannel();

  bool init();

  std::unique_ptr<rpc::RPCMessage> recvMessage();
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg);

private:
  std::uint32_t _rank;
  int _socket;

  std::string _master_addr;
  std::uint16_t _master_port;
};

} // namespace thd
