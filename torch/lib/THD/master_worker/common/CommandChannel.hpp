#pragma once

#include "RPC.hpp"

#include <memory>
#include <string>
#include <vector>

#include <asio.hpp>

namespace thd {

struct MasterCommandChannel {
  MasterCommandChannel();
  ~MasterCommandChannel();

  std::unique_ptr<rpc::RPCMessage> recvMessage(int rank);
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank);

private:
  int _rank;
  asio::io_service _io;
  int _world_size;
  std::vector<asio::ip::tcp::socket> _sockets;

  void _load_env(unsigned short& port, int& world_size);
};

struct WorkerCommandChannel {
  WorkerCommandChannel(int rank);
  ~WorkerCommandChannel();

  std::unique_ptr<rpc::RPCMessage> recvMessage();
  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg);

private:
  int _rank;
  asio::io_service _io;
  asio::ip::tcp::socket _socket;

  void _load_env(std::string& ip_addr, unsigned short& port);
};

} // namespace thd
