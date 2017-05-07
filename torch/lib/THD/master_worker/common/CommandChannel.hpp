#pragma once

#include "RPC.hpp"

#include <sys/poll.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace thd {

struct MasterCommandChannel {
  MasterCommandChannel();
  ~MasterCommandChannel();

  bool init();

  void sendMessage(std::unique_ptr<rpc::RPCMessage> msg, int rank);

private:
  std::tuple<rank_type, std::string> recvError();
  void errorHandler();

  rank_type _rank;
  std::vector<int> _sockets;
  std::unique_ptr<struct pollfd[]> _poll_events;

  int _error_pipe; // informs error handler thread that we are exiting
  std::unique_ptr<std::string> _error;
  std::thread _error_thread;

  port_type _port;

  std::vector<std::mutex> _mutexes;
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
