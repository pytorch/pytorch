#pragma once

#include "../DataChannel.hpp"
#include "../ChannelEnvVars.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace thd {

struct DataChannelTCP : DataChannel {
  DataChannelTCP();
  DataChannelTCP(int timeout);
  virtual ~DataChannelTCP();

  bool init() override;

  int getRank() override;
  int getNumProcesses() override;

  void allReduce(Tensor& data, THDReduceOp operation) override;
  void reduce(Tensor& data, THDReduceOp operation, int dst_rank) override;
  void broadcast(Tensor& data, int src_id) override;
  void send(Tensor& data, int dst_id) override;
  void receive(Tensor& data, int src_id) override;

private:
  // Defines process to which master or worker is connected
  struct Process {
    std::uint32_t rank;
    std::string address;
    std::uint16_t port;
    int socket;
  };


  void listen(std::uint16_t port);
  int connect(const std::string& address, std::uint16_t port, int wait) const;
  std::tuple<int, std::string> accept() const;

  bool initMaster();
  bool initWorker();

  void reduce_(Tensor& result, Tensor& data, THDReduceOp operation) const;
  template<typename T>
  void reduce_(Tensor& result, Tensor& data, THDReduceOp operation) const;


  int _rank; // Rank of current process, range: [0.._processes.size()-1]
  int _socket; // Socket on which process is listening
  int _port; // Port on which process is listening
  int _timeout; // Accept waiting timeout in milliseconds (it is optional, default = infinity)

  std::vector<Process> _processes; // Other processes in network
};

} // namespace thd
