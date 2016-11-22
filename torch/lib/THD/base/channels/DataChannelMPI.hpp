#pragma once

#include "../DataChannel.hpp"

namespace thd {

struct DataChannelMPI : DataChannel {
public:
  DataChannelMPI();
  virtual ~DataChannelMPI();

  bool init() override;

  int getRank() override;
  int getNumProcesses() override;

  void allReduce(Tensor& data, THDReduceOp operation) override;
  void reduce(Tensor& data, THDReduceOp operation, int dst_rank) override;
  void broadcast(Tensor& data, int src_rank) override;
  void send(Tensor& data, int dst_rank) override;
  void receive(Tensor& data, int src_rank) override;

private:
  void broadcastPack(Tensor& data, int src_rank) const;
  void broadcastUnpack(Tensor& data, int src_rank) const;


  int m_rank; // Current process' rank
  int m_num_processes; // Number of processes in network
};

} // namespace thd
