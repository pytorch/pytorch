#pragma once

#include "../DataChannel.hpp"

namespace thd {

struct DataChannelMPI : DataChannel {
public:
  DataChannelMPI();
  virtual ~DataChannelMPI();

  bool init() override;

  int getRank() const override;
  int getNumProcesses() const override;

  void allReduce(Tensor& data) override;
  void reduce(Tensor& data, int dst_rank) override;
  void broadcast(Tensor& data, int src_rank) override;
  void send(Tensor& data, int dst_rank) override;
  void receive(Tensor& data, int src_rank) override;

private:
  void broadcastPack(Tensor& data, int src_rank);
  void broadcastUnpack(Tensor& data, int src_rank);


  int m_rank; // Current process' rank
  int m_num_processes; // Number of processes in network
};

} // namespace thd
