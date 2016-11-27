#pragma once

#include "../DataChannel.hpp"

#include <mpi.h>
#include <utility>
#include <unordered_map>
#include <vector>

namespace thd {

struct DataChannelMPI : DataChannel {
public:
  DataChannelMPI();
  virtual ~DataChannelMPI();

  bool init() override;

  int getRank() override;
  int getNumProcesses() override;

  void allReduce(Tensor& data, THDReduceOp operation, THDGroup group_id = THDGroupWORLD) override;
  void reduce(Tensor& data, THDReduceOp operation, int dst_rank, THDGroup group_id = THDGroupWORLD) override;
  void broadcast(Tensor& data, int src_rank, THDGroup group_id = THDGroupWORLD) override;
  void send(Tensor& data, int dst_rank) override;
  void receive(Tensor& data, int src_rank) override;

  THDGroup newGroup(std::vector<int> ranks) override;

private:
  void broadcastPack(Tensor& data, int src_rank, MPI_Comm comm) const;
  void broadcastUnpack(Tensor& data, int src_rank, MPI_Comm comm) const;


  int _rank; // Current process' rank
  int _num_processes; // Number of processes in network
  std::unordered_map<THDGroup, std::pair<MPI_Comm, std::vector<int>>> _groups;
};

} // namespace thd
