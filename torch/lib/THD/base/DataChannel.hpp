#pragma once

#include "ChannelType.h"
#include "DataChannel.h"
#include "Tensor.hpp"

namespace thd {

struct DataChannel {
public:
  DataChannel() {};
  virtual ~DataChannel() {};

  virtual bool init() = 0;

  virtual int getRank() = 0;
  virtual int getNumProcesses() = 0;

  virtual void allReduce(Tensor& data, THDReduceOp operation, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void reduce(Tensor& data, THDReduceOp operation, int dst_rank,
                      THDGroup group_id = THDGroupWORLD) = 0;
  virtual void broadcast(Tensor& data, int src_rank, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void send(Tensor& data, int dst_rank) = 0;
  virtual void receive(Tensor& data, int src_rank) = 0;

  virtual THDGroup newGroup(std::vector<int> ranks) = 0;

  static DataChannel* newChannel(THDChannelType type);
};


} // namespace thd
