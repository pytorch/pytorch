#pragma once

#include "ChannelType.h"
#include "DataChannel.h"
#include "Tensor.hpp"
#include <unordered_map>
#include <cstdint>

namespace thd {

struct DataChannel {
public:
  DataChannel() {};
  virtual ~DataChannel() {};

  virtual bool init() = 0;

  virtual int getRank() = 0;
  virtual int getNumProcesses() = 0;

  virtual void allReduce(Tensor& data, THDReduceOp operation) = 0;
  virtual void reduce(Tensor& data, THDReduceOp operation, int dst_rank) = 0;
  virtual void broadcast(Tensor& data, int src_rank) = 0;
  virtual void send(Tensor& data, int dst_rank) = 0;
  virtual void receive(Tensor& data, int src_rank) = 0;

  static DataChannel* newChannel(THDChannelType type);
};


} // namespace thd
