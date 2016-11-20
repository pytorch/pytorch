#pragma once

#include "Tensor.hpp"
#include "ChannelType.h"
#include <unordered_map>

namespace thd {

struct DataChannel {
public:
  DataChannel() {};
  virtual ~DataChannel() {};

  virtual bool init() = 0;

  virtual int getRank() const = 0;
  virtual int getNumProcesses() const = 0;

  virtual void allReduce(Tensor& data) = 0;
  virtual void reduce(Tensor& data, int dst_rank) = 0;
  virtual void broadcast(Tensor& data, int src_rank) = 0;
  virtual void send(Tensor& data, int dst_rank) = 0;
  virtual void receive(Tensor& data, int src_rank) = 0;

  static DataChannel* newChannel(THDChannelType type);
};


} // namespace thd
