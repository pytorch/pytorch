#pragma once

#include "Tensor.hpp"
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
};


struct DataChannelRegistry {
  using channel_key_type = int;
  using channels_map = std::unordered_map<channel_key_type, DataChannel*>;

  static DataChannel* dataChannelFor(channel_key_type type);

private:
  static channels_map s_registered_channels;
};

} // namespace thd
