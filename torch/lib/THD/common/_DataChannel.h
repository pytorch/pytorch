#ifndef _THD_BACKEND_H
#define _THD_BACKEND_H

#include <unordered_map>

#include "_Tensor.h"

namespace thd {


struct DataChannel {
  virtual bool init();

  virtual int get_id();
  virtual int get_num_processes();

  virtual void all_reduce(Tensor &data);

  virtual void reduce(Tensor &data, int dst_id);

  virtual void broadcast(Tensor &data, int src_id);

  virtual void send(Tensor &data, int dst_id);

  virtual void recieve(Tensor &data, int src_id);
};


struct DataChannelRegistry {
  using channel_key_type = int;

  static DataChannel& backend_for(channel_key_type type);

private:
  static std::unordered_map<channel_key_type, DataChannel> s_registered_channels;
};


} // namespace thd

#endif
