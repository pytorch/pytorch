#include "General.hpp"

namespace thd {
std::unique_ptr<DataChannel> dataChannel;
} // namespace thd

using namespace thd;

bool THDProcessGroupInit(THDChannelType channel_type, std::string init_method,
                         int world_size, std::string group_name) {
  dataChannel = std::unique_ptr<DataChannel>(
      thd::DataChannel::newChannel(channel_type, init_method, world_size,
                                   group_name));
  if (!dataChannel->init()) return false;
  return true;
}
