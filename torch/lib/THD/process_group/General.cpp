#include "General.hpp"
#include "../base/Exceptions.hpp"

namespace thd {
std::unique_ptr<DataChannel> dataChannel;
} // namespace thd

using namespace thd;

void THDProcessGroupInit(THDChannelType channel_type, std::string init_method = "env://",
                         int world_size = -1, std::string group_name = "", int rank = -1) {
  HANDLE_EXCEPTIONS
  dataChannel = std::unique_ptr<DataChannel>(
      thd::DataChannel::newChannel(channel_type, init_method, world_size,
                                   group_name, rank));
  dataChannel->init();
  END_HANDLE_EXCEPTIONS
}
