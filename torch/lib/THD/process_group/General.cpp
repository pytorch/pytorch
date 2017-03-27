#include "General.hpp"

namespace thd {
std::unique_ptr<DataChannel> dataChannel;
} // namespace thd

using namespace thd;

bool THDProcessGroupInit(THDChannelType channel_type) {
  dataChannel = std::unique_ptr<DataChannel>(
      thd::DataChannel::newChannel(channel_type));
  if (!dataChannel->init()) return false;
  return true;
}
