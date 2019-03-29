#include <THD/process_group/General.hpp>
#include <THD/base/Exceptions.hpp>

namespace thd {
std::unique_ptr<DataChannel> dataChannel;
} // namespace thd

using namespace thd;

void THDProcessGroupInit(
    THDChannelType channel_type,
    std::string init_method = "env://",
    int world_size = -1,
    std::string group_name = "",
    int rank = -1) {
  HANDLE_EXCEPTIONS
  dataChannel = std::unique_ptr<DataChannel>(thd::DataChannel::newChannel(
      channel_type, init_method, world_size, group_name, rank));
  dataChannel->init();
  END_HANDLE_EXCEPTIONS
}

void THDProcessGroupDestroy() {
  HANDLE_EXCEPTIONS
  if (dataChannel) {
    dataChannel->destroy();
    dataChannel.reset(nullptr);
  }
  END_HANDLE_EXCEPTIONS
}

void THDClearGroupCache(THDGroup group) {
  HANDLE_EXCEPTIONS
  if (dataChannel) {
    dataChannel->clearGroupCache(group);
  }
  END_HANDLE_EXCEPTIONS
}
