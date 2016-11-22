#include "DataChannel.hpp"
#include "channels/DataChannelMPI.hpp"
#include "channels/DataChannelTCP.hpp"

#include <stdexcept>

namespace thd {

DataChannel* DataChannel::newChannel(THDChannelType type) {
  if (type == THDChannelTCP)
    return new DataChannelTCP();
  else if (type == THDChannelMPI)
    return new DataChannelMPI();
  throw std::runtime_error("unsupported data channel type");
}

} // namespace thd
