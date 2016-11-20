#include "DataChannel.hpp"

namespace thd {

DataChannelRegistry::channels_map DataChannelRegistry::s_registered_channels;

DataChannel* DataChannelRegistry::dataChannelFor(channel_key_type key) {
  return s_registered_channels.at(key);
}

} // namespace thd
