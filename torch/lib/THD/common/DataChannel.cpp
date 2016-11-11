#include "../_THD.h"

namespace thd {

std::unordered_map<DataChannelRegistry::channel_key_type, DataChannel>
    DataChannelRegistry::s_registered_channels;


DataChannel& DataChannelRegistry::backend_for(channel_key_type key) {
  return s_registered_channels.at(key);
}

} // namespace thd
