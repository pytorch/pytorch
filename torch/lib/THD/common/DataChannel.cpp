#include "../_THD.h"

namespace thd {

std::unordered_map<DataChannelRegistry::channel_key_type, DataChannel>
    DataChannelRegistry::registered_channels;


DataChannel& DataChannelRegistry::backend_for(channel_key_type key) {
  return registered_channels.at(key);
}

} // namespace thd
