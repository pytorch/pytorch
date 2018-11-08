#pragma once

namespace caffe2 {

// Get value from map given key. Return suppiled default value if not found
// This is a stripped down version from folly:
// https://github.com/facebook/folly/blob/5a07e203d79324b68d69f294fa38e43b9671e9b1/folly/MapUtil.h#L35-L45
template <
    class Map,
    typename Key = typename Map::key_type,
    typename Value = typename Map::mapped_type>
typename Map::mapped_type
get_default(const Map& map, const Key& key, Value&& dflt) {
  using M = typename Map::mapped_type;
  auto pos = map.find(key);
  return (pos != map.end()) ? (pos->second) : M(std::forward<Value>(dflt));
}

} // namespace caffe2
