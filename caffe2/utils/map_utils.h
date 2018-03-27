/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
