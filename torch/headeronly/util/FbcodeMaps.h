#pragma once
#ifndef C10_UTIL_FBCODEMAPS_H_
#define C10_UTIL_FBCODEMAPS_H_

// Map typedefs so that we can use folly's F14 maps in fbcode without
// taking a folly dependency.

#include <torch/headeronly/macros/Macros.h>

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#else
#include <unordered_map>
#include <unordered_set>
#endif

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
#ifdef FBCODE_CAFFE2
template <typename Key, typename Value>
using FastMap = folly::F14FastMap<Key, Value>;
template <typename Key>
using FastSet = folly::F14FastSet<Key>;
#else
template <typename Key, typename Value>
using FastMap = std::unordered_map<Key, Value>;
template <typename Key>
using FastSet = std::unordered_set<Key>;
#endif
HIDDEN_NAMESPACE_END(torch, headeronly)

namespace c10 {
using torch::headeronly::FastMap;
using torch::headeronly::FastSet;
} // namespace c10

#endif // C10_UTIL_FBCODEMAPS_H_
