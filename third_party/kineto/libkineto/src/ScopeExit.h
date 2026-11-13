/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Implement a simple scope handler allowing a function to release
// resources when an error or exception occurs

template <typename T>
class ScopeExit {
 public:
  explicit ScopeExit(T t) : t(t) {}
  ~ScopeExit() {
    t();
  }
  T t;
};

template <typename T>
ScopeExit<T> makeScopeExit(T t) {
  return ScopeExit<T>(t);
}

// Add a level of indirection so __LINE__ is expanded
#define __kINETO_CONCAT(name, line) name##line
#define ANON_VAR(name, line) __kINETO_CONCAT(name, line)

#define SCOPE_EXIT(func) \
  const auto ANON_VAR(SCOPE_BLOCK, __LINE__) = makeScopeExit([=]() { func(); })
