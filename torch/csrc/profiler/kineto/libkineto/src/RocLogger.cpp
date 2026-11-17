/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RocLogger.h"

ApiIdList::ApiIdList() : invert_(true) {}

void ApiIdList::add(const std::string& apiName) {
  uint32_t cid = mapName(apiName);
  if (cid > 0)
    filter_[cid] = 1;
}

void ApiIdList::remove(const std::string& apiName) {
  uint32_t cid = mapName(apiName);
  if (cid > 0)
    filter_.erase(cid);
}

bool ApiIdList::loadUserPrefs() {
  // FIXME: check an ENV variable that points to an exclude file
  return false;
}

bool ApiIdList::contains(uint32_t apiId) {
  return (filter_.find(apiId) != filter_.end()) ? !invert_ : invert_; // XOR
}
