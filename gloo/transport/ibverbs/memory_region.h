/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/transport/ibverbs/device.h"

namespace gloo {
namespace transport {
namespace ibverbs {

// MemoryRegion is used to send local ibv_mr to remote side of pair.
// Every pair has one instance per slot to receive ibv_mr's.
// For every receive buffer created on this pair, another instance
// is created to the ibv_mr of that buffer can be sent to its peer.
class MemoryRegion {
 public:
  explicit MemoryRegion(struct ibv_pd*);
  explicit MemoryRegion(struct ibv_pd*, struct ibv_mr*);
  MemoryRegion(const MemoryRegion& that) = delete;
  MemoryRegion& operator=(const MemoryRegion& that) = delete;
  ~MemoryRegion();

  // Construct and return scatter/gather element for this memory region.
  struct ibv_sge sge() const {
    struct ibv_sge list;
    list.addr = (uint64_t)&src_;
    list.length = sizeof(src_);
    list.lkey = mr_->lkey;
    return list;
  }

  struct ibv_mr mr() const {
    return src_;
  }

 protected:
  // The ibv_mr that is read from or written to.
  struct ibv_mr src_;

  // The ibv_mr to hold the registration of src_.
  struct ibv_mr* mr_;
};

} // namespace ibverbs
} // namespace transport
} // namespace gloo
