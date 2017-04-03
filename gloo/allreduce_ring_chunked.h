/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace gloo {

template <typename T>
class AllreduceRingChunked : public Algorithm {
 public:
  AllreduceRingChunked(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum)
      : Algorithm(context),
        ptrs_(ptrs),
        count_(count),
        bytes_(count_ * sizeof(T)),
        fn_(fn),
        leftPair_(this->getLeftPair()),
        rightPair_(this->getRightPair()) {
    // Use chunks of no less than 1024 bytes (256 * sizeof(float))
    constexpr unsigned long minSize = 256;
    chunks_ = this->contextSize_ * 2;
    chunkSize_ = std::max(minSize, (count_ + chunks_ - 1) / chunks_);
    chunkBytes_ = chunkSize_ * sizeof(T);

    // Allocate inboxes
    for (int i = 0; i < 2; i++) {
      inbox_[i] = static_cast<T*>(malloc(bytes_));
    }

    for (int i = 0; i < 2; i++) {
      auto slot = this->context_->nextSlot();

      // Buffer to send to (rank+1).
      sendDataBuf_[i] =
        rightPair_->createSendBuffer(slot, ptrs_[0], bytes_);
      // Buffer that (rank-1) writes to.
      recvDataBuf_[i] =
        leftPair_->createRecvBuffer(slot, inbox_[i], chunkBytes_);
    }

    // Dummy buffers for localized barrier.
    // Before sending to the right, we only need to know that the node
    // on the right is done using the inbox that's about to be written
    // into. No need for a global barrier.
    auto notificationSlot = this->context_->nextSlot();
    sendNotificationBuf_ =
      leftPair_->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
    recvNotificationBuf_ =
      rightPair_->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));
  }

  virtual ~AllreduceRingChunked() {
    for (int i = 0; i < 2; i++) {
      if (inbox_[i] != nullptr) {
        free(inbox_[i]);
      }
    }
  }

  void run() {
    // Reduce specified pointers into ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      fn_->call(ptrs_[0], ptrs_[i], count_);
    }

    // Kick off copying initial chunks
    copyChunkAtOffset(2 * this->contextRank_);
    copyChunkAtOffset(2 * this->contextRank_ + 1);

    // Start with reduction of previously copied chunk
    for (int round = 2; round < chunks_; round++) {
      // We loop over all chunks starting at 2, since we just sent two
      // chunks to fill both buffers. Imagine a square grid with
      // chunks of memory laid out vertically and nodes horizontally.
      // The diagonal of this grid marks which nodes sends which
      // chunks of memory in the prelude. Processing happens by moving
      // this diagonal forward and have it wrap around the edge. This
      // means that node with rank 0 at round 2 will process the last
      // chunk. This explains why we subtract the round in the offset
      // equation below.
      //
      // Because we're dealing with double buffering in this
      // implementation, we have twice the number of chunks and
      // process them in pairs. This explains why we ignore the LSB on
      // the round number when subtracting it. The LSB is later added
      // to flip back and forth between the two buffers for this pair
      // of chunks. The number of chunks is finally added to make sure
      // we can wrap correctly (no modulo against negative number).
      //
      auto chunkOffset = ((2 * this->contextRank_) - (round & ~0x1) +
                          (round & 0x1) + chunks_) %
          chunks_;
      auto offset = chunkOffset * chunkSize_;
      auto length = chunkSize_;
      if (offset + length <= count_) {
        // Chunk completely in range, copy full chunk.
      } else if (offset < count_) {
        // Chunk partially in range, copy partial chunk.
        length = count_ - offset;
      } else {
        // Chunk out of range, copy nothing.
        length = 0;
      }

      // Wait for inbox write to complete
      recvDataBuf_[chunkOffset & 1]->waitRecv();

      // Reduce
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], inbox_[chunkOffset & 1], length);
      }

      // Send notification to node on the left that
      // this node is ready for an inbox write.
      sendNotificationBuf_->send();

      // Wait for notification from node on the right
      // to be sure this node can start an inbox write.
      recvNotificationBuf_->waitRecv();

      // Copy accumulated chunk
      copyChunkAtOffset(chunkOffset);
    }

    // Second pass around the ring to broadcast result.
    // End at chunks_-2 since that's where the accumulation
    // stopped in the previous set of rounds.
    for (int round = 0; round < (chunks_ - 2); round++) {
      auto chunkOffset = ((2 * this->contextRank_) - (round & ~0x1) +
                          (round & 0x1) + chunks_) %
          chunks_;
      auto offset = chunkOffset * chunkSize_;
      auto length = chunkSize_;
      if (offset + length <= count_) {
        // Chunk completely in range, copy full chunk.
      } else if (offset < count_) {
        // Chunk partially in range, copy partial chunk.
        length = count_ - offset;
      } else {
        // Chunk out of range, copy nothing.
        length = 0;
      }

      // Wait for inbox write to complete
      recvDataBuf_[chunkOffset & 1]->waitRecv();

      // Copy
      if (length > 0) {
        memcpy(&ptrs_[0][offset], inbox_[chunkOffset & 1], length * sizeof(T));
      }

      // Skip copying in the last two rounds
      if (round < (chunks_ - 4)) {
        // Send notification to node on the left that
        // this node is ready for an inbox write.
        sendNotificationBuf_->send();

        // Wait for notification from node on the right
        // to be sure this node can start an inbox write.
        recvNotificationBuf_->waitRecv();

        // Copy accumulated chunks
        copyChunkAtOffset(chunkOffset);
      }
    }

    // Final barrier to make sure every node has finished
    // Otherwise, a second all reduce call might interfere
    // with one that it still in progress on some nodes.
    sendNotificationBuf_->send();
    recvNotificationBuf_->waitRecv();

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }

 protected:
  void copyChunkAtOffset(int chunkOffset) {
    // Populate inbox of next participant in the ring.
    auto offset = (chunkOffset % chunks_) * chunkSize_;
    auto length = chunkSize_;
    if (offset + length <= count_) {
      // Chunk completely in range, copy full chunk.
    } else if (offset < count_) {
      // Chunk partially in range, copy partial chunk.
      length = count_ - offset;
    } else {
      // Chunk out of range, copy _something_.
      // When nothing is put on the wire for empty chunks. @pietern
      // has seen this algorithm hang. This is probably related to the
      // chunk iteration order described in the run function.
      offset = 0;
      length = 1;
    }

    // Initiate write to inbox of node on the right.
    sendDataBuf_[chunkOffset & 0x1]->send(
        offset * sizeof(T), length * sizeof(T));
  }

  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const ReductionFunction<T>* fn_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  size_t chunks_;
  size_t chunkSize_;
  size_t chunkBytes_;

  T* inbox_[2];
  std::unique_ptr<transport::Buffer> sendDataBuf_[2];
  std::unique_ptr<transport::Buffer> recvDataBuf_[2];

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

} // namespace gloo
