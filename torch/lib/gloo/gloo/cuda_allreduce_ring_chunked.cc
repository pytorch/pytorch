/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_ring_chunked.h"

#include "gloo/cuda_collectives_device.h"
#include "gloo/cuda_collectives_host.h"
#include "gloo/cuda_private.h"

namespace gloo {

template <typename T, typename W>
struct CudaAllreduceRingChunked<T, W>::ChunkContext {
  ChunkContext(
      typename W::Pointer&& scratch,
      std::unique_ptr<LocalOp<T> >&& reduceOp,
      std::unique_ptr<LocalOp<T> >&& broadcastOp)
      : scratch(std::move(scratch)),
        length(this->scratch.getCount()),
        reduceOp(std::move(reduceOp)),
        broadcastOp(std::move(broadcastOp)) {}
  ChunkContext(ChunkContext&& other) = default;

  // Instances cannot be copied or copy-assigned
  ChunkContext(const ChunkContext&) = delete;
  ChunkContext& operator=(const ChunkContext&) = delete;

  // Pointer to chunk in scratch buffer
  typename W::Pointer scratch;
  const size_t length;

  // The operations used for local device reduce before running the
  // algorithm and local device broadcast after.
  std::unique_ptr<LocalOp<T> > reduceOp;
  std::unique_ptr<LocalOp<T> > broadcastOp;
};

template <typename T, typename W>
CudaAllreduceRingChunked<T, W>::CudaAllreduceRingChunked(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    const int count,
    const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
      count_(count),
      bytes_(count * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      fn_(CudaReductionFunction<T>::sum),
      leftPair_(this->getLeftPair()),
      rightPair_(this->getRightPair()) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    if (newStream) {
      streams_.push_back(CudaStream(ptr.getDeviceID()));
    } else {
      streams_.push_back(CudaStream(ptr.getDeviceID(), streams[i]));
    }
    devicePtrs_.push_back(std::move(ptr));
  }

  // Determine chunk size. Use chunks of no less than 1024 bytes
  // (256 * sizeof(float)).
  constexpr unsigned long minSize = 256;
  chunks_ = this->contextSize_ * 2;
  chunkSize_ = std::max(minSize, (count_ + chunks_ - 1) / chunks_);
  chunkBytes_ = chunkSize_ * sizeof(T);

  // Workspace specific initialization (see below)
  init();

  for (auto offset = 0; offset < count_; offset += chunkSize_) {
    auto length = chunkSize_;
    if (offset + length <= count_) {
      // Chunk completely in range, full chunk.
    } else {
      // Chunk partially in range, partial chunk.
      length = count_ - offset;
    }

    chunkContext_.push_back(
        ChunkContext(
            scratch_.range(offset, length),
            cudaDeviceReduce(
              streams_, devicePtrs_, scratch_, fn_, offset, length),
            cudaDeviceBroadcast(
              streams_, devicePtrs_, scratch_, offset, length)));
  }

  for (auto i = 0; i < 2; i++) {
    auto slot = this->context_->nextSlot();

    // Buffer to send to (rank+1).
    sendDataBuf_[i] =
      rightPair_->createSendBuffer(slot, *scratch_, bytes_);
    // Buffer that (rank-1) writes to.
    recvDataBuf_[i] =
      leftPair_->createRecvBuffer(slot, *inbox_[i], chunkBytes_);
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

template <typename T, typename W>
CudaAllreduceRingChunked<T, W>::~CudaAllreduceRingChunked() {
}

template <typename T, typename W>
void CudaAllreduceRingChunked<T, W>::run() {
  CudaDeviceGuard guard;
  CudaStream& stream = streams_[0];

  // Kick off local reduction for each chunk.
  // The result is stored in scratch_ at the corresponding chunk offset.
  // Make sure to iterate over the chunks in the order they will be sent.
  for (auto i = 0; i < chunks_; i++) {
    const auto chunkOffset = getChunkOffset(i);
    if (chunkOffset < chunkContext_.size()) {
      auto& context = chunkContext_[chunkOffset];
      context.reduceOp->runAsync();
    }
  }

  // First pass reduces a chunk in each round
  for (auto round = 0; round < chunks_; round++) {
    const auto chunkOffset = getChunkOffset(round);

    if (chunkOffset < chunkContext_.size()) {
      auto& context = chunkContext_[chunkOffset];

      // Wait for the local reduction to complete
      // When using the host workspace this also makes sure the reduction
      // result is copied into the host side scratch buffer.
      context.reduceOp->wait();

      // Reduce chunk from previous round. Nothing to do for initial rounds.
      if (round >= 2) {
        // Wait for inbox write to complete
        recvDataBuf_[chunkOffset & 1]->waitRecv();

        // Reduce
        fn_->call(
            context.scratch,
            inbox_[chunkOffset & 1],
            context.scratch.getCount(),
            stream);
        stream.wait();
      }
    } else {
      // Empty chunk but still need to wait on the inbox write to ensure the
      // algorithm progresses. Nothing to do for initial rounds.
      if (round >= 2) {
        recvDataBuf_[chunkOffset & 1]->waitRecv();
      }
    }

    // Skip buffer passing notifications in initial rounds
    if (round >= 2) {
      // Send notification to node on the left that
      // this node is ready for an inbox write.
      sendNotificationBuf_->send();

      // Wait for notification from node on the right
      // to be sure this node can start an inbox write.
      recvNotificationBuf_->waitRecv();
    }

    // Copy accumulated chunk
    copyChunkAtOffset(chunkOffset);
  }

  // Second pass around the ring to broadcast result.
  for (int round = 0; round < chunks_; round++) {
    const auto chunkOffset = getChunkOffset(round);

    if (chunkOffset < chunkContext_.size()) {
      auto& context = chunkContext_[chunkOffset];

      // End at chunks_-2 since that's where the accumulation
      // stopped in the previous set of rounds.
      if (round < (chunks_ - 2)) {
        // Wait for inbox write to complete
        recvDataBuf_[chunkOffset & 1]->waitRecv();

        // Copy chunk from inbox to scratch space
        stream.copyAsync(context.scratch, inbox_[chunkOffset & 1]);
        stream.wait();
      }

      // Broadcast chunk to devices. Do this in all rounds with non-empty chunk.
      context.broadcastOp->runAsync();
    } else {
      // Empty chunk but still need to wait on the inbox write to ensure the
      // algorithm progresses.
      if (round < (chunks_ - 2)) {
        recvDataBuf_[chunkOffset & 1]->waitRecv();
      }
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

  // If running synchronously, wait for all chunk broadcasts to complete
  if (synchronizeDeviceOutputs_) {
    for (auto i = 0; i < chunks_; i++) {
      const auto chunkOffset = getChunkOffset(i);
      if (chunkOffset < chunkContext_.size()) {
        chunkContext_[chunkOffset].broadcastOp->wait();
      }
    }
  }
}

template <typename T, typename W>
int CudaAllreduceRingChunked<T, W>::getChunkOffset(int round) {
  // Imagine a square grid with chunks of memory laid out vertically and nodes
  // horizontally. The diagonal of this grid marks which nodes sends which
  // chunks of memory in the prelude. Processing happens by moving this
  // diagonal forward and have it wrap around the edge. This means that node
  // with rank 0 at round 2 will process the last chunk. This explains why
  // we subtract the round in the offset equation below.
  //
  // Because we're dealing with double buffering in this implementation, we
  // have twice the number of chunks and process them in pairs. This explains
  // why we ignore the LSB on the round number when subtracting it. The LSB is
  // later added to flip back and forth between the two buffers for this pair
  // of chunks. The number of chunks is finally added to make sure we can wrap
  // correctly (no modulo against negative number).
  return ((2 * this->contextRank_) - (round & ~0x1) + (round & 0x1) + chunks_) %
      chunks_;
}

template <typename T, typename W>
void CudaAllreduceRingChunked<T, W>::copyChunkAtOffset(int chunkOffset) {
  // Populate inbox of next participant in the ring.
  size_t offset;
  size_t length;
  if (chunkOffset < chunkContext_.size()) {
    const auto& context = chunkContext_[chunkOffset];
    offset = chunkOffset * chunkSize_;
    length = context.length;
  } else {
    // When nothing is put on the wire for empty chunks. @pietern
    // has seen this algorithm hang. This is probably related to the
    // chunk iteration order described in the run function.
    // Chunk out of range, copy _something_.
    offset = 0;
    length = 1;
  }

  // Initiate write to inbox of node on the right.
  sendDataBuf_[chunkOffset & 0x1]->send(offset * sizeof(T), length * sizeof(T));
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceRingChunked<T, W>::init(
    typename std::enable_if<std::is_same<U, CudaHostWorkspace<T> >::value,
    typename U::Pointer>::type*) {
  // Since reduction is executed on the CPU, the scratch space
  // where the reduction is accumulated is a new host side buffer.
  scratch_ = W::Pointer::alloc(count_);

  // Allocate inboxes
  for (auto i = 0; i < 2; i++) {
    inbox_[i] = W::Pointer::alloc(chunkSize_);
  }
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceRingChunked<T, W>::init(
    typename std::enable_if<std::is_same<U, CudaDeviceWorkspace<T> >::value,
    typename U::Pointer>::type*) {
  // Since reduction is executed on the GPU, the scratch space
  // can use an existing input buffer to accumulate.
  auto& ptr = devicePtrs_[0];
  auto count = ptr.getCount();
  scratch_ = CudaDevicePointer<T>::create(*ptr, count);

  // Allocate inboxes
  for (auto i = 0; i < 2; i++) {
    inbox_[i] = W::Pointer::alloc(chunkSize_);
  }
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                                         \
template class CudaAllreduceRingChunked<T, CudaHostWorkspace<T> >;      \
template class CudaAllreduceRingChunked<T, CudaDeviceWorkspace<T> >;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
