/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_ring_chunked.h"

#include <string.h>

#include "gloo/cuda_nccl.h"
#include "gloo/cuda_private.h"

namespace gloo {

template <typename T>
struct CudaAllreduceRingChunked<T>::ChunkContext {
  ChunkContext(
      CudaDevicePointer<T>&& rootDevicePtr,
      T* hostPtr,
      size_t length,
      std::vector<nccl::NCCLElement<T>>&& reduceElements,
      std::vector<nccl::NCCLElement<T>>&& broadcastElements)
      : rootDevicePtr(std::move(rootDevicePtr)),
        hostPtr(hostPtr),
        length(length),
        reduceOp(nccl::NCCLContext<T>(
            this->rootDevicePtr.getDeviceID(),
            this->rootDevicePtr.getStream(),
            std::move(reduceElements),
            this->rootDevicePtr.getDeviceID())),
        broadcastOp(nccl::NCCLContext<T>(
            this->rootDevicePtr.getDeviceID(),
            this->rootDevicePtr.getStream(),
            std::move(broadcastElements),
            this->rootDevicePtr.getDeviceID())) {
  }
  ChunkContext(ChunkContext&& other) = default;

  // Instances cannot be copied or copy-assigned
  ChunkContext(const ChunkContext&) = delete;
  ChunkContext& operator=(const ChunkContext&) = delete;

  // Pointers for copying between the device and host
  CudaDevicePointer<T> rootDevicePtr;
  T* hostPtr;
  const size_t length;
  // The NCCL operations used for local device reduce before running the
  // algorithm and local device broadcast after.
  nccl::ReduceOp<T> reduceOp;
  nccl::BroadcastOp<T> broadcastOp;
};

template <typename T>
CudaAllreduceRingChunked<T>::CudaAllreduceRingChunked(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    int count,
    const std::vector<cudaStream_t>& streams)
    : Allreduce<T>(context, nullptr),
      count_(count),
      bytes_(count * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      leftPair_(this->getLeftPair()),
      rightPair_(this->getRightPair()) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (int i = 0; i < ptrs.size(); i++) {
    if (newStream) {
      devicePtrs_.push_back(CudaDevicePointer<T>::create(ptrs[i], count_));
    } else {
      devicePtrs_.push_back(
          CudaDevicePointer<T>::create(ptrs[i], count_, streams[i]));
    }
  }

  // Determine chunk size. Use chunks of no less than 1024 bytes
  // (256 * sizeof(float)).
  constexpr unsigned long minSize = 256;
  chunks_ = this->contextSize_ * 2;
  chunkSize_ = std::max(minSize, (count_ + chunks_ - 1) / chunks_);
  chunkBytes_ = chunkSize_ * sizeof(T);

  // Setup host and device memory
  {
    // Synchronize memory allocation with NCCL operations
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaMallocHost(&hostPtr_, bytes_));
  }
  for (auto offset = 0; offset < count_; offset += chunkSize_) {
    auto length = chunkSize_;
    if (offset + length <= count_) {
      // Chunk completely in range, full chunk.
    } else {
      // Chunk partially in range, partial chunk.
      length = count_ - offset;
    }

    // Create NCCL elements for the chunk on each device
    std::vector<nccl::NCCLElement<T>> reduceElements;
    std::vector<nccl::NCCLElement<T>> broadcastElements;
    for (auto i = 0; i < ptrs.size(); i++) {
      const auto chunkPtr = *devicePtrs_[i] + offset;
      const auto stream = devicePtrs_[i].getStream();
      reduceElements.push_back(nccl::NCCLElement<T>(
          CudaDevicePointer<T>::create(chunkPtr, length, stream),
          CudaDevicePointer<T>::create(chunkPtr, length, stream)));
      broadcastElements.push_back(nccl::NCCLElement<T>(
          CudaDevicePointer<T>::create(chunkPtr, length, stream),
          CudaDevicePointer<T>::create(chunkPtr, length, stream)));
    }

    // Create a device pointer for the chunk on device ptrs[0]. We will use the
    // associated stream to serialize NCCL operations and device-host memcpys.
    // The NCCL operation will synchronize the independent device streams with
    // this master stream.
    CudaDevicePointer<T> rootDevicePtr =
        CudaDevicePointer<T>::create(ptrs[0] + offset, length);
    chunkContext_.push_back(ChunkContext(
        std::move(rootDevicePtr),
        hostPtr_ + offset,
        length,
        std::move(reduceElements),
        std::move(broadcastElements)));
  }

  // Allocate inboxes
  for (auto i = 0; i < 2; i++) {
    inbox_[i] = static_cast<T*>(malloc(bytes_));
  }

  for (auto i = 0; i < 2; i++) {
    auto slot = this->context_->nextSlot();

    // Buffer to send to (rank+1).
    sendDataBuf_[i] = rightPair_->createSendBuffer(slot, hostPtr_, bytes_);
    // Buffer that (rank-1) writes to.
    recvDataBuf_[i] = leftPair_->createRecvBuffer(slot, inbox_[i], chunkBytes_);
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

template <typename T>
CudaAllreduceRingChunked<T>::~CudaAllreduceRingChunked() {
  {
    // Synchronize memory allocation with NCCL operations
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaFreeHost(hostPtr_));
  }
  for (auto i = 0; i < 2; i++) {
    if (inbox_[i] != nullptr) {
      free(inbox_[i]);
    }
  }
}

template <typename T>
void CudaAllreduceRingChunked<T>::run() {
  CudaDeviceGuard guard;

  // Kick off local reduction for each chunk, then copy result to host.
  // Make sure to iterate over the chunks in the order they will be sent.
  for (auto i = 0; i < chunks_; i++) {
    const auto chunkOffset = getChunkOffset(i);
    if (chunkOffset < chunkContext_.size()) {
      auto& context = chunkContext_[chunkOffset];
      context.reduceOp.runAsync();
      context.rootDevicePtr.copyToHostAsync(context.hostPtr);
    }
  }

  // First pass reduces a chunk in each round
  for (auto round = 0; round < chunks_; round++) {
    const auto chunkOffset = getChunkOffset(round);

    if (chunkOffset < chunkContext_.size()) {
      auto& context = chunkContext_[chunkOffset];

      // Wait for the local reduction and copy to host memory to complete
      context.rootDevicePtr.wait();

      // Reduce chunk from previous round. Nothing to do for initial rounds.
      if (round >= 2) {
        // Wait for inbox write to complete
        recvDataBuf_[chunkOffset & 1]->waitRecv();

        // Reduce
        this->fn_(context.hostPtr, inbox_[chunkOffset & 1], context.length);
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

        // Copy from inbox
        memcpy(
            context.hostPtr,
            inbox_[chunkOffset & 1],
            context.length * sizeof(T));
      }

      // Broadcast chunk to devices. Do this in all rounds with non-empty chunk.
      context.rootDevicePtr.copyFromHostAsync(context.hostPtr);
      context.broadcastOp.runAsync();
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
        chunkContext_[chunkOffset].broadcastOp.wait();
      }
    }
  }
}

template <typename T>
int CudaAllreduceRingChunked<T>::getChunkOffset(int round) {
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

template <typename T>
void CudaAllreduceRingChunked<T>::copyChunkAtOffset(int chunkOffset) {
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

// Instantiate template
template class CudaAllreduceRingChunked<float>;

} // namespace gloo
