/**
* Copyright (c) 2017-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include "gloo/cuda_allreduce_halving_doubling.h"

#include "gloo/cuda_collectives_device.h"
#include "gloo/cuda_collectives_host.h"
#include "gloo/cuda_private.h"

namespace gloo {

namespace {
// returns the last n bits of ctr reversed
uint32_t reverseLastNBits(uint32_t ctr, uint32_t n) {
  uint32_t bitMask = 1;
  uint32_t reversed = 0;
  while (bitMask < (static_cast<uint32_t>(1) << n)) {
    reversed <<= 1;
    if (ctr & bitMask) {
      reversed |= 1;
    }
    bitMask <<= 1;
  }
  return reversed;
}
}

template <typename T, typename W>
void CudaAllreduceHalvingDoubling<T, W>::initBinaryBlocks() {
  uint32_t offset = this->contextSize_;
  uint32_t blockSize = 1;
  uint32_t currentBlockSize = 0;
  uint32_t prevBlockSize = 0;
  do {
    if (this->contextSize_ & blockSize) {
      prevBlockSize = currentBlockSize;
      currentBlockSize = blockSize;
      offset -= blockSize;
      if (myBinaryBlockSize_ != 0) {
        nextLargerBlockSize_ = currentBlockSize;
        break;
      }
      if (offset <= this->context_->rank) {
        offsetToMyBinaryBlock_ = offset;
        myBinaryBlockSize_ = currentBlockSize;
        nextSmallerBlockSize_ = prevBlockSize;
      }
    }
    blockSize <<= 1;
  } while (offset != 0);

  stepsWithinBlock_ = log2(myBinaryBlockSize_);
  rankInBinaryBlock_ = this->context_->rank % myBinaryBlockSize_;
}

template <typename T, typename W>
CudaAllreduceHalvingDoubling<T, W>::CudaAllreduceHalvingDoubling(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    const int count,
    const std::vector<cudaStream_t>& streams,
    bool pipelineBroadcastAndReduce)
    : Algorithm(context),
      count_(count),
      bytes_(count_ * sizeof(T)),
      steps_(log2(this->contextSize_)),
      chunks_(1 << steps_),
      chunkSize_((count_ + chunks_ - 1) / chunks_),
      chunkBytes_(chunkSize_ * sizeof(T)),
      fn_(CudaReductionFunction<T>::sum),
      sendOffsets_(steps_),
      recvOffsets_(steps_),
      sendCounts_(steps_, 0),
      recvCounts_(steps_, 0),
      sendCountToLargerBlock_(0),
      devicePtrsForBroadcast_(steps_),
      pipelined_(pipelineBroadcastAndReduce),
      offsetToMyBinaryBlock_(0),
      myBinaryBlockSize_(0),
      stepsWithinBlock_(0),
      rankInBinaryBlock_(0),
      nextSmallerBlockSize_(0),
      nextLargerBlockSize_(0) {
  initBinaryBlocks();
  sendDataBufs_.reserve(stepsWithinBlock_);
  recvDataBufs_.reserve(stepsWithinBlock_);
  // Reserve max needed number of context slots. Up to 2 slots per process
  // pair are needed (one for regular sends and one for notifications). For
  // simplicity, the same mapping is used on all processes so that the slots
  // trivially match across processes
  slotOffset_ = this->context_->nextSlot(
      2 * this->contextSize_ * (this->contextSize_ - 1));

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

  // Workspace-specific initialization
  init();

  size_t bitmask = 1;
  size_t stepChunkSize = chunkSize_ << (steps_ - 1);
  size_t stepChunkBytes = stepChunkSize * sizeof(T);
  size_t sendOffset = 0;
  size_t recvOffset = 0;
  size_t bufferOffset = 0; // offset into recvBuf_
  for (int i = 0; i < stepsWithinBlock_; i++) {
    const int destRank = static_cast<int>((this->context_->rank) ^ bitmask);
    auto& pair = this->context_->getPair(destRank);
    const auto myRank = this->context_->rank;
    auto slot = slotOffset_ +
        2 * (std::min(myRank, destRank) * this->contextSize_ +
             std::max(myRank, destRank));
    sendOffsets_[i] = sendOffset + ((destRank & bitmask) ? stepChunkSize : 0);
    recvOffsets_[i] =
        recvOffset + ((this->context_->rank & bitmask) ? stepChunkSize : 0);
    if (sendOffsets_[i] < count_) {
      // specifies number of elements of scratch_ buffer to send in each step
      if (sendOffsets_[i] + stepChunkSize > count_) {
        sendCounts_[i] = count_ - sendOffsets_[i];
      } else {
        sendCounts_[i] = stepChunkSize;
      }
    }
    sendDataBufs_.push_back(pair->createSendBuffer(slot, *scratch_, bytes_));
    if (recvOffsets_[i] < count_) {
      // specifies number of elements received in each step
      if (recvOffsets_[i] + stepChunkSize > count_) {
        recvCounts_[i] = count_ - recvOffsets_[i];
      } else {
        recvCounts_[i] = stepChunkSize;
      }
    }
    recvDataBufs_.push_back(
        pair->createRecvBuffer(
            slot, &recvBuf_[bufferOffset], stepChunkBytes));
    bufferOffset += stepChunkSize;
    if (this->context_->rank & bitmask) {
      sendOffset += stepChunkSize;
      recvOffset += stepChunkSize;
    }
    bitmask <<= 1;
    stepChunkSize >>= 1;
    stepChunkBytes >>= 1;

    ++slot;
    sendNotificationBufs_.push_back(
        pair->createSendBuffer(slot, &dummy_, sizeof(dummy_)));
    recvNotificationBufs_.push_back(
        pair->createRecvBuffer(slot, &dummy_, sizeof(dummy_)));
  }

  if (nextSmallerBlockSize_ != 0) {
    const auto offsetToSmallerBlock =
        offsetToMyBinaryBlock_ + myBinaryBlockSize_;
    const int destRank = static_cast<int>(
        offsetToSmallerBlock + rankInBinaryBlock_ % nextSmallerBlockSize_);
    auto& destPair = this->context_->getPair(destRank);
    const auto myRank = this->context_->rank;
    const auto slot = slotOffset_ +
        2 * (std::min(myRank, destRank) * this->contextSize_ +
             std::max(myRank, destRank));
    smallerBlockSendDataBuf_ = destPair->createSendBuffer(
        slot, *scratch_, bytes_);
    const auto itemCount = recvCounts_[stepsWithinBlock_ - 1];
    if (itemCount > 0) {
      smallerBlockRecvDataBuf_ = destPair->createRecvBuffer(
          slot, &recvBuf_[bufferOffset], itemCount * sizeof(T));
    }
  }
  if (nextLargerBlockSize_ != 0) {
    // Due to the design decision of sending large messages to nearby ranks,
    // after the reduce-scatter the reduced chunks end up in an order
    // according to the reversed bit pattern of each proc's rank within the
    // block. So, instead of ranks 0, 1, 2, ... 7 having blocks A, B, C, D, E,
    // F, G, H etc. what you get is A, E, C, G, B, F, D, H. Taking this
    // example further, if there is also a smaller binary block of size 2
    // (with the reduced blocks A - D, E - H), rank 0 within the smaller block
    // will need to send chunks of its buffer to ranks 0, 4, 2, 6 within the
    // larger block (in that order) and rank 1 will send to 1, 5, 3, 7. Within
    // the reversed bit patterns, this communication is actually 0 to [0, 1,
    // 2, 3] and 1 to [4, 5, 6, 7].
    const auto offsetToLargerBlock =
        offsetToMyBinaryBlock_ - nextLargerBlockSize_;
    const auto numSendsAndReceivesToLargerBlock =
        nextLargerBlockSize_ / myBinaryBlockSize_;
    const auto totalItemsToSend =
        stepsWithinBlock_ > 0 ? recvCounts_[stepsWithinBlock_ - 1] : count_;
    sendCountToLargerBlock_ = stepChunkSize >>
        (static_cast<size_t>(log2(numSendsAndReceivesToLargerBlock)) - 1);
    auto srcOrdinal =
        reverseLastNBits(rankInBinaryBlock_, log2(myBinaryBlockSize_));
    auto destOrdinal = srcOrdinal * numSendsAndReceivesToLargerBlock;
    for (int i = 0; i < numSendsAndReceivesToLargerBlock; i++) {
      const int destRank = offsetToLargerBlock +
          reverseLastNBits(destOrdinal, log2(nextLargerBlockSize_));
      auto& destPair = this->context_->getPair(destRank);
      const auto myRank = this->context_->rank;
      const auto slot = slotOffset_ +
          2 * (std::min(myRank, destRank) * this->contextSize_ +
               std::max(myRank, destRank));
      largerBlockSendDataBufs_.push_back(
          destPair->createSendBuffer(slot, *scratch_, bytes_));
      if (sendCountToLargerBlock_ * i < totalItemsToSend) {
        const auto toSend = std::min(
            sendCountToLargerBlock_,
            totalItemsToSend - sendCountToLargerBlock_ * i);
        largerBlockRecvDataBufs_.push_back(
            destPair->createRecvBuffer(
                slot, &recvBuf_[bufferOffset], toSend * sizeof(T)));
        bufferOffset += toSend;
      }
      destOrdinal++;
    }
  }

  if (pipelined_) {
    devicePointerInit();
    // Workspace-specific initialization for pipelined reductions/broadcasts
    initReductionsAndBroadcasts();
  }
}

template <typename T, typename W>
void CudaAllreduceHalvingDoubling<T, W>::run() {
  CudaDeviceGuard guard;
  CudaStream& stream = streams_[0];
  size_t bufferOffset = 0;
  size_t numItems = stepsWithinBlock_ > 0 ? chunkSize_ << (steps_ - 1) : count_;

  if (pipelined_ && reduceBeforeFirstSend_) {
    reduceBeforeFirstSend_->run();
  } else if (localReduceOp_) {
    localReduceOp_->run();
  }

  // Reduce-scatter
  for (int i = 0; i < stepsWithinBlock_; i++) {
    if (sendOffsets_[i] < count_) {
      sendDataBufs_[i]->send(
          sendOffsets_[i] * sizeof(T), sendCounts_[i] * sizeof(T));
    }
    if (recvOffsets_[i] < count_) {
      if (pipelined_ && i == 0 && reduceBeforeFirstRecv_) {
        reduceBeforeFirstRecv_->runAsync();
      }
      recvDataBufs_[i]->waitRecv();
      if (pipelined_ && i == 0 && reduceBeforeFirstRecv_) {
        reduceBeforeFirstRecv_->wait();
      }
      auto recvBufAtOffset = recvBuf_.range(bufferOffset, recvCounts_[i]);
      auto scratchAtOffset = scratch_.range(recvOffsets_[i], recvCounts_[i]);
      fn_->call(scratchAtOffset, recvBufAtOffset, recvCounts_[i], stream);
      stream.wait();
    }
    sendNotificationBufs_[i]->send();
    bufferOffset += numItems;
    if (i != stepsWithinBlock_ - 1) {
      numItems >>= 1;
    }
  }

  // Communication across binary blocks for non-power-of-two number of
  // processes

  // receive from smaller block
  // data sizes same as in the last step of intrablock reduce-scatter above
  if (nextSmallerBlockSize_ != 0 && smallerBlockRecvDataBuf_ != nullptr) {
    smallerBlockRecvDataBuf_->waitRecv();
    auto recvBufAtOffset =
        recvBuf_.range(bufferOffset, recvCounts_[stepsWithinBlock_ - 1]);
    auto scratchAtOffset = scratch_.range(
        recvOffsets_[stepsWithinBlock_ - 1],
        recvCounts_[stepsWithinBlock_ - 1]);
    fn_->call(
        scratchAtOffset,
        recvBufAtOffset,
        recvCounts_[stepsWithinBlock_ - 1],
        stream);
    stream.wait();
  }

  const auto totalItemsToSend =
      stepsWithinBlock_ > 0 ? recvCounts_[stepsWithinBlock_ - 1] : count_;
  if (nextLargerBlockSize_ != 0 && totalItemsToSend != 0) {
    // scatter to larger block
    const auto offset =
        stepsWithinBlock_ > 0 ? recvOffsets_[stepsWithinBlock_ - 1] : 0;
    const auto numSendsAndReceivesToLargerBlock =
        nextLargerBlockSize_ / myBinaryBlockSize_;
    for (int i = 0; i < numSendsAndReceivesToLargerBlock; i++) {
      if (sendCountToLargerBlock_ * i < totalItemsToSend) {
        largerBlockSendDataBufs_[i]->send(
            (offset + i * sendCountToLargerBlock_) * sizeof(T),
            std::min(
                sendCountToLargerBlock_,
                totalItemsToSend - sendCountToLargerBlock_ * i) *
                sizeof(T));
      }
    }
    // no notification is needed because the forward and backward messages
    // across blocks are serialized in relation to each other

    // receive from larger blocks
    for (int i = 0; i < numSendsAndReceivesToLargerBlock; i++) {
      if (sendCountToLargerBlock_ * i < totalItemsToSend) {
        largerBlockRecvDataBufs_[i]->waitRecv();
      }
    }
    auto recvBufAtOffset = recvBuf_.range(bufferOffset, totalItemsToSend);
    auto scratchAtOffset = scratch_.range(offset, totalItemsToSend);
    // msg from larger block is the final result, no reduce needed
    stream.copyAsync(scratchAtOffset, recvBufAtOffset);
    stream.wait();
  }

  // Send to smaller block (technically the beginning of allgather)
  bool sentToSmallerBlock = false;
  if (nextSmallerBlockSize_ != 0) {
    if (recvOffsets_[stepsWithinBlock_ - 1] < count_) {
      sentToSmallerBlock = true;
      smallerBlockSendDataBuf_->send(
          recvOffsets_[stepsWithinBlock_ - 1] * sizeof(T),
          recvCounts_[stepsWithinBlock_ - 1] * sizeof(T));
    }
  }

  // Allgather
  numItems = chunkSize_ << (steps_ - stepsWithinBlock_);
  for (int i = stepsWithinBlock_ - 1; i >= 0; i--) {
    // verify that destination rank has received and processed this rank's
    // message during the reduce-scatter phase
    recvNotificationBufs_[i]->waitRecv();
    if (recvOffsets_[i] < count_) {
      sendDataBufs_[i]->send(
          recvOffsets_[i] * sizeof(T), recvCounts_[i] * sizeof(T));
    }
    bufferOffset -= numItems;
    if (sendOffsets_[i] < count_) {
      recvDataBufs_[i]->waitRecv();
      auto recvBufAtOffset = recvBuf_.range(bufferOffset, sendCounts_[i]);
      auto scratchAtOffset = scratch_.range(sendOffsets_[i], sendCounts_[i]);
      stream.copyAsync(scratchAtOffset, recvBufAtOffset);
      stream.wait();
    }
    if (pipelined_ && broadcastOps_[i]) {
      broadcastOps_[i]->runAsync();
    }
    numItems <<= 1;

    // Send notification to the pair we just received from that
    // we're done dealing with the receive buffer.
    sendNotificationBufs_[i]->send();
  }

  if (pipelined_ && stepsWithinBlock_ > 0) {
    for (int i = stepsWithinBlock_ - 1; i >= 0; i--) {
      if (broadcastOps_[i]) {
        broadcastOps_[i]->wait();
      }
    }
  } else if (localBroadcastOp_) {
    localBroadcastOp_->runAsync();
    localBroadcastOp_->wait();
  }

  // Wait for notifications from our peers within the block to make
  // sure we can send data immediately without risking overwriting
  // data in its receive buffer before it consumed that data.
  for (int i = stepsWithinBlock_ - 1; i >= 0; i--) {
    recvNotificationBufs_[i]->waitRecv();
  }

  // We have to be sure the send to the smaller block (if any) has
  // completed before returning. If we don't, the buffer contents may
  // be modified by our caller.
  if (sentToSmallerBlock) {
    smallerBlockSendDataBuf_->waitSend();
  }
}

template <typename T, typename W>
void CudaAllreduceHalvingDoubling<T, W>::devicePointerInit() {
  size_t offset, numElements;

  for (int i = 0; i < stepsWithinBlock_; i++) {
    // in the first broadcast (with step 'steps_ - 1'), include both the local
    // chunk result from reduce-scatter and the first received chunk
    offset = i == stepsWithinBlock_ - 1
        ? std::min(recvOffsets_[i], sendOffsets_[i])
        : sendOffsets_[i];
    numElements = i == stepsWithinBlock_ - 1 ? recvCounts_[i] + sendCounts_[i]
                                             : sendCounts_[i];
    if (offset > count_) {
      scratchPtrForBroadcast_.push_back(typename W::Pointer());
      continue;
    }
    if (offset + numElements > count_) {
      numElements = count_ - offset;
    }

    scratchPtrForBroadcast_.push_back(scratch_.range(offset, numElements));
    for (int j = 0; j < devicePtrs_.size(); j++) {
      devicePtrsForBroadcast_[i].push_back(
          devicePtrs_[j].range(offset, numElements));
    }
  }
  if (sendOffsets_[0] < count_) {
    scratchPtrForFirstSend_ = scratch_.range(sendOffsets_[0], sendCounts_[0]);
  }
  if (recvOffsets_[0] < count_) {
    scratchPtrForFirstRecv_ = scratch_.range(recvOffsets_[0], recvCounts_[0]);
  }

  for (int i = 0; i < devicePtrs_.size(); i++) {
    if (sendOffsets_[0] < count_) {
      devicePtrsForFirstSend_.push_back(
          devicePtrs_[i].range(sendOffsets_[0], sendCounts_[0]));
    }
    if (recvOffsets_[0] < count_) {
      devicePtrsForFirstRecv_.push_back(
          devicePtrs_[i].range(recvOffsets_[0], recvCounts_[0]));
    }
  }
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceHalvingDoubling<T, W>::init(
    typename std::enable_if<
        std::is_same<U, CudaHostWorkspace<T>>::value,
        typename U::Pointer>::type*) {
  // Since reduction is executed on the CPU, the scratch space
  // where they are accumulated is a new host side buffer.
  scratch_ = W::Pointer::alloc(count_);
  // pad receive buffer size to nearest power of 2 to ensure sufficient space
  recvBuf_ = W::Pointer::alloc(chunkSize_ << steps_);

  // Set up local reduction and broadcast operations on the host.
  // If devicePtrs_.size() == 1 these functions construct an op that
  // executes a memcpy such that scratch_ always holds the result.

  // local reduce and broadcast ops are only used in the non-pipelined case and
  // for blocks of size 1
  if (pipelined_ && stepsWithinBlock_ > 0) {
    return;
  }
  if (bytes_ < kOnDeviceThreshold) {
    localReduceOp_ =
        cudaHostReduce(streams_, devicePtrs_, scratch_, fn_, 0, count_);
    localBroadcastOp_ =
        cudaHostBroadcast(streams_, devicePtrs_, scratch_, 0, count_);
  } else {
    localReduceOp_ =
        cudaDeviceReduce(streams_, devicePtrs_, scratch_, fn_, 0, count_);
    localBroadcastOp_ =
        cudaDeviceBroadcast(streams_, devicePtrs_, scratch_, 0, count_);
  }
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceHalvingDoubling<T, W>::init(
    typename std::enable_if<
        std::is_same<U, CudaDeviceWorkspace<T>>::value,
        typename U::Pointer>::type*) {
  // Since reduction is executed on the GPU, the scratch space
  // can use an existing input buffer to accumulate.
  auto& ptr = devicePtrs_[0];
  auto count = ptr.getCount();
  scratch_ = CudaDevicePointer<T>::create(*ptr, count);

  // Inbox/outbox must be colocated with scratch buffer to avoid
  // cross device copies while accumulating the reduction.
  {
    CudaDeviceScope scope(scratch_.getDeviceID());
    // pad receive buffer size to nearest power of 2 to ensure sufficient space
    recvBuf_ = W::Pointer::alloc(chunkSize_ << steps_);
  }

  // Set up local reduction and broadcast operations on the device.
  // When running with a device workspace we intend to never leave the device.

  // local reduce and broadcast ops are only used in the non-pipelined case and
  // for blocks of size 1
  if (pipelined_ && stepsWithinBlock_ > 0) {
    return;
  }
  if (devicePtrs_.size() > 1) {
    localReduceOp_ =
        cudaDeviceReduce(streams_, devicePtrs_, scratch_, fn_, 0, count_);
    localBroadcastOp_ =
        cudaDeviceBroadcast(streams_, devicePtrs_, scratch_, 0, count_);
  }
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceHalvingDoubling<T, W>::initReductionsAndBroadcasts(
    typename std::enable_if<
        std::is_same<U, CudaHostWorkspace<T>>::value,
        typename U::Pointer>::type*) {
  if (stepsWithinBlock_ == 0) {
    return;
  }
  if (sendCounts_[0] * sizeof(T) < kOnDeviceThreshold) {
    if (!devicePtrsForFirstSend_.empty()) {
      reduceBeforeFirstSend_ = cudaHostReduce(
          streams_,
          devicePtrsForFirstSend_,
          scratchPtrForFirstSend_,
          fn_,
          0,
          sendCounts_[0]);
    }
    if (!devicePtrsForFirstRecv_.empty()) {
      reduceBeforeFirstRecv_ = cudaHostReduce(
          streams_,
          devicePtrsForFirstRecv_,
          scratchPtrForFirstRecv_,
          fn_,
          0,
          recvCounts_[0]);
    }
  } else {
    if (!devicePtrsForFirstSend_.empty()) {
      reduceBeforeFirstSend_ = cudaDeviceReduce(
          streams_,
          devicePtrsForFirstSend_,
          scratchPtrForFirstSend_,
          fn_,
          0,
          sendCounts_[0]);
    }
    if (!devicePtrsForFirstRecv_.empty()) {
      reduceBeforeFirstRecv_ = cudaDeviceReduce(
          streams_,
          devicePtrsForFirstRecv_,
          scratchPtrForFirstRecv_,
          fn_,
          0,
          recvCounts_[0]);
    }
  }
  for (int i = 0; i < stepsWithinBlock_; i++) {
    if (devicePtrsForBroadcast_[i].empty()) {
      broadcastOps_.push_back(nullptr);
      continue;
    }
    const size_t numElementsInBcast = i == stepsWithinBlock_ - 1
        ? sendCounts_[i] + recvCounts_[i]
        : sendCounts_[i];
    if (numElementsInBcast * sizeof(T) < kOnDeviceThreshold) {
      broadcastOps_.push_back(cudaHostBroadcast(
          streams_,
          devicePtrsForBroadcast_[i],
          scratchPtrForBroadcast_[i],
          0,
          numElementsInBcast));
    } else {
      broadcastOps_.push_back(cudaDeviceBroadcast(
          streams_,
          devicePtrsForBroadcast_[i],
          scratchPtrForBroadcast_[i],
          0,
          numElementsInBcast));
    }
  }
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceHalvingDoubling<T, W>::initReductionsAndBroadcasts(
    typename std::enable_if<
        std::is_same<U, CudaDeviceWorkspace<T>>::value,
        typename U::Pointer>::type*) {
  if (stepsWithinBlock_ == 0) {
    return;
  }
  if (!devicePtrsForFirstSend_.empty()) {
    reduceBeforeFirstSend_ = cudaDeviceReduce(
        streams_,
        devicePtrsForFirstSend_,
        scratchPtrForFirstSend_,
        fn_,
        0,
        sendCounts_[0]);
  }
  if (!devicePtrsForFirstRecv_.empty()) {
    reduceBeforeFirstRecv_ = cudaDeviceReduce(
        streams_,
        devicePtrsForFirstRecv_,
        scratchPtrForFirstRecv_,
        fn_,
        0,
        recvCounts_[0]);
  }
  for (int i = 0; i < stepsWithinBlock_; i++) {
    if (devicePtrsForBroadcast_[i].empty()) {
      broadcastOps_.push_back(nullptr);
      continue;
    }
    broadcastOps_.push_back(cudaDeviceBroadcast(
        streams_,
        devicePtrsForBroadcast_[i],
        scratchPtrForBroadcast_[i],
        0,
        i == stepsWithinBlock_ - 1 ? sendCounts_[i] + recvCounts_[i]
                                   : sendCounts_[i]));
  }
}

#define INSTANTIATE_TEMPLATE(T)                                         \
  template class CudaAllreduceHalvingDoubling<T, CudaHostWorkspace<T>>; \
  template class CudaAllreduceHalvingDoubling<T, CudaDeviceWorkspace<T>>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
