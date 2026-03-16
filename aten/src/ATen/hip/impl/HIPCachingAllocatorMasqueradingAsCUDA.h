#pragma once

#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace c10 {
// forward declaration
class DataPtr;
namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

C10_CUDA_API c10::cuda::CUDACachingAllocator::CUDAAllocator* get();
C10_CUDA_API void recordStreamMasqueradingAsCUDA(const DataPtr& ptr, HIPStreamMasqueradingAsCUDA stream);

using c10::cuda::CUDACachingAllocator::raw_alloc;
using c10::cuda::CUDACachingAllocator::raw_alloc_with_stream;
using c10::cuda::CUDACachingAllocator::raw_delete;
using c10::cuda::CUDACachingAllocator::init;
using c10::cuda::CUDACachingAllocator::getMemoryFraction;
using c10::cuda::CUDACachingAllocator::setMemoryFraction;
using c10::cuda::CUDACachingAllocator::emptyCache;
using c10::cuda::CUDACachingAllocator::enable;
using c10::cuda::CUDACachingAllocator::isEnabled;
using c10::cuda::CUDACachingAllocator::cacheInfo;
using c10::cuda::CUDACachingAllocator::getBaseAllocation;
using c10::cuda::CUDACachingAllocator::getDeviceStats;
using c10::cuda::CUDACachingAllocator::resetAccumulatedStats;
using c10::cuda::CUDACachingAllocator::resetPeakStats;
using c10::cuda::CUDACachingAllocator::snapshot;
using c10::cuda::CUDACachingAllocator::getCheckpointState;
using c10::cuda::CUDACachingAllocator::setCheckpointPoolState;
using c10::cuda::CUDACachingAllocator::beginAllocateToPool;
using c10::cuda::CUDACachingAllocator::endAllocateToPool;
using c10::cuda::CUDACachingAllocator::recordHistory;
using c10::cuda::CUDACachingAllocator::recordAnnotation;
using c10::cuda::CUDACachingAllocator::pushCompileContext;
using c10::cuda::CUDACachingAllocator::popCompileContext;
using c10::cuda::CUDACachingAllocator::isHistoryEnabled;
using c10::cuda::CUDACachingAllocator::checkPoolLiveAllocations;
using c10::cuda::CUDACachingAllocator::attachOutOfMemoryObserver;
using c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker;
using c10::cuda::CUDACachingAllocator::releasePool;
using c10::cuda::CUDACachingAllocator::createOrIncrefPool;
using c10::cuda::CUDACachingAllocator::setUseOnOOM;
using c10::cuda::CUDACachingAllocator::setNoSplit;
using c10::cuda::CUDACachingAllocator::getPoolUseCount;
using c10::cuda::CUDACachingAllocator::getIpcDevPtr;
using c10::cuda::CUDACachingAllocator::shareIpcHandle;
using c10::cuda::CUDACachingAllocator::name;
using c10::cuda::CUDACachingAllocator::memcpyAsync;
using c10::cuda::CUDACachingAllocator::enablePeerAccess;
} // namespace HIPCachingAllocatorMasqueradingAsCUDA
} // namespace hip
} // namespace c10
