/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "TypedMetadata.h"

// Catalog of typed metadata field keys, grouped by producer domain.

namespace libkineto::GenericMetadataFields {
inline constexpr MetadataField<InputShapes> kInputDims{"Input Dims"};
inline constexpr MetadataField<InputShapes> kInputStrides{"Input Strides"};
inline constexpr MetadataField<std::vector<std::string>> kInputType{
    "Input type"};
inline constexpr MetadataField<int64_t> kSequenceNumber{"Sequence number"};
inline constexpr MetadataField<uint64_t> kFwdThreadId{"Fwd thread id"};
inline constexpr MetadataField<uint64_t> kRecordFunctionId{
    "Record function id"};
inline constexpr MetadataField<int64_t> kEvIdx{"Ev Idx"};
inline constexpr MetadataField<uint64_t> kAddr{"Addr"};
inline constexpr MetadataField<int64_t> kBytes{"Bytes"};
inline constexpr MetadataField<uint64_t> kTotalAllocated{"Total Allocated"};
inline constexpr MetadataField<uint64_t> kTotalReserved{"Total Reserved"};
inline constexpr MetadataField<int64_t> kDeviceType{"Device Type"};
inline constexpr MetadataField<int64_t> kDeviceId{"Device Id"};
inline constexpr MetadataField<uint64_t> kPythonId{"Python id"};
inline constexpr MetadataField<uint64_t> kPythonParentId{"Python parent id"};
inline constexpr MetadataField<uint64_t> kPythonModuleId{"Python module id"};
inline constexpr MetadataField<uint64_t> kPythonThread{"Python thread"};
inline constexpr MetadataField<std::string> kBackend{"Backend"};
inline constexpr MetadataField<std::string> kCallFrom{"CallFrom"};
inline constexpr MetadataField<std::string> kCallStack{"Call stack"};
inline constexpr MetadataField<std::string> kModuleHierarchy{
    "Module Hierarchy"};
inline constexpr MetadataField<std::vector<std::string>> kConcreteInputs{
    "Concrete Inputs"};
} // namespace libkineto::GenericMetadataFields

namespace libkineto::CollectiveMetadataFields {
inline constexpr MetadataField<std::string> kCollectiveName{"Collective name"};
inline constexpr MetadataField<std::string> kDtype{"dtype"};
inline constexpr MetadataField<int64_t> kInMsgNelems{"In msg nelems"};
inline constexpr MetadataField<int64_t> kOutMsgNelems{"Out msg nelems"};
inline constexpr MetadataField<std::string> kInSplit{"In split size"};
inline constexpr MetadataField<std::string> kOutSplit{"Out split size"};
inline constexpr MetadataField<int64_t> kGlobalRankStart{"Global rank start"};
inline constexpr MetadataField<int64_t> kGlobalRankStride{"Global rank stride"};
inline constexpr MetadataField<int64_t> kGroupSize{"Group size"};
inline constexpr MetadataField<std::string> kProcessGroupName{
    "Process Group Name"};
inline constexpr MetadataField<std::string> kProcessGroupDesc{
    "Process Group Description"};
inline constexpr MetadataField<std::string> kGroupRanks{"Process Group Ranks"};
inline constexpr MetadataField<int64_t> kRank{"Rank"};
inline constexpr MetadataField<int64_t> kP2pSrc{"Src Rank"};
inline constexpr MetadataField<int64_t> kP2pDst{"Dst Rank"};
inline constexpr MetadataField<int64_t> kSeqNum{"Seq"};
inline constexpr MetadataField<std::string> kInTensorsStart{
    "Input Tensors start"};
inline constexpr MetadataField<std::string> kOutTensorsStart{
    "Output Tensors start"};
inline constexpr MetadataField<bool> kIsAsynchronizedOp{"Is asynchronized op"};
inline constexpr MetadataField<uint64_t> kCommsId{"Comms Id"};
} // namespace libkineto::CollectiveMetadataFields

namespace libkineto::DevicePropertyMetadataFields {
inline constexpr MetadataField<uint64_t> kId{"id"};
inline constexpr MetadataField<std::string> kName{"name"};
inline constexpr MetadataField<uint64_t> kTotalGlobalMem{"totalGlobalMem"};
inline constexpr MetadataField<int64_t> kComputeMajor{"computeMajor"};
inline constexpr MetadataField<int64_t> kComputeMinor{"computeMinor"};
inline constexpr MetadataField<int64_t> kMaxThreadsPerBlock{
    "maxThreadsPerBlock"};
inline constexpr MetadataField<int64_t> kMaxThreadsPerMultiprocessor{
    "maxThreadsPerMultiprocessor"};
inline constexpr MetadataField<int64_t> kRegsPerBlock{"regsPerBlock"};
inline constexpr MetadataField<int64_t> kWarpSize{"warpSize"};
inline constexpr MetadataField<uint64_t> kSharedMemPerBlock{
    "sharedMemPerBlock"};
inline constexpr MetadataField<int64_t> kNumSms{"numSms"};
inline constexpr MetadataField<int64_t> kRegsPerMultiprocessor{
    "regsPerMultiprocessor"};
inline constexpr MetadataField<uint64_t> kSharedMemPerBlockOptin{
    "sharedMemPerBlockOptin"};
inline constexpr MetadataField<uint64_t> kSharedMemPerMultiprocessor{
    "sharedMemPerMultiprocessor"};
inline constexpr MetadataField<uint64_t> kMaxSharedMemoryPerMultiProcessor{
    "maxSharedMemoryPerMultiProcessor"};

// XPU-only device properties.
inline constexpr MetadataField<uint64_t> kMaxComputeUnits{"maxComputeUnits"};
inline constexpr MetadataField<uint64_t> kMaxWorkGroupSize{"maxWorkGroupSize"};
inline constexpr MetadataField<uint64_t> kMaxClockFrequency{
    "maxClockFrequency"};
inline constexpr MetadataField<uint64_t> kMaxMemAllocSize{"maxMemAllocSize"};
inline constexpr MetadataField<uint64_t> kLocalMemSize{"localMemSize"};
inline constexpr MetadataField<std::string> kVendor{"vendor"};
inline constexpr MetadataField<std::string> kDriverVersion{"driverVersion"};
} // namespace libkineto::DevicePropertyMetadataFields

namespace libkineto::CudaMetadataFields {
inline constexpr MetadataDict kOccupancy{"occupancy"};
inline constexpr MetadataField<int64_t> kActiveBlocksPerMultiprocessor{
    "activeBlocksPerMultiprocessor"};
inline constexpr MetadataField<int64_t> kAllocatedRegistersPerBlock{
    "allocatedRegistersPerBlock"};
inline constexpr MetadataField<uint64_t> kAllocatedSharedMemPerBlock{
    "allocatedSharedMemPerBlock"};
inline constexpr MetadataField<std::vector<int64_t>> kBlock{"block"};
inline constexpr MetadataField<int64_t> kBlockLimitBarriers{
    "blockLimitBarriers"};
inline constexpr MetadataField<int64_t> kBlockLimitBlocks{"blockLimitBlocks"};
inline constexpr MetadataField<int64_t> kBlockLimitRegs{"blockLimitRegs"};
inline constexpr MetadataField<int64_t> kBlockLimitSharedMem{
    "blockLimitSharedMem"};
inline constexpr MetadataField<int64_t> kBlockLimitWarps{"blockLimitWarps"};
inline constexpr MetadataField<double> kBlocksPerSm{"blocks per SM"};
inline constexpr MetadataField<uint64_t> kBytes{"bytes"};
inline constexpr MetadataField<uint64_t> kCbid{"cbid"};
inline constexpr MetadataField<uint64_t> kChannel{"channel"};
inline constexpr MetadataField<int64_t> kChannelType{"channel_type"};
inline constexpr MetadataField<uint64_t> kContext{"context"};
inline constexpr MetadataField<uint64_t> kCorrelation{"correlation"};
inline constexpr MetadataField<std::string> kCudaSyncKind{"cuda_sync_kind"};
inline constexpr MetadataField<int64_t> kDevice{"device"};
inline constexpr MetadataField<int64_t> kEstAchievedOccupancyPercent{
    "est. achieved occupancy %"};
inline constexpr MetadataField<uint64_t> kEventId{"event_id"};
inline constexpr MetadataField<uint64_t> kFromContext{"fromContext"};
inline constexpr MetadataField<uint64_t> kFromDevice{"fromDevice"};
inline constexpr MetadataField<uint64_t> kGraphId{"graph id"};
inline constexpr MetadataField<uint64_t> kGraphNodeId{"graph node id"};
inline constexpr MetadataField<std::vector<int64_t>> kGrid{"grid"};
inline constexpr MetadataField<uint64_t> kInContext{"inContext"};
inline constexpr MetadataField<uint64_t> kInDevice{"inDevice"};
inline constexpr MetadataField<std::string> kLimitingFactors{"limitingFactors"};
inline constexpr MetadataField<double> kMemoryBandwidthGbps{
    "memory bandwidth (GB/s)"};
inline constexpr MetadataField<int64_t> kPriority{"priority"};
inline constexpr MetadataField<uint64_t> kQueued{"queued"};
inline constexpr MetadataField<uint64_t> kRegistersPerThread{
    "registers per thread"};
inline constexpr MetadataField<int64_t> kSharedMemory{"shared memory"};
inline constexpr MetadataField<int64_t> kStream{"stream"};
inline constexpr MetadataField<uint64_t> kToContext{"toContext"};
inline constexpr MetadataField<uint64_t> kToDevice{"toDevice"};
inline constexpr MetadataField<uint64_t> kWaitOnCudaEventId{
    "wait_on_cuda_event_id"};
inline constexpr MetadataField<int64_t> kWaitOnCudaEventRecordCorrId{
    "wait_on_cuda_event_record_corr_id"};
inline constexpr MetadataField<int64_t> kWaitOnStream{"wait_on_stream"};
inline constexpr MetadataField<double> kWarpsPerSm{"warps per SM"};
} // namespace libkineto::CudaMetadataFields

namespace libkineto::RocmMetadataFields {
inline constexpr MetadataField<std::vector<int64_t>> kBlock{"block"};
inline constexpr MetadataField<uint64_t> kBytes{"bytes"};
inline constexpr MetadataField<uint64_t> kCid{"cid"};
inline constexpr MetadataField<uint64_t> kCorrelation{"correlation"};
inline constexpr MetadataField<int64_t> kDevice{"device"};
inline constexpr MetadataField<std::string> kDst{"dst"};
inline constexpr MetadataField<std::vector<int64_t>> kGrid{"grid"};
inline constexpr MetadataField<uint64_t> kHsaQueue{"hsa_queue"};
inline constexpr MetadataField<std::string> kKernel{"kernel"};
inline constexpr MetadataField<std::string> kKind{"kind"};
inline constexpr MetadataField<double> kMemoryBandwidthGbps{
    "memory bandwidth (GB/s)"};
inline constexpr MetadataField<std::string> kPtr{"ptr"};
inline constexpr MetadataField<uint64_t> kSharedMemory{"shared memory"};
inline constexpr MetadataField<std::string> kSrc{"src"};
inline constexpr MetadataField<int64_t> kStream{"stream"};
} // namespace libkineto::RocmMetadataFields

namespace libkineto::XpuMetadataFields {
inline constexpr MetadataField<uint64_t> kBytes{"bytes"};
inline constexpr MetadataField<uint64_t> kCommunicatorId{"Communicator_id"};
inline constexpr MetadataField<uint64_t> kCorrelation{"correlation"};
inline constexpr MetadataField<int64_t> kDevice{"device"};
inline constexpr MetadataField<uint64_t> kEngineIndex{"engine_index"};
inline constexpr MetadataField<uint64_t> kEngineOrdinal{"engine_ordinal"};
inline constexpr MetadataField<uint64_t> kKernelId{"kernel_id"};
inline constexpr MetadataField<double> kMemoryBandwidthGbps{
    "memory bandwidth (GB/s)"};
inline constexpr MetadataField<uint64_t> kMemoryOperationId{
    "memory opration id"};
inline constexpr MetadataField<uint64_t> kNumberWaitEvents{
    "Number_wait_events"};
inline constexpr MetadataField<uint64_t> kOverheadCost{"overhead cost"};
inline constexpr MetadataField<uint64_t> kOverheadCount{"overhead count"};
inline constexpr MetadataField<int64_t> kReturnCode{"Return_code"};
inline constexpr MetadataField<uint64_t> kSourceLineNumber{
    "source_line_number"};
inline constexpr MetadataField<uint64_t> kSyclInvocationId{
    "sycl_invocation_id"};
inline constexpr MetadataField<uint64_t> kSyclNodeId{"sycl_node_id"};
inline constexpr MetadataField<uint64_t> kSyclQueue{"sycl queue"};
} // namespace libkineto::XpuMetadataFields
