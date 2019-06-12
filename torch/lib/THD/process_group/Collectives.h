#pragma once

#include "../THD.h"
#include "../base/DataChannel.h"

THD_API int THDGetRank();
THD_API int THDGetNumProcesses();
THD_API void THDAllReduceMultiGPU(
    THDTensorDescriptor* data,
    size_t len,
    THDReduceOp operation,
    THDGroup group);
THD_API void THDAllReduce(
    THDTensorDescriptor& desc,
    THDReduceOp operation,
    THDGroup group);
THD_API void THDReduceMultiGPU(
    THDTensorDescriptor* desc,
    size_t len,
    THDReduceOp operation,
    int dst_rank,
    THDGroup group);
THD_API void THDReduce(
    THDTensorDescriptor& desc,
    THDReduceOp operation,
    int dst_rank,
    THDGroup group);
THD_API void THDBroadcastMultiGPU(
    THDTensorDescriptor* desc,
    size_t len,
    int src_rank,
    THDGroup group);
THD_API void THDBroadcast(
    THDTensorDescriptor& desc,
    int src_rank,
    THDGroup group);
THD_API THDRequest* THDIsend(THDTensorDescriptor& desc, int dst_rank);
THD_API THDRequest* THDIrecv(THDTensorDescriptor& desc, int src_rank);
THD_API void THDSend(THDTensorDescriptor& desc, int dst_rank);
THD_API int THDRecvAnySource(THDTensorDescriptor& desc);
THD_API void THDRecv(THDTensorDescriptor& desc, int src_rank);
THD_API void THDAllGatherMultiGPU(
    THDTensorDescriptor* output,
    size_t outputLen,
    THDTensorDescriptor* input,
    size_t inputLen,
    THDGroup group);
THD_API void THDAllGather(
    THDTensorDescriptor* output,
    size_t len,
    THDTensorDescriptor& input,
    THDGroup group);
THD_API void THDGatherSend(
    THDTensorDescriptor& input,
    int dst_rank,
    THDGroup group);
THD_API void THDGatherRecv(
    THDTensorDescriptor* output,
    size_t len,
    THDTensorDescriptor& input,
    THDGroup group);
THD_API void THDScatterSend(
    THDTensorDescriptor* input,
    size_t len,
    THDTensorDescriptor& output,
    THDGroup group);
THD_API void THDScatterRecv(
    THDTensorDescriptor& output,
    int src_rank,
    THDGroup group);
THD_API void THDBarrier(THDGroup group);
THD_API THDGroup THDNewGroup(const int* ranks, size_t len);
THD_API bool THDRequest_isCompleted(THDRequest* request);
THD_API void THDRequest_wait(THDRequest* request);
