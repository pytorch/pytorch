#include <THD/process_group/Collectives.hpp>
#include <THD/base/ChannelUtils.hpp>
#include <THD/process_group/General.hpp>

#include <vector>

using namespace thd;

int THDGetRank() {
  return static_cast<int>(dataChannel->getRank());
}

int THDGetNumProcesses() {
  return static_cast<int>(dataChannel->getNumProcesses());
}

void THDAllReduceMultiGPU(
    THDTensorDescriptor* data,
    size_t len,
    THDReduceOp operation,
    THDGroup group) {
  std::vector<at::Tensor> dataVec(data, data + len);
  dataChannel->allReduce(dataVec, operation, group);
}

void THDAllReduce(
    THDTensorDescriptor& desc,
    THDReduceOp operation,
    THDGroup group) {
  dataChannel->allReduce(desc, operation, group);
}

void THDReduceMultiGPU(
    THDTensorDescriptor* desc,
    size_t len,
    THDReduceOp operation,
    int dst_rank,
    THDGroup group) {
  std::vector<at::Tensor> dataVec(desc, desc + len);
  dataChannel->reduce(dataVec, operation, convertToRank(dst_rank), group);
}

void THDReduce(
    THDTensorDescriptor& desc,
    THDReduceOp operation,
    int dst_rank,
    THDGroup group) {
  dataChannel->reduce(desc, operation, convertToRank(dst_rank), group);
}

void THDBroadcastMultiGPU(
    THDTensorDescriptor* desc,
    size_t len,
    int src_rank,
    THDGroup group) {
  std::vector<at::Tensor> dataVec(desc, desc + len);
  dataChannel->broadcast(dataVec, convertToRank(src_rank), group);
}

void THDBroadcast(THDTensorDescriptor& desc, int src_rank, THDGroup group) {
  dataChannel->broadcast(desc, convertToRank(src_rank), group);
}

THDRequest* THDIsend(THDTensorDescriptor& desc, int dst_rank) {
  return dataChannel->isend(desc, convertToRank(dst_rank));
}

THDRequest* THDIrecv(THDTensorDescriptor& desc, int src_rank) {
  return dataChannel->ireceive(desc, convertToRank(src_rank));
}

void THDSend(THDTensorDescriptor& desc, int dst_rank) {
  dataChannel->send(desc, convertToRank(dst_rank));
}

int THDRecvAnySource(THDTensorDescriptor& desc) {
  return dataChannel->receive(desc);
}

void THDRecv(THDTensorDescriptor& desc, int src_rank) {
  dataChannel->receive(desc, convertToRank(src_rank));
}

void THDAllGatherMultiGPU(
    THDTensorDescriptor* output,
    size_t outputLen,
    THDTensorDescriptor* input,
    size_t inputLen,
    THDGroup group) {
  std::vector<at::Tensor> outputVec(output, output + outputLen);
  std::vector<at::Tensor> inputVec(input, input + inputLen);
  dataChannel->allGather(outputVec, inputVec, group);
}

void THDAllGather(
    THDTensorDescriptor* output,
    size_t len,
    THDTensorDescriptor& input,
    THDGroup group) {
  std::vector<at::Tensor> v_output(output, output + len);
  dataChannel->allGather(v_output, input, group);
}

void THDGatherSend(THDTensorDescriptor& input, int dst_rank, THDGroup group) {
  std::vector<at::Tensor> v_output;
  dataChannel->gather(v_output, input, convertToRank(dst_rank), group);
}

void THDGatherRecv(
    THDTensorDescriptor* output,
    size_t len,
    THDTensorDescriptor& input,
    THDGroup group) {
  std::vector<at::Tensor> v_output(output, output + len);
  dataChannel->gather(v_output, input, dataChannel->getRank(), group);
}

void THDScatterSend(
    THDTensorDescriptor* input,
    size_t len,
    THDTensorDescriptor& output,
    THDGroup group) {
  std::vector<at::Tensor> v_input(input, input + len);
  dataChannel->scatter(v_input, output, dataChannel->getRank(), group);
}

void THDScatterRecv(THDTensorDescriptor& output, int src_rank, THDGroup group) {
  if (src_rank < 0)
    throw std::domain_error("src_rank should not be negative");

  std::vector<at::Tensor> v_input;
  dataChannel->scatter(v_input, output, convertToRank(src_rank), group);
}

void THDBarrier(THDGroup group) {
  dataChannel->barrier(group);
}

THDGroup THDNewGroup(const int* ranks, size_t len) {
  std::vector<rank_type> v_ranks(len);
  for (size_t i = 0; i < len; ++i) {
    v_ranks[i] = convertToRank(ranks[i]);
  }

  return dataChannel->newGroup(v_ranks);
}

bool THDRequest_isCompleted(THDRequest* request) {
  return request->isCompleted();
}

void THDRequest_wait(THDRequest* request) {
  request->wait();
}
