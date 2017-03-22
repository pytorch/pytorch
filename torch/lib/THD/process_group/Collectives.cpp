#include "Collectives.hpp"
#include "General.hpp"

#include <vector>

using namespace thd;

int THDGetRank() {
  return dataChannel->getRank();
}

int THDGetNumProcesses() {
  return dataChannel->getNumProcesses();
}

void THDAllReduce(THDTensorDescriptor* desc, THDReduceOp operation, THDGroup group) {
  dataChannel->allReduce(*desc, operation, group);
}

void THDReduce(THDTensorDescriptor* desc, THDReduceOp operation,
               int dst_rank, THDGroup group) {
  dataChannel->reduce(*desc, operation, dst_rank, group);
}

void THDBroadcast(THDTensorDescriptor* desc, int src_rank, THDGroup group) {
  dataChannel->broadcast(*desc, src_rank, group);
}

THDRequest* THDIsend(THDTensorDescriptor* desc, int dst_rank) {
  return dataChannel->isend(*desc, dst_rank);
}

THDRequest* THDIrecv(THDTensorDescriptor* desc, int src_rank) {
  return dataChannel->ireceive(*desc, src_rank);
}

void THDSend(THDTensorDescriptor* desc, int dst_rank) {
  dataChannel->send(*desc, dst_rank);
}

void THDRecvAnySource(THDTensorDescriptor* desc) {
  dataChannel->receive(*desc);
}

void THDRecv(THDTensorDescriptor* desc, int src_rank) {
  dataChannel->receive(*desc, src_rank);
}

void THDAllGather(THDTensorDescriptor** output, size_t len,
                  THDTensorDescriptor* input, THDGroup group) {
  std::vector<thpp::Tensor*> v_output(output, output + len);
  dataChannel->allGather(v_output, *input, group);
}

void THDGatherSend(THDTensorDescriptor* input, int dst_rank, THDGroup group) {
  std::vector<thpp::Tensor*> v_output;
  dataChannel->gather(v_output, *input, dst_rank, group);
}

void THDGatherRecv(THDTensorDescriptor** output, size_t len,
                   THDTensorDescriptor* input, THDGroup group) {
  std::vector<thpp::Tensor*> v_output(output, output + len);
  dataChannel->gather(v_output, *input, dataChannel->getRank(), group);
}

void THDScatterSend(THDTensorDescriptor** input, size_t len,
                    THDTensorDescriptor* output, THDGroup group) {
  std::vector<thpp::Tensor*> v_input(input, input + len);
  dataChannel->scatter(v_input, *output, dataChannel->getRank(), group);
}

void THDScatterRecv(THDTensorDescriptor* output, int src_rank, THDGroup group) {
  std::vector<thpp::Tensor*> v_input;
  dataChannel->scatter(v_input, *output, src_rank, group);
}

void THDBarrier(THDGroup group) {
  dataChannel->barrier(group);
}

THDGroup THDNewGroup(const int *ranks, size_t len) {
  std::vector<int> v_ranks(ranks, ranks + len);
  return dataChannel->newGroup(v_ranks);
}

bool THDRequest_isCompleted(THDRequest* request) {
  return request->isCompleted();
}

void THDRequest_wait(THDRequest* request) {
  request->wait();
}
