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

void THDSend(THDTensorDescriptor* desc, int dst_rank) {
  dataChannel->send(*desc, dst_rank);
}

void THDReceive(THDTensorDescriptor* desc, int src_rank) {
  dataChannel->receive(*desc, src_rank);
}

THDGroup THDNewGroup(const int *ranks, size_t len) {
  std::vector<int> v_ranks;
  for (size_t i = 0; i < len; ++i)
    v_ranks.push_back(ranks[i]);
  return dataChannel->newGroup(v_ranks);
}
