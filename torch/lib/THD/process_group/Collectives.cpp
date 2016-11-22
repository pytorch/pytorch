#include "Collectives.hpp"
#include "General.hpp"

namespace thd {

int THDGetRank() {
  return dataChannel->getRank();
}

int THDGetNumProcesses() {
  return dataChannel->getNumProcesses();
}

void THDAllReduce(THDTensorDescriptor desc, THDReduceOp operation) {
  dataChannel->allReduce(*desc, operation);
}

void THDReduce(THDTensorDescriptor desc, THDReduceOp operation, int dst_rank) {
  dataChannel->reduce(*desc, operation, dst_rank);
}

void THDBroadcast(THDTensorDescriptor desc, int src_rank) {
  dataChannel->broadcast(*desc, src_rank);
}

void THDSend(THDTensorDescriptor desc, int dst_rank) {
  dataChannel->send(*desc, dst_rank);
}

void THDReceive(THDTensorDescriptor desc, int src_rank) {
  dataChannel->receive(*desc, src_rank);
}

} // namespace thd
