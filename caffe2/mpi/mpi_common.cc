#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

static std::mutex gCaffe2MPIMutex;

std::mutex& MPIMutex() {
  return gCaffe2MPIMutex;
}
static MPI_Comm gCaffe2MPIComm = MPI_COMM_WORLD;

MPI_Comm GlobalMPIComm() {
  return gCaffe2MPIComm;
}

void SetGlobalMPIComm(MPI_Comm new_comm) {
  if (gCaffe2MPIComm != MPI_COMM_WORLD) {
    MPI_Comm_free(&gCaffe2MPIComm);
  }
  gCaffe2MPIComm = new_comm;
}

int MPICommSize(MPI_Comm comm) {
  int comm_size;
  MPI_CHECK(MPI_Comm_size(comm, &comm_size));
  return comm_size;
}

int MPICommRank(MPI_Comm comm) {
  int comm_rank;
  MPI_CHECK(MPI_Comm_rank(comm, &comm_rank));
  return comm_rank;
}
}  // namespace caffe2
