#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

static std::mutex gCaffe2MPIMutex;

std::mutex& MPIMutex() {
  return gCaffe2MPIMutex;
}


static MPI_Comm gCaffe2MPIComm = MPI_COMM_WORLD;
MPI_Comm MPIComm() {
  return gCaffe2MPIComm;
}

void SetMPIComm(MPI_Comm new_mpi_comm) {
  gCaffe2MPIComm = new_mpi_comm;
}

size_t MPISize() {
  int comm_size;
  MPI_CHECK(MPI_Comm_size(caffe2::MPIComm(), &comm_size));
  return comm_size;
}

}  // namespace caffe2
