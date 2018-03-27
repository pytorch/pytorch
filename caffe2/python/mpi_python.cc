#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

namespace py = pybind11;

PYBIND11_MODULE(mpi_utils, m) {
  m.doc() = "MPI helper functions";
  m.def(
      "SetupPeers",
      &MPISetupPeers,
      py::arg("replicas"),
      py::arg("role"),
      py::arg("job_path"));
  m.def("CommSize", [] {
    auto comm = GlobalMPIComm();
    return MPICommSize(comm);
  });
  m.def("CommRank", [] {
    auto comm = GlobalMPIComm();
    return MPICommRank(comm);
  });
  m.def("Finalize", [] {
    // NOTE(pietern): Doesn't seem to work when calling it
    // from Python. It ends up calling pthread_join on a
    // thread that doesn't exit. For now, running mpirun
    // with `-quiet` and skipping the finalize call.
    MPI_Finalize();
  });
  m.def("Broadcast", [](py::bytes in) -> py::bytes {
    std::string str = in;
    auto comm = GlobalMPIComm();
    auto length = str.length();
    MPI_Bcast(&length, sizeof(length), MPI_CHAR, 0, comm);
    auto ptr = caffe2::make_unique<char[]>(length);
    if (MPICommRank(comm) == 0) {
      memcpy(ptr.get(), str.data(), str.length());
    }
    MPI_Bcast(ptr.get(), length, MPI_CHAR, 0, comm);
    return std::string(ptr.get(), length);
  });
}

} // namespace caffe2
