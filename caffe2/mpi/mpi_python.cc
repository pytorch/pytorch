#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/caffe2/mpi/mpi_common.h"

namespace caffe2 {

namespace py = pybind11;

PYBIND11_PLUGIN(mpi) {
  py::module m("mpi", "MPI helper functions");
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
  return m.ptr();
}

} // namespace caffe2
