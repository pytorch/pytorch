/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

namespace py = pybind11;

PYBIND11_PLUGIN(mpi_utils) {
  py::module m("mpi_utils", "MPI helper functions");
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
  return m.ptr();
}

} // namespace caffe2
