// Runs a plan with mpi enabled without mpirun. This differs from run_plan_mpi
// in the sense that the common world is formed by joining during runtime,
// instead of being set up by mpirun.
//
// This util assumes that you have a common path (like NFS) that multiple
// instances can read from.

#include <mpi.h>

#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

using caffe2::MPICommSize;
using caffe2::GlobalMPIComm;

CAFFE2_DEFINE_string(plan, "", "The given path to the plan protobuffer.");
CAFFE2_DEFINE_string(role, "", "server | client");
CAFFE2_DEFINE_int(
    replicas,
    2,
    "The total number of replicas (clients + server) to wait for");
CAFFE2_DEFINE_string(job_path, "", "The path to write to");

namespace {

// RAAI for MPI so that we always run MPI_Finalize when exiting.
class MPIContext {
 public:
  MPIContext(int argc, char** argv) {
    int mpi_ret;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_ret));
    if (mpi_ret != MPI_THREAD_MULTIPLE && mpi_ret != MPI_THREAD_SERIALIZED) {
      throw std::runtime_error(
          "Caffe2 MPI requires the underlying MPI to support the "
          "MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE mode.");
    }
  }

  ~MPIContext() {
    MPI_Finalize();
  }
};
}

int main(int argc, char** argv) {
  MPIContext mpi_context(argc, argv);

  caffe2::SetUsageMessage("Runs a caffe2 plan that has MPI operators in it.");
  caffe2::GlobalInit(&argc, &argv);
  caffe2::MPISetupPeers(
      caffe2::FLAGS_replicas, caffe2::FLAGS_role, caffe2::FLAGS_job_path);

  // Only check if plan is specified AFTER MPI setup such that we can test
  // whether or not MPI setup works without having a plan to run.
  if (FLAGS_plan == "") {
    std::cerr << "No plan defined! Exiting...\n";
    return 0;
  }

  caffe2::PlanDef plan_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_plan, &plan_def));
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  workspace->RunPlan(plan_def);

  // This is to allow us to use memory leak checks.
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
