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


CAFFE2_DEFINE_string(plan, "", "The given path to the plan protobuffer.");
CAFFE2_DEFINE_string(role, "", "server | client");
CAFFE2_DEFINE_int(
    replicas,
    2,
    "The total number of replicas (clients + server) to wait for");
CAFFE2_DEFINE_string(job_path, "", "The path to write to");

// The Borg routine: given
//   (1) a (quiesced) intra-communicator with one or more members, and
//   (2) a (quiesced) inter-communicator with exactly two members, one
//       of which is rank zero of the intra-communicator, and
//       the other of which is an unrelated spawned rank,
// return a new intra-communicator which is the union of both inputs.
//
// This is a collective operation. All ranks of the intra-
// communicator, and the remote rank of the inter-communicator, must
// call this routine. Ranks that are members of the intra-comm must
// supply the proper value for the "intra" argument, and MPI_COMM_NULL
// for the "inter" argument. The remote inter-comm rank must supply
// MPI_COMM_NULL for the "intra" argument, and the proper value for
// the "inter" argument. Rank zero (only) of the intra-comm must
// supply proper values for both arguments.
//
// N.B. It would make a certain amount of sense to split this into
// separate routines for the intra-communicator processes and the
// remote inter-communicator process. The reason we don't do that is
// that, despite the relatively few lines of code, what's going on
// here is really pretty complicated, and requires close coordination
// of the participating processes. Putting all the code for all the
// processes into this one routine makes it easier to be sure
// everything "lines up" properly.
MPI_Comm assimilateComm(MPI_Comm intra, MPI_Comm inter) {
  MPI_Comm peer = MPI_COMM_NULL;
  MPI_Comm newInterComm = MPI_COMM_NULL;
  MPI_Comm newIntraComm = MPI_COMM_NULL;

  // The spawned rank will be the "high" rank in the new intra-comm
  int high = (MPI_COMM_NULL == intra) ? 1 : 0;

  // If this is one of the (two) ranks in the inter-comm,
  // create a new intra-comm from the inter-comm
  if (MPI_COMM_NULL != inter) {
    MPI_CHECK(MPI_Intercomm_merge(inter, high, &peer));
  } else {
    peer = MPI_COMM_NULL;
  }

  // Create a new inter-comm between the pre-existing intra-comm
  // (all of it, not only rank zero), and the remote (spawned) rank,
  // using the just-created intra-comm as the peer communicator.
  int tag = 12345;
  if (MPI_COMM_NULL != intra) {
    // This task is a member of the pre-existing intra-comm
    MPI_CHECK(MPI_Intercomm_create(intra, 0, peer, 1, tag, &newInterComm));
  } else {
    // This is the remote (spawned) task
    MPI_CHECK(
        MPI_Intercomm_create(MPI_COMM_SELF, 0, peer, 0, tag, &newInterComm));
  }

  // Now convert this inter-comm into an intra-comm
  MPI_CHECK(MPI_Intercomm_merge(newInterComm, high, &newIntraComm));

  // Clean up the intermediaries
  if (MPI_COMM_NULL != peer) {
    MPI_CHECK(MPI_Comm_free(&peer));
  }
  MPI_CHECK(MPI_Comm_free(&newInterComm));

  // Delete the original intra-comm
  if (MPI_COMM_NULL != intra && MPI_COMM_WORLD != intra) {
    MPI_Comm_free(&intra);
  }

  // Return the new intra-comm
  return newIntraComm;
}

MPI_Comm connect() {
  // TODO: atomic
  std::string port_name;
  while (!caffe2::ReadStringFromFile(caffe2::FLAGS_job_path.c_str(),
                                     &port_name)) {
    CAFFE_LOG_INFO << "Failed reading job path";
    // Busy wait
    std::this_thread::sleep_for(std::chrono::seconds(1));
    continue;
  }
  // There's a (tiny) race between the client reading the file and the
  // server transitioning to MPI_Comm_accept.
  std::this_thread::sleep_for(std::chrono::seconds(1));
  MPI_Comm icomm;
  CAFFE_LOG_INFO << "Connecting to port name: " << port_name;
  MPI_CHECK(MPI_Comm_connect(
      const_cast<char*>(port_name.c_str()), MPI_INFO_NULL, 0,
      caffe2::MPIComm(), &icomm));
  CAFFE_LOG_INFO << "Connected";
  return icomm;
}

MPI_Comm accept() {
  std::string port_name;
  CAFFE_CHECK(caffe2::ReadStringFromFile(
      caffe2::FLAGS_job_path.c_str(), &port_name));
  MPI_Comm icomm;
  CAFFE_LOG_INFO << "Accepting a client on port name: " << port_name;
  MPI_CHECK(MPI_Comm_accept(
      const_cast<char*>(port_name.c_str()), MPI_INFO_NULL, 0,
      MPI_COMM_SELF, &icomm));
  CAFFE_LOG_INFO << "Finished accepting";
  return icomm;
}

void registerServer() {
  char port_name[MPI_MAX_PORT_NAME];
  MPI_CHECK(MPI_Open_port(MPI_INFO_NULL, port_name));
  CAFFE_LOG_INFO << "Port name: " << port_name;
  // TODO: atomic
  CAFFE_CHECK(caffe2::WriteStringToFile(
      std::string(port_name), caffe2::FLAGS_job_path.c_str()));
  CAFFE_LOG_INFO << "Wrote to file: " << caffe2::FLAGS_job_path;
}

size_t MPICommSize(MPI_Comm comm) {
  int comm_size;
  MPI_CHECK(MPI_Comm_size(comm, &comm_size));
  return comm_size;
}

int main(int argc, char** argv) {
  caffe2::SetUsageMessage("Runs a caffe2 plan that has MPI operators in it.");
  int mpi_ret;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_ret));
  if (mpi_ret != MPI_THREAD_MULTIPLE && mpi_ret != MPI_THREAD_SERIALIZED) {
    std::cerr << "Caffe2 MPI requires the underlying MPI to support the "
                 "MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE mode.\n";
    return 1;
  }

  caffe2::GlobalInit(&argc, &argv);

  // Initialize client/server
  CAFFE_CHECK(caffe2::FLAGS_role == "server" || caffe2::FLAGS_role == "client");
  if (caffe2::FLAGS_role == "server") {
    CAFFE_LOG_INFO << "Registering server";
    registerServer();
    CAFFE_LOG_INFO << "Registered server";
  }

  if (caffe2::FLAGS_role == "client") {
    CAFFE_LOG_INFO << "Client connecting";
    CAFFE_LOG_INFO << "Initial comm size: " << MPICommSize(caffe2::MPIComm());
    MPI_Comm icomm = connect();
    CAFFE_LOG_INFO << "Connected";
    CAFFE_LOG_INFO << "After connect size: " << MPICommSize(caffe2::MPIComm());
    CAFFE_LOG_INFO << "New comm size: " << MPICommSize(icomm);
    MPI_Comm newIntraComm = assimilateComm(MPI_COMM_NULL, icomm);
    caffe2::SetMPIComm(newIntraComm);
  }

  while (caffe2::MPISize() < caffe2::FLAGS_replicas) {
    CAFFE_LOG_INFO << "Still setting up, current known instances: "
              << caffe2::MPISize();

    if (caffe2::FLAGS_role == "server") {
      CAFFE_LOG_INFO << "Server Accepting";
      MPI_Comm icomm = accept();
      CAFFE_LOG_INFO << "Accepted";
      CAFFE_LOG_INFO << "After accept size: " << MPICommSize(caffe2::MPIComm());
      CAFFE_LOG_INFO << "New comm size: " << MPICommSize(icomm);
      MPI_Comm newIntraComm = assimilateComm(caffe2::MPIComm(), icomm);
      CAFFE_LOG_INFO << "Server assimilated, size: "
                     << MPICommSize(newIntraComm);
      caffe2::SetMPIComm(newIntraComm);
    } else {
      CAFFE_LOG_INFO << "Client assimilating";
      MPI_Comm newIntraComm = assimilateComm(caffe2::MPIComm(), MPI_COMM_NULL);
      CAFFE_LOG_INFO << "Client assimilated, size: "
                     << MPICommSize(newIntraComm);
      caffe2::SetMPIComm(newIntraComm);
    }
  }

  MPI_Barrier(caffe2::MPIComm());

  CAFFE_LOG_INFO << "Running with a communicator of size: "
                 << caffe2::MPISize();
  CAFFE_LOG_INFO << "Loading plan: " << caffe2::FLAGS_plan;
  caffe2::PlanDef plan_def;
  CAFFE_CHECK(ReadProtoFromFile(caffe2::FLAGS_plan, &plan_def));
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  workspace->RunPlan(plan_def);

  // This is to allow us to use memory leak checks.
  google::protobuf::ShutdownProtobufLibrary();
  MPI_Finalize();
  return 0;
}
