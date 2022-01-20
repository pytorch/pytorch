#include "caffe2/mpi/mpi_common.h"

#include <thread>

#include <c10/util/typeid.h>
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(MPICommonWorldWrapper);

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

/**
 * Helper function used to setup MPI intercommunicator.
 */
static MPI_Comm AssimilateComm(MPI_Comm intra, MPI_Comm inter) {
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
  if (MPI_COMM_NULL != intra && MPI_COMM_WORLD != intra &&
      GlobalMPIComm() != intra) {
    MPI_CHECK(MPI_Comm_free(&intra));
  }

  // Return the new intra-comm
  return newIntraComm;
}

void MPISetupPeers(
    const int replicas,
    const string& role,
    const string& job_path) {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    int mpi_ret;
    MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_ret));
    if (mpi_ret != MPI_THREAD_MULTIPLE && mpi_ret != MPI_THREAD_SERIALIZED) {
      LOG(FATAL) << "This test requires the underlying MPI to support the "
                 << "MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE mode.";
      return;
    }
  }

  if (MPICommSize(MPI_COMM_WORLD) != 1) {
    LOG(ERROR) << "MPI_COMM_WORLD size is not 1: did you already run "
                  "MPISetupPeers? Note that if you execute your program with "
                  "mpirun to launch multiple local processes, you should not "
                  "call MPISetupPeers.";
    return;
  }

  if (role == "server") {
    // Open a port to accept connections.
    char port_name[MPI_MAX_PORT_NAME] = {'\0'};
    MPI_CHECK(MPI_Open_port(MPI_INFO_NULL, port_name));
    VLOG(1) << "MPI server: port: " << port_name;

    // Writes the port name to the file.
    CHECK(WriteStringToFile(std::string(port_name), job_path.c_str()));
    VLOG(1) << "MPI server: wrote to file: " << job_path;

    int comm_size = MPICommSize(GlobalMPIComm());
    while (comm_size < replicas) {
      MPI_Comm icomm;
      VLOG(1) << "MPI server: waiting for client "
              << "(" << comm_size << "/" << replicas << " have connected)";
      MPI_CHECK(
          MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &icomm));
      VLOG(1) << "MPI server: accepted client";
      MPI_Comm new_intra_comm = AssimilateComm(GlobalMPIComm(), icomm);
      SetGlobalMPIComm(new_intra_comm);
      comm_size = MPICommSize(new_intra_comm);
    }
  } else {
    // Opens the job path file to obtain server address.
    std::string port_name;
    while (!ReadStringFromFile(job_path.c_str(), &port_name) ||
           port_name.length() == 0) {
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Connect to server.
    MPI_Comm icomm;
    VLOG(1) << "MPI client: connecting to port: " << port_name;
    MPI_CHECK(MPI_Comm_connect(
        const_cast<char*>(port_name.c_str()),
        MPI_INFO_NULL,
        0,
        GlobalMPIComm(),
        &icomm));

    VLOG(1) << "MPI client: connected";

    // Join the server's reference intracommunicator.
    MPI_Comm new_intra_comm = AssimilateComm(MPI_COMM_NULL, icomm);
    SetGlobalMPIComm(new_intra_comm);

    // Let other clients join the intracommunicator we're now a part of.
    while (MPICommSize(GlobalMPIComm()) < replicas) {
      MPI_Comm comm = AssimilateComm(GlobalMPIComm(), MPI_COMM_NULL);
      SetGlobalMPIComm(comm);
    }
  }

  // After all peers have assimilated, do a barrier.
  MPI_Barrier(GlobalMPIComm());
  VLOG(1) << "MPI using a communicator of size: "
          << MPICommSize(GlobalMPIComm());
}

} // namespace caffe2
