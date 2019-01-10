#ifndef CAFFE2_MPI_MPI_COMMON_H_
#define CAFFE2_MPI_MPI_COMMON_H_

#include <mpi.h>
#include <mutex>

#include "caffe2/core/logging.h"

namespace caffe2 {

inline void CheckInitializedMPI() {
  int flag;
  MPI_Initialized(&flag);
  CAFFE_ENFORCE(flag, "MPI does not seem to have been initialized.");
}

template <typename T> class MPIDataTypeWrapper;

#define MPI_DATATYPE_WRAPPER(c_type, mpi_type)                                 \
  template<> class MPIDataTypeWrapper<c_type> {                                \
   public:                                                                     \
    inline static MPI_Datatype type() { return mpi_type; }                     \
  };

MPI_DATATYPE_WRAPPER(char, MPI_CHAR)
MPI_DATATYPE_WRAPPER(float, MPI_FLOAT)
MPI_DATATYPE_WRAPPER(double, MPI_DOUBLE)
// Note(Yangqing): as necessary, add more specializations.
#undef MPI_DATATYPE_WRAPPER

// For all Caffe MPI calls, we will wrap it inside an MPI mutex lock guard.
std::mutex& MPIMutex();

#define MPI_CHECK(condition)                                 \
  do {                                                       \
    std::lock_guard<std::mutex> guard(::caffe2::MPIMutex()); \
    int error = (condition);                                 \
    CAFFE_ENFORCE(                                           \
        error == MPI_SUCCESS,                                \
        "Caffe2 MPI Error at: ",                             \
        __FILE__,                                            \
        ":",                                                 \
        __LINE__,                                            \
        ": ",                                                \
        error);                                              \
  } while (0)

/**
 * @brief Gets the global MPI communicator used by Caffe2. In default, this
 * is MPI_COMM_WORLD unless you call SetGlobalMPIComm().
 */
MPI_Comm GlobalMPIComm();

/**
 * @brief Sets the global MPI communicator. Caffe2 takes over the ownership
 * of the passed in communicator.
 */
void SetGlobalMPIComm(MPI_Comm new_comm);

/**
 * @brief A helper function to return the size of the given communicator.
 */
int MPICommSize(MPI_Comm comm);

/**
 * @brief A helper function to return the rank of the given communicator.
 */
int MPICommRank(MPI_Comm comm);

/**
 * @brief A simple wrapper over an MPI common world.
 */
class MPICommonWorldWrapper {
 public:
  /**
   * @brief Creates a common world wrapper.
   *
   * The new common world is created by taking the existing communicator
   * passed in as src_comm, and splitting it using the color and the rank
   * specified. In default, we will split from Caffe2's global communicator,
   * and use color 0 as well as rank implicitly given by src_comm. As a result,
   * the default constructor basically creates a comm identical to the source
   * comm world.
   */
  explicit MPICommonWorldWrapper(
      MPI_Comm src_comm = MPI_COMM_NULL,
      int color = 0,
      int rank = -1) {
    if (src_comm == MPI_COMM_NULL) {
      src_comm = GlobalMPIComm();
    }
    if (rank == -1) {
      MPI_CHECK(MPI_Comm_rank(src_comm, &rank));
    }
    MPI_CHECK(MPI_Comm_split(src_comm, color, rank, &comm_));
    MPI_CHECK(MPI_Comm_size(comm_, &size_));
    MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
  }

  ~MPICommonWorldWrapper() {
    int ret;
    MPI_CHECK(MPI_Finalized(&ret));
    if (!ret) {
      MPI_Comm_free(&comm_);
    }
  }

  /**
   * @brief Returns the common world held by the wrapper.
   */
  inline MPI_Comm comm() const {
    return comm_;
  }
  /**
   * @brief Returns the size of the world.
   */
  inline int size() const {
    return size_;
  }
  /**
   * @brief Returns the rank of this process in the world.
   */
  inline int rank() const {
    return rank_;
  }

 private:
  MPI_Comm comm_;
  int size_;
  int rank_;
};

/**
 * A function used to perform peer setup so one does not need to use
 * mpirun / mpiexec to run the binary. Note that if you use mpirun or mpiexec
 * to set up the common world, do not use this function - MPI_Init would have
 * already set that up.
 *
 * This also assumes that you have a common path (like NFS) that multiple
 * instances can read from.
 *
 * Inputs:
 *   replicas (int): the number of replicas that mpi will run with.
 *   role (string): the role of this process, "server" or "client".
 *   job_path (string): a file name that the server will write its port into
 *       and the clients will read the server's port from.
 */
void MPISetupPeers(
    const int replicas,
    const string& role,
    const string& job_path);
}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_COMMON_H_
