#ifndef CAFFE2_MPI_MPI_COMMON_H_
#define CAFFE2_MPI_MPI_COMMON_H_

#include <mutex>
#include <mpi.h>

#include "caffe2/core/logging.h"

namespace caffe2 {

inline void CheckInitializedMPI() {
  int flag;
  MPI_Initialized(&flag);
  CAFFE_CHECK(flag) << "MPI does not seem to have been initialized.";
}

template <typename T> class MPIDataTypeWrapper;

#define MPI_DATATYPE_WRAPPER(c_type, mpi_type)                                 \
  template<> class MPIDataTypeWrapper<c_type> {                                \
   public:                                                                     \
    inline static MPI_Datatype type() { return  mpi_type; }                    \
  };

MPI_DATATYPE_WRAPPER(char, MPI_CHAR)
MPI_DATATYPE_WRAPPER(float, MPI_FLOAT)
MPI_DATATYPE_WRAPPER(double, MPI_DOUBLE)
// Note(Yangqing): as necessary, add more specializations.
#undef MPI_DATATYPE_WRAPPER

// For all Caffe MPI calls, we will wrap it inside an MPI mutex lock guard.
std::mutex& MPIMutex();

#define MPI_CHECK(condition)                                                   \
  do {                                                                         \
    std::lock_guard<std::mutex> guard(::caffe2::MPIMutex());                   \
    int error = (condition);                                                   \
    CAFFE_CHECK_EQ(error, MPI_SUCCESS)                                         \
        << "Caffe2 MPI Error at: " << __FILE__ << ":" << __LINE__ << ": "      \
        << error;                                                              \
  } while (0)


}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_COMMON_H_
