#ifndef CAFFE2_MPI_MPI_COMMON_H_
#define CAFFE2_MPI_MPI_COMMON_H_

namespace caffe2 {

inline void CheckInitializedMPI() {
  int flag;
  MPI_Initialized(&flag);
  CHECK(flag) << "MPI does not seem to have been initialized.";
}

template <typename T> class MPIDataTypeWrapper;

#define MPI_DATATYPE_WRAPPER(c_type, mpi_type)                                 \
  template<> class MPIDataTypeWrapper<c_type> {                                \
   public:                                                                     \
    inline static MPI_Datatype type() { return  mpi_type; }                    \
  };

MPI_DATATYPE_WRAPPER(float, MPI_FLOAT)
MPI_DATATYPE_WRAPPER(double, MPI_DOUBLE)
// Note(Yangqing): as necessary, add more specializations.

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_COMMON_H_
