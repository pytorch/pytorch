#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

static std::mutex gCaffe2MPIMutex;

std::mutex& MPIMutex() {
  return gCaffe2MPIMutex;
}

}  // namespace caffe2
