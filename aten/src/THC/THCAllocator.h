#ifndef THC_ALLOCATOR_INC
#define THC_ALLOCATOR_INC

#include "THCGeneral.h"

// IPC doesn't support (re)allocation

#ifdef __cplusplus
class CAFFE2_API THCIpcDeleter {
 public:
  THCIpcDeleter(void* data, int device) : data_(data), device_(device) {};
  ~THCIpcDeleter();
  static at::DataPtr makeDataPtr(void* data, int device);
private:
  void* data_;
  int device_;
};
#endif

#endif
