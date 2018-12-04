#ifndef THC_ALLOCATOR_INC
#define THC_ALLOCATOR_INC

#include "THCGeneral.h"

// IPC doesn't support (re)allocation

#ifdef __cplusplus
class CAFFE2_API THCIpcDeleter {
 public:
  THCIpcDeleter(std::shared_ptr<void> basePtr, int device);
  ~THCIpcDeleter();
  static at::DataPtr makeDataPtr(std::shared_ptr<void> basePtr, void* data, int device);
private:
  std::shared_ptr<void> basePtr_;
  int device_;
};
#endif

#endif
