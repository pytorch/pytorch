#ifndef THC_ALLOCATOR_INC
#define THC_ALLOCATOR_INC

#include <THC/THCGeneral.h>

// IPC doesn't support (re)allocation

class CAFFE2_API THCIpcDeleter {
 public:
  THCIpcDeleter(std::shared_ptr<void> basePtr);
  ~THCIpcDeleter();
  static at::DataPtr makeDataPtr(std::shared_ptr<void> basePtr, void* data);
private:
  std::shared_ptr<void> basePtr_;
};

#endif
