#ifndef THC_THRUST_ALLOCATOR_INC
#define THC_THRUST_ALLOCATOR_INC

#include <cstddef>
#include <THC/THCGeneral.h>

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class THCThrustAllocator {
 public:
  typedef char value_type;

  THCThrustAllocator(THCState* state)
      : state_(state) {
  }

  ~THCThrustAllocator() {
  }

  char* allocate(std::ptrdiff_t size) {
    return static_cast<char*>(THCudaMalloc(state_, size));
  }

  void deallocate(char* p, size_t size) {
    THCudaFree(state_, p);
  }

 private:
  THCState* state_;
};

#endif // THC_THRUST_ALLOCATOR_INC
