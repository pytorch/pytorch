
#include "../core/GLImageAllocator.h"
#include "../core/arm_neon_support.h"

template <typename T>
GLImageAllocator<T>* GLImageAllocator<T>::newGLImageAllocator() {
  return new GLImageAllocator<T>();
}

template GLImageAllocator<float16_t>* GLImageAllocator<float16_t>::newGLImageAllocator();
template GLImageAllocator<uint8_t>* GLImageAllocator<uint8_t>::newGLImageAllocator();
