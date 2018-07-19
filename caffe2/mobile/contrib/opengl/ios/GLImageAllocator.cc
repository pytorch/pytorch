
#include "IOSGLImageAllocator.h"
#include <arm_neon.h>

template <typename T>
GLImageAllocator<T>* GLImageAllocator<T>::newGLImageAllocator() {
  return new IOSGLImageAllocator<T>();
}

template GLImageAllocator<float16_t>* GLImageAllocator<float16_t>::newGLImageAllocator();
template GLImageAllocator<uint8_t>* GLImageAllocator<uint8_t>::newGLImageAllocator();
