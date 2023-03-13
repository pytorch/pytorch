#pragma once

#include <c10/macros/Macros.h>

#include <cstddef>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace c10 {

C10_API void* alloc_cpu(size_t nbytes);
C10_API void free_cpu(void* data);

} // namespace c10
