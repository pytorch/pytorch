#pragma once

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>

#include <cstddef>
#include <string>

namespace c10 {

C10_API void* alloc_cpu(size_t nbytes);
C10_API void free_cpu(void* data);

C10_API bool is_mimalloc_enabled();
C10_API std::string get_mimalloc_stats_json();
C10_API void reset_mimalloc_stats();

C10_API void set_mimalloc_option(const std::string& name, int64_t value);
C10_API int64_t get_mimalloc_option(const std::string& name);

#if defined(__linux__) && !defined(__ANDROID__)
C10_API size_t c10_compute_alignment(size_t nbytes);
#endif

#ifdef USE_MIMALLOC_ON_MKL
namespace mi_malloc_wrapper {
C10_API void* c10_mi_malloc(size_t size);
C10_API void* c10_mi_calloc(size_t count, size_t size);
C10_API void* c10_mi_realloc(void* p, size_t newsize);
C10_API void* c10_mi_malloc_aligned(size_t size, size_t alignment);
C10_API void c10_mi_free(void* p);
} // namespace mi_malloc_wrapper
#endif

} // namespace c10
