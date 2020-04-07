#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>

//#define VULKAN_DEBUG_LOG_ENABLED

#define FLF __FILE__ << ":" << __LINE__ << " " << __FUNCTION__
#define FLPF __FILE__ << ":" << __LINE__ << " " << __PRETTY_FUNCTION__

#ifdef VULKAN_DEBUG_LOG_ENABLED
#define COUT_FLF std::cout << FLF << std::endl
#define COUT_FLPF std::cout << FLPF << std::endl
#else
#define COUT_FLF
#define COUT_FLPF
#endif

namespace at {
namespace native {
namespace vulkan {
namespace debug {

void vk_print(const char* m, const float* t, uint32_t rank, uint32_t* dims);

void vk_print4d(
    const char* m,
    const float* data,
    uint32_t d0,
    uint32_t d1,
    uint32_t d2,
    uint32_t d3);

void vk_print1d(const char* m, const float* data, uint32_t d0);

void vk_print2d(const char* m, const float* data, uint32_t d0, uint32_t d1);

void vk_print3d(
    const char* m,
    const float* data,
    uint32_t d0,
    uint32_t d1,
    uint32_t d2);

void vk_print_tensor_data(const char* m, const float* data, IntArrayRef sizes);

} // namespace debug
} // namespace vulkan
} // namespace native
} // namespace at
