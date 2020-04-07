#include <cstdio>
#include <iostream>
#include <vector>

#include <ATen/native/vulkan/VulkanDebugUtils.h>

namespace at {
namespace native {
namespace vulkan {
namespace debug {

void vk_print(const char* m, const float* t, uint32_t rank, uint32_t* dims) {
  static const char* kFloatFormat = "%4.4f";
  std::cout << m;
  std::cout << " dims:(";
  for (uint32_t i = 0; i < rank; i++) {
    std::cout << dims[i] << " ";
  }
  std::cout << ")" << std::endl;

  if (rank == 0) {
    std::cout << *t;
  } else if (rank == 1 || rank == 2) {
    char fbuf[12];
    uint32_t rows = rank == 1 ? 1 : dims[0];
    uint32_t cols = rank == 1 ? dims[0] : dims[1];
    for (uint32_t i = 0; i < rows; ++i) {
      std::cout << "\n";
      for (uint32_t j = 0; j < cols; ++j) {
        sprintf(fbuf, kFloatFormat, t[i * cols + j]);
        std::cout << fbuf << " ";
      }
    }
  } else if (rank == 3) {
    uint32_t d0 = dims[0];
    uint32_t d12size = dims[1] * dims[2];
    for (uint32_t i = 0; i < d0; i++) {
      char s[80];
      sprintf(s, "[%d, *, *]", i);
      vk_print(s, t + i * d12size, 2, dims + 1);
    }
  } else if (rank == 4) {
    uint32_t d0 = dims[0];
    uint32_t d1 = dims[1];
    uint32_t d23size = dims[2] * dims[3];
    for (uint32_t i = 0; i < d0; ++i) {
      for (uint32_t j = 0; j < d1; ++j) {
        char s[80];
        sprintf(s, "[%d, %d, *, *]", i, j);
        vk_print(s, t + i * d1 * d23size + j * d23size, 2, dims + 2);
      }
    }
  }
  std::cout << std::endl;
}

void vk_print4d(
    const char* m,
    const float* data,
    uint32_t d0,
    uint32_t d1,
    uint32_t d2,
    uint32_t d3) {
  uint32_t dims[4] = {d0, d1, d2, d3};
  vk_print(m, data, 4, dims);
}

void vk_print1d(const char* m, const float* data, uint32_t d0) {
  uint32_t dims[1] = {d0};
  vk_print(m, data, 1, dims);
}

void vk_print2d(const char* m, const float* data, uint32_t d0, uint32_t d1) {
  uint32_t dims[2] = {d0, d1};
  vk_print(m, data, 2, dims);
}

void vk_print3d(
    const char* m,
    const float* data,
    uint32_t d0,
    uint32_t d1,
    uint32_t d2) {
  uint32_t dims[3] = {d0, d1, d2};
  vk_print(m, data, 3, dims);
}

void vk_print_tensor_data(const char* m, const float* data, IntArrayRef sizes) {
  auto dim = sizes.size();
  if (dim == 1) {
    vk_print1d(m, data, sizes[0]);
  } else if (dim == 2) {
    vk_print2d(m, data, sizes[0], sizes[1]);
  } else if (dim == 3) {
    vk_print3d(m, data, sizes[0], sizes[1], sizes[2]);
  } else if (dim == 4) {
    vk_print4d(m, data, sizes[0], sizes[1], sizes[2], sizes[3]);
  } else {
    assert(false);
  }
}

} // namespace debug
} // namespace vulkan
} // namespace native
} // namespace at
