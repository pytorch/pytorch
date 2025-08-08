#include "include/openreg.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

struct MemoryGuard {
  MemoryGuard(void* ptr) : ptr_(ptr) {
    orMemoryUnprotect(ptr_);
  }
  ~MemoryGuard() {
    orMemoryProtect(ptr_);
  }

 private:
  void* ptr_{};
};

void add_kernel(float* out, float* a, float* b, int n) {
  // Will remove this one when https://github.com/pytorch/pytorch/pull/159441 merged
  MemoryGuard go(out, 10);
  MemoryGuard ga(a, 11);
  MemoryGuard gb(b, 12);

  for (int i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

int main() {
  int device_count = 0;
  orGetDeviceCount(&device_count);

  orSetDevice(0);
  int current_device = -1;
  orGetDevice(&current_device);

  const int n = 1024 * 1024;
  const size_t size = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n, 0.0f);
  float* d_a, d_b, d_c;

  std::iota(h_a.begin(), h_a.end(), 0.0f);
  for (int i = 0; i < n; ++i) {
    h_b[i] = 2.0f;
  }

  orMalloc((void**)&d_a, size);
  orMalloc((void**)&d_b, size);
  orMalloc((void**)&d_c, size);

  MemoryGuard a{d_a, 1};
  MemoryGuard b{d_b, 2};
  MemoryGuard c{d_c, 3};

  orStream_t stream1, stream2;
  orEvent_t start_event, stop_event, stream2_event;

  orStreamCreate(&stream1);
  orStreamCreate(&stream2);
  orEventCreateWithFlags(&start_event, orEventDisableTiming);
  orEventCreateWithFlags(&stop_event, orEventDisableTiming);
  orEventCreateWithFlags(&stream2_event, orEventDisableTiming);

  std::cout << "\n--- Starting asynchronous operations ---" << std::endl;
  orMemcpyAsync(d_a, h_a.data(), size, orMemcpyHostToDevice, stream1);
  orMemcpyAsync(d_b, h_b.data(), size, orMemcpyHostToDevice, stream1);

  orEventRecord(start_event, stream1);

  orLaunchKernel(stream1, add_kernel, d_c, d_a, d_b, n);

  orEventRecord(stop_event, stream1);

  orStreamWaitEvent(stream2, stop_event, 0);

  orMemcpyAsync(h_c.data(), d_c, size, orMemcpyDeviceToHost, stream2);

  orEventRecord(stream2_event, stream2);

  std::cout << "All tasks have been submitted. Main thread is now waiting..."
            << std::endl;

  orEventSynchronize(stream2_event);

  float elapsed_ms = 0.0f;
  orEventElapsedTime(&elapsed_ms, start_event, stop_event);
  std::cout << "Kernel execution time: " << elapsed_ms << " ms" << std::endl;

  bool success = true;
  for (int i = 0; i < n; ++i) {
    if (std::abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
      std::cout << "Verification FAILED at index " << i << "! Expected "
                << (h_a[i] + h_b[i]) << ", got " << h_c[i] << std::endl;
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Verification PASSED!" << std::endl;
  }

  orFree(d_a);
  orFree(d_b);
  orFree(d_c);
  orStreamDestroy(stream1);
  orStreamDestroy(stream2);
  orEventDestroy(start_event);
  orEventDestroy(stop_event);
  orEventDestroy(stream2_event);
  std::cout << "All resources freed." << std::endl;

  return 0;
}
