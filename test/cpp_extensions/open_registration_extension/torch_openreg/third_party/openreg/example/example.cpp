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

void add_kernel(float* out, float* a, float* b, int num) {
  for (int i = 0; i < num; ++i) {
    out[i] = a[i] + b[i];
  }
}

int main() {
  int device_count = 0;
  orGetDeviceCount(&device_count);

  std::cout << "Current environment have " << device_count << " devices"
            << std::endl;

  orSetDevice(0);
  int current_device = -1;
  orGetDevice(&current_device);

  std::cout << "Current is " << current_device << " device" << std::endl;

  constexpr int num = 50000;
  constexpr size_t size = num * sizeof(float);

  std::vector<float> host_a(num), host_b(num), host_out(num, 0.0f);
  std::iota(host_a.begin(), host_a.end(), 0.0f);
  for (int i = 0; i < num; ++i) {
    host_b[i] = 2.0f;
  }

  float *dev_a, *dev_b, *dev_out;
  orMalloc((void**)&dev_a, size);
  orMalloc((void**)&dev_b, size);
  orMalloc((void**)&dev_out, size);

  // There will be subsequent memory access operations, so memory protection
  // needs to be released
  MemoryGuard a{dev_a};
  MemoryGuard b{dev_b};
  MemoryGuard c{dev_out};

  orStream_t stream1, stream2;
  orEvent_t start_event, stop_event;

  orStreamCreate(&stream1);
  orStreamCreate(&stream2);
  orEventCreateWithFlags(&start_event, orEventEnableTiming);
  orEventCreateWithFlags(&stop_event, orEventEnableTiming);

  // Copy input from host to device
  orMemcpyAsync(dev_a, host_a.data(), size, orMemcpyHostToDevice, stream1);
  orMemcpyAsync(dev_b, host_b.data(), size, orMemcpyHostToDevice, stream1);

  // Submit compute kernel and two events those are used for calculating time.
  orEventRecord(start_event, stream1);
  orLaunchKernel(stream1, add_kernel, dev_out, dev_a, dev_b, num);
  orEventRecord(stop_event, stream1);

  // Synchronization between streams.
  orStreamWaitEvent(stream2, stop_event, 0);
  orMemcpyAsync(host_out.data(), dev_out, size, orMemcpyDeviceToHost, stream2);
  orStreamSynchronize(stream2);

  std::cout << "All tasks have been submitted." << std::endl;

  float elapsed_ms = 0.0f;
  orEventElapsedTime(&elapsed_ms, start_event, stop_event);
  std::cout << "Kernel execution time: " << elapsed_ms << " ms" << std::endl;

  bool success = true;
  for (int i = 0; i < num; ++i) {
    if (std::abs(host_out[i] - (host_a[i] + host_b[i])) > 1e-5) {
      std::cout << "Verification FAILED at index " << i << "! Expected "
                << (host_a[i] + host_b[i]) << ", got " << host_out[i]
                << std::endl;
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Verification PASSED!" << std::endl;
  }

  orFree(dev_a);
  orFree(dev_b);
  orFree(dev_out);

  orStreamDestroy(stream1);
  orStreamDestroy(stream2);

  orEventDestroy(start_event);
  orEventDestroy(stop_event);

  return 0;
}
