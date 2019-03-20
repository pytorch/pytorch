#include <c10/core/QuantizedAllocator.h>
#include <c10/util/typeid.h>
#include <c10/core/DeviceType.h>

namespace c10 {

void* alloc_quantized(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  // We might have clowny upstream code that tries to alloc a negative number
  // of bytes. Let's catch it early.
  CAFFE_ENFORCE(
    ((ptrdiff_t)nbytes) >= 0,
    "alloc_cpu() seems to have been called with negative number: ", nbytes);

  void* data;
#ifdef __ANDROID__
  data = memalign(gAlignment, nbytes);
#elif defined(_MSC_VER)
  data = _aligned_malloc(nbytes, gAlignment);
#else
  CAFFE_ENFORCE_EQ(posix_memalign(&data, gAlignment, nbytes), 0);
#endif

  CAFFE_ENFORCE(
      data,
      "DefaultQuantizedAllocator: not enough memory: you tried to allocate %dGB. Buy new RAM!",
      nbytes / 1073741824);

  // move data to a thread's NUMA node
  NUMAMove(data, nbytes, GetCurrentNUMANode());
  CHECK(
      !FLAGS_caffe2_cpu_allocator_do_zero_fill ||
      !FLAGS_caffe2_cpu_allocator_do_junk_fill)
    << "Cannot request both zero-fill and junk-fill at the same time";
  if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
    memset(data, 0, nbytes);
  } else if (FLAGS_caffe2_cpu_allocator_do_junk_fill) {
    memset_junk(data, nbytes);
  }

  return data;
}

void free_quantized(void* data) {
#ifdef _MSC_VER
  _aligned_free(data);
#else
  free(data);
#endif
}

struct C10_API DefaultQuantizedAllocator final : at::Allocator {
  DefaultQuantizedAllocator() {}
  ~DefaultQuantizedAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = alloc_quantized(nbytes);
    std::cout << "data " << data << std::endl;
    return {data, data, &free_quantized, at::Device(at::DeviceType::QUANTIZED)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &free_quantized;
  }

};

at::Allocator* GetQuantizedAllocator() {
  return GetAllocator(DeviceType::QUANTIZED);
}

void SetQuantizedAllocator(at::Allocator* alloc) {
  SetAllocator(DeviceType::QUANTIZED, alloc);
}

// Global default Quantized Allocator
static DefaultQuantizedAllocator g_quantized_alloc;

at::Allocator* GetDefaultQuantizedAllocator() {
  return &g_quantized_alloc;
}

REGISTER_ALLOCATOR(DeviceType::QUANTIZED, &g_quantized_alloc);

} // namespace c10
