#include <c10/core/CPUAllocator.h>
#include <c10/util/typeid.h>
#include <c10/core/DeviceType.h>

// TODO: rename flags to C10
C10_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage");

C10_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    false,
    "If set, do memory zerofilling when allocating on CPU");

C10_DEFINE_bool(
    caffe2_cpu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on CPU");

namespace c10 {

void memset_junk(void* data, size_t num) {
  // This garbage pattern is NaN when interpreted as floating point values,
  // or as very large integer values.
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
  int32_t int64_count = num / sizeof(kJunkPattern64);
  int32_t remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  for (int i = 0; i < int64_count; i++) {
    data_i64[i] = kJunkPattern64;
  }
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

void* alloc_cpu(size_t nbytes) {
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
      "DefaultCPUAllocator: not enough memory: you tried to allocate %dGB. Buy new RAM!",
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

// A virtual struct that is used to report C10's memory allocation and
// deallocation status
class C10_API MemoryAllocationReporter {
 public:
  MemoryAllocationReporter() : allocated_(0) {}
  void New(void* ptr, size_t nbytes);
  void Delete(void* ptr);

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_;
};

struct C10_API DefaultCPUAllocator final : at::Allocator {
  DefaultCPUAllocator() {}
  ~DefaultCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = alloc_cpu(nbytes);
    if (FLAGS_caffe2_report_cpu_memory_usage && nbytes > 0) {
      getMemoryAllocationReporter().New(data, nbytes);
      return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
    }
    return {data, data, &Delete, at::Device(at::DeviceType::CPU)};
  }

#ifdef _MSC_VER
  static void Delete(void* data) {
    _aligned_free(data);
  }
#else
  static void Delete(void* data) {
    free(data);
  }
#endif

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    getMemoryAllocationReporter().Delete(ptr);
    Delete(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      return &ReportAndDelete;
    }
    return &Delete;
  }

 protected:
  static MemoryAllocationReporter& getMemoryAllocationReporter() {
    static MemoryAllocationReporter reporter_;
    return reporter_;
  }

};

void NoDelete(void*) {}

at::Allocator* GetCPUAllocator() {
  return GetAllocator(DeviceType::CPU);
}

void SetCPUAllocator(at::Allocator* alloc) {
  SetAllocator(DeviceType::CPU, alloc);
}

// Global default CPU Allocator
static DefaultCPUAllocator g_cpu_alloc;

at::Allocator* GetDefaultCPUAllocator() {
  return &g_cpu_alloc;
}

REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_alloc);

void MemoryAllocationReporter::New(void* ptr, size_t nbytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  size_table_[ptr] = nbytes;
  allocated_ += nbytes;
  LOG(INFO) << "C10 alloc " << nbytes << " bytes, total alloc " << allocated_
            << " bytes.";
}

void MemoryAllocationReporter::Delete(void* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = size_table_.find(ptr);
  CHECK(it != size_table_.end());
  allocated_ -= it->second;
  LOG(INFO) << "C10 deleted " << it->second << " bytes, total alloc "
            << allocated_ << " bytes.";
  size_table_.erase(it);
}

} // namespace c10
