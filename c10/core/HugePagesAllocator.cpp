#include "c10/core/HugePagesAllocator.h"
#include "caffe2/core/context.h"

#include <jemalloc/jemalloc.h>
#include <sys/mman.h>
#include <iostream>

template <typename T>
c10::optional<T> jemallctl_get( // NOLINT(readability-identifier-naming)
    const std::string& ctlName) {
  T rv;
  size_t sz = sizeof(T);
  if (auto ret = jemallctl(ctlName.c_str(), &rv, &sz, nullptr, 0)) {
    std::cerr << "jemallctl: unable to read " << ctlName << ": "
              << strerror(ret);
    return {};
  }
  return {rv};
}

template <typename T>
bool jemallctl_set( // NOLINT(readability-identifier-naming)
    const std::string& ctlName,
    T value) {
  if (auto ret =
          jemallctl(ctlName.c_str(), nullptr, nullptr, &value, sizeof(T))) {
    std::cerr << "jemallctl: unable to set " << ctlName << ": "
              << strerror(ret);
    return false;
  }
  return true;
}

int64_t time_since_epoch() {
  auto t = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             t.time_since_epoch())
      .count();
}

std::string get_stats() {
  // Update the statistics cached by mallctl.
  uint64_t epoch = 1;
  size_t sz;
  sz = sizeof(epoch);
  jemallctl("epoch", &epoch, &sz, &epoch, sz);

  // Get basic allocation statistics.  Take care to check for
  // errors, since --enable-stats must have been specified at
  // build time for these statistics to be available.
  size_t allocated, active, metadata, resident, mapped, retained;
  std::ostringstream out;
  sz = sizeof(size_t);
  if (jemallctl("stats.allocated", &allocated, &sz, NULL, 0) == 0 &&
      jemallctl("stats.active", &active, &sz, NULL, 0) == 0 &&
      jemallctl("stats.metadata", &metadata, &sz, NULL, 0) == 0 &&
      jemallctl("stats.resident", &resident, &sz, NULL, 0) == 0 &&
      jemallctl("stats.mapped", &mapped, &sz, NULL, 0) == 0 &&
      jemallctl("stats.retained", &retained, &sz, NULL, 0) == 0) {
    out << allocated << "," << active << "," << metadata << "," << resident
        << "," << mapped << "," << retained;
  }
  return out.str();
}

namespace c10 {

void* je_alloc_cpu(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  // We might have clowny upstream code that tries to alloc a negative number
  // of bytes. Let's catch it early.
  CAFFE_ENFORCE(
      ((ptrdiff_t)nbytes) >= 0,
      "alloc_cpu() seems to have been called with negative number: ",
      nbytes);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* data;
  int err = jeposix_memalign(&data, gAlignment, nbytes);
  if (err != 0) {
    CAFFE_THROW(
        "JEMallocCPUAllocator: can't allocate memory: you tried to allocate ",
        nbytes,
        " bytes. Error code ",
        err,
        " (",
        strerror(err),
        ")");
  }

  CAFFE_ENFORCE(
      data,
      "JEMallocCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");

  // move data to a thread's NUMA node
  CHECK(
      !FLAGS_caffe2_cpu_allocator_do_zero_fill ||
      !FLAGS_caffe2_cpu_allocator_do_junk_fill)
      << "Cannot request both zero-fill and junk-fill at the same time";
  if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
    memset(data, 0, nbytes);
  } else if (FLAGS_caffe2_cpu_allocator_do_junk_fill) {
    memset_junk(data, nbytes);
  }

  //  if (const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
  //    std::cout << "alloc," << time_since_epoch() << "," << data << ","
  //              << get_stats() << std::endl;
  //  }

  return data;
}

void je_free_cpu(void* data) {
  jefree(data);
  //  if (const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
  //    std::cout << "free," << time_since_epoch() << "," << data << ","
  //              << get_stats() << std::endl;
  //  }
}

struct JEMallocCPUAllocator final : at::Allocator {
  JEMallocCPUAllocator() {
    //    std::cout << "JEMallocCPUAllocator cons called\n";
  }
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = je_alloc_cpu(nbytes);
    profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    profiledCPUMemoryReporter().Delete(ptr);
    je_free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

static JEMallocCPUAllocator g_je_malloc_cpu_alloc;

bool installJEMallocCPUAllocator() {
  // Setting allocator globally with priority = 1, it will not
  // be overrridden unless another CPU allocator is set with a
  // higher priority value.
  c10::SetCPUAllocator(&g_je_malloc_cpu_alloc, 1 /* priority */);
  return true;
}

// install the allocator statically
static bool registry = installJEMallocCPUAllocator();

int str2int(const string& str) {
  std::stringstream ss(str);
  int num;
  if ((ss >> num).fail()) {
    CAFFE_THROW("couldn't parse num");
  }
  return num;
}

} // namespace c10