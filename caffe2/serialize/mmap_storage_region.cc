#include "caffe2/serialize/mmap_storage_region.h"

#ifndef _WIN32
#include <fcntl.h> // O_RDONLY
#include <sys/mman.h> // MAP_PRIVATE, PROT_READ, PROT_WRITE
#include <sys/stat.h> // stat
#include <unistd.h> // close
#else
#include <stdexcept>
#endif

namespace caffe2 {
namespace serialize {

MmapStorageRegion::MmapStorageRegion(const std::string& filename) {
#ifndef _WIN32
  int fd = open(filename.c_str(), O_RDONLY);
  TORCH_INTERNAL_ASSERT(
    fd != -1,
    "unable to open file <", filename, ">: ", strerror(errno), " (", errno, ")");
  struct stat st{};
  TORCH_INTERNAL_ASSERT(
    fstat(fd, &st) != -1,
    "unable to get file status for <", filename, ">: ", strerror(errno), " (", errno, ")");
  size = st.st_size;
  region = reinterpret_cast<uint8_t*>(
    mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)
  );
  TORCH_INTERNAL_ASSERT(
    region != MAP_FAILED,
    "unable to mmap ", size, " bytes from file <", filename, ">: ", strerror(errno), " (", errno, ")");
  close(fd);
#else
  throw std::runtime_error("MmapStorageRegion not supported on this platform");
#endif
}

MmapStorageRegion::~MmapStorageRegion() {
#ifndef _WIN32
  munmap(reinterpret_cast<void*>(region), size);
#else
  throw std::runtime_error("MmapStorageRegion not supported on this platform");
#endif
}

at::DataPtr MmapStorageRegion::getData(size_t offset, Device device) {
  c10::raw::intrusive_ptr::incref(this);
  void* address = reinterpret_cast<void*>(region + offset);
  void* ctx = reinterpret_cast<void*>(this);
  return at::DataPtr(address, ctx, deleter, device);
}

void MmapStorageRegion::deleter(void *ctx) {
  MmapStorageRegion* storage = (MmapStorageRegion*) ctx;
  c10::raw::intrusive_ptr::decref(storage);
}

bool MmapStorageRegion::isSupportedByPlatform() {
#ifndef _WIN32
  return true;
#else
  return false;
#endif
}

} // namespace serialize
} // namespace caffe2
