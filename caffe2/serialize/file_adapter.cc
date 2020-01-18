#include "caffe2/serialize/file_adapter.h"
#include <c10/util/Exception.h>
#include "caffe2/core/common.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace caffe2 {
namespace serialize {

FileAdapter::FileAdapter(const std::string& file_name)
: file_name_(file_name) {
  file_stream_.open(file_name, std::ifstream::in | std::ifstream::binary);
  if (!file_stream_) {
    AT_ERROR("open file failed, file path: ", file_name);
  }
  istream_adapter_ = std::make_unique<IStreamAdapter>(&file_stream_);
}

size_t FileAdapter::size() const {
  return istream_adapter_->size();
}

size_t FileAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  return istream_adapter_->read(pos, buf, n, what);
}


bool FileAdapter::canMMap() const {
  // TODO MMAP windows off
  return true;
}

struct MMapFile {
  MMapFile(const std::string& file_name) {
    fd_ = open(file_name.c_str(), O_RDONLY);
    if (fd_ == -1) {
      AT_ERROR("open file failed for mmap, file path: ", file_name);
    }
  }
  ~MMapFile() {
    if (fd_ != -1) {
      close(fd_);
    }
  }
  int fd_;
};


namespace {

struct Mapping {
  Mapping(std::shared_ptr<MMapFile> file, off_t offset, size_t entry_size)
  : file_(std::move(file)) {
    entry_size_ = entry_size;
    static const size_t page_size = sysconf(_SC_PAGE_SIZE);
    size_t remainder = offset % page_size;
    size_t offset_aligned = offset - remainder;
    allocation_size_ = remainder + entry_size;
    allocation_data_ = mmap(nullptr, allocation_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, file_->fd_, offset_aligned);
    AT_ASSERT(allocation_data_ != nullptr, "mmap failed");
    entry_data_ = (char*)allocation_data_ + remainder;
  }
  ~Mapping() {
    munmap(allocation_data_, allocation_size_);
  }
  void* entry_data_;
  size_t entry_size_;

  // because the offset into the file must be page aligned, the actually
  // allocation into the file will be bigger than what the offset is.
  void* allocation_data_;
  size_t allocation_size_;

  std::shared_ptr<MMapFile> file_;
};

}

c10::DataPtr FileAdapter::mmap(uint64_t pos, size_t n) {
  if (!mmap_file_) {
    mmap_file_ = std::make_shared<MMapFile>(file_name_);
  }
  Mapping* mapping = new Mapping(mmap_file_, pos, n);
  DataPtr result(mapping->entry_data_, mapping, [](void* ctx) {
    Mapping* mapping = static_cast<Mapping*>(ctx);
    delete mapping;
  }, DeviceType::CPU);
  return result;
}

FileAdapter::~FileAdapter() {}

} // namespace serialize
} // namespace caffe2
