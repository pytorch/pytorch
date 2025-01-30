#include "caffe2/serialize/file_adapter.h"
#include <c10/util/Exception.h>
#include <cerrno>
#include <cstdio>
#include <string>
#include "caffe2/core/common.h"

namespace caffe2 {
namespace serialize {

FileAdapter::RAIIFile::RAIIFile(const std::string& file_name) {
  fp_ = fopen(file_name.c_str(), "rb");
  if (fp_ == nullptr) {
    auto old_errno = errno;
#if defined(_WIN32) && (defined(__MINGW32__) || defined(_MSC_VER))
    char buf[1024];
    buf[0] = '\0';
    char* error_msg = buf;
    strerror_s(buf, sizeof(buf), old_errno);
#else
    auto error_msg =
        std::system_category().default_error_condition(old_errno).message();
#endif
    TORCH_CHECK(
        false,
        "open file failed because of errno ",
        old_errno,
        " on fopen: ",
        error_msg,
        ", file path: ",
        file_name);
  }
}

FileAdapter::RAIIFile::~RAIIFile() {
  if (fp_ != nullptr) {
    fclose(fp_);
  }
}

// FileAdapter directly calls C file API.
FileAdapter::FileAdapter(const std::string& file_name) : file_(file_name) {
  const int fseek_ret = fseek(file_.fp_, 0L, SEEK_END);
  TORCH_CHECK(fseek_ret == 0, "fseek returned ", fseek_ret);
#if defined(_MSC_VER)
  const int64_t ftell_ret = _ftelli64(file_.fp_);
#else
  const off_t ftell_ret = ftello(file_.fp_);
#endif
  TORCH_CHECK(ftell_ret != -1L, "ftell returned ", ftell_ret);
  size_ = ftell_ret;
  rewind(file_.fp_);
}

size_t FileAdapter::size() const {
  return size_;
}

size_t FileAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  // Ensure that pos doesn't exceed size_.
  pos = std::min(pos, size_);
  // If pos doesn't exceed size_, then size_ - pos can never be negative (in
  // signed math) or since these are unsigned values, a very large value.
  // Clamp 'n' to the smaller of 'size_ - pos' and 'n' itself. i.e. if the
  // user requested to read beyond the end of the file, we clamp to just the
  // end of the file.
  n = std::min(static_cast<size_t>(size_ - pos), n);
#if defined(_MSC_VER)
  const int fseek_ret = _fseeki64(file_.fp_, pos, SEEK_SET);
#else
  const int fseek_ret = fseeko(file_.fp_, pos, SEEK_SET);
#endif
  TORCH_CHECK(
      fseek_ret == 0, "fseek returned ", fseek_ret, ", context: ", what);
  return fread(buf, 1, n, file_.fp_);
}

FileAdapter::~FileAdapter() = default;

} // namespace serialize
} // namespace caffe2
