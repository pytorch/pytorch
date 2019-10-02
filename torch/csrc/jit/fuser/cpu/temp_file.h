#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/disallow_copy.h>

#ifdef _WIN32
#include <Windows.h>
#include <io.h>
#include <stdio.h>
#include <fcntl.h>
#include <random>
#include <process.h>
#include <WinError.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

#ifdef _MSC_VER
int mkstemps(char* tmpl, int suffix_len) {
  int len;
  char* name;
  int fd = -1;
  int save_errno = errno;

  len = strlen(tmpl);
  if (len < 6 + suffix_len ||
      strncmp(&tmpl[len - 6 - suffix_len], "XXXXXX", 6)) {
    return -1;
  }

  name = &tmpl[len - 6 - suffix_len];

  std::random_device rd;
  do {
    for (unsigned i = 0; i < 6; ++i) {
        name[i] = "abcdefghijklmnopqrstuvwxyz0123456789"[rd() % 36];
    }

    fd = _open(tmpl, _O_RDWR | _O_CREAT | _O_EXCL, _S_IWRITE | _S_IREAD);
  } while (errno == EEXIST);

  if (fd >= 0) {
    errno = save_errno;
    return fd;
  } else {
    return -1;
  }
}
#endif

struct TempFile {
  TH_DISALLOW_COPY_AND_ASSIGN(TempFile);

  TempFile(const std::string& t, int suffix) {
    // mkstemps edits its first argument in places
    // so we make a copy of the string here, including null terminator
    std::vector<char> tt(t.c_str(), t.c_str() + t.size() + 1);
    int fd = mkstemps(tt.data(), suffix);
    AT_ASSERT(fd != -1);
    #ifdef _MSC_VER
    file_ = _fdopen(fd, "r+");
    #else
    file_ = fdopen(fd, "r+");
    #endif

    // - 1 becuase tt.size() includes the null terminator,
    // but std::string does not expect one
    name_ = std::string(tt.begin(), tt.end() - 1);
  }

  const std::string& name() const {
    return name_;
  }

  void sync() {
    fflush(file_);
  }

  void write(const std::string& str) {
    size_t result = fwrite(str.c_str(), 1, str.size(), file_);
    AT_ASSERT(str.size() == result);
  }

#ifdef _MSC_VER
  void close() {
    if (file_ != nullptr) {
      fclose(file_);
    }
    file_ = nullptr;
  }
#endif

  FILE* file() {
    return file_;
  }

  ~TempFile() {
#ifdef _MSC_VER
    if (file_ != nullptr) {
      fclose(file_);
    }
    if (!name_.empty() && _access(name_.c_str(), 0) != -1) {
      _unlink(name_.c_str());
    }
#else
    if (file_ != nullptr) {
      // unlink first to ensure another mkstemps doesn't
      // race between close and unlink
      unlink(name_.c_str());
      fclose(file_);
    }
#endif
  }

 private:
  FILE* file_ = nullptr;
  std::string name_;
};

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
