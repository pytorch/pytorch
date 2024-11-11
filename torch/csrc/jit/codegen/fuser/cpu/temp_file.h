#pragma once

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>

#ifdef _WIN32
#include <WinError.h>
#include <c10/util/Unicode.h>
#include <c10/util/win32-headers.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdio.h>
#include <sys/stat.h>
#include <random>
#else
#include <unistd.h>
#endif

#include <string>
#include <vector>

namespace torch::jit::fuser::cpu {

#ifdef _MSC_VER
int wmkstemps(wchar_t* tmpl, int suffix_len) {
  int len;
  wchar_t* name;
  int fd = -1;
  int save_errno = errno;

  len = wcslen(tmpl);
  if (len < 6 + suffix_len ||
      wcsncmp(&tmpl[len - 6 - suffix_len], L"XXXXXX", 6)) {
    return -1;
  }

  name = &tmpl[len - 6 - suffix_len];

  std::random_device rd;
  do {
    for (unsigned i = 0; i < 6; ++i) {
      name[i] = "abcdefghijklmnopqrstuvwxyz0123456789"[rd() % 36];
    }

    fd = _wopen(tmpl, _O_RDWR | _O_CREAT | _O_EXCL, _S_IWRITE | _S_IREAD);
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
  AT_DISALLOW_COPY_AND_ASSIGN(TempFile);

  TempFile(const std::string& t, int suffix) {
#ifdef _MSC_VER
    auto wt = c10::u8u16(t);
    std::vector<wchar_t> tt(wt.c_str(), wt.c_str() + wt.size() + 1);
    int fd = wmkstemps(tt.data(), suffix);
    AT_ASSERT(fd != -1);
    file_ = _wfdopen(fd, L"r+");
    auto wname = std::wstring(tt.begin(), tt.end() - 1);
    name_ = c10::u16u8(wname);
#else
    // mkstemps edits its first argument in places
    // so we make a copy of the string here, including null terminator
    std::vector<char> tt(t.c_str(), t.c_str() + t.size() + 1);
    int fd = mkstemps(tt.data(), suffix);
    AT_ASSERT(fd != -1);
    file_ = fdopen(fd, "r+");
    // - 1 because tt.size() includes the null terminator,
    // but std::string does not expect one
    name_ = std::string(tt.begin(), tt.end() - 1);
#endif
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
    auto wname = c10::u8u16(name_);
    if (!wname.empty() && _waccess(wname.c_str(), 0) != -1) {
      _wunlink(wname.c_str());
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

} // namespace torch::jit::fuser::cpu
