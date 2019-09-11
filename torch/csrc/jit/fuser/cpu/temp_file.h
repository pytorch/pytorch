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
/* mkstemps extracted from gcc/libiberty/mkstemps.c.  Copyright (C) 1991-2019
   Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.  */

static const char letters[] = "abcdefghijklmnopqrstuvwxyz0123456789";

/*
@deftypefn Replacement int mkstemps (char *@var{pattern}, int @var{suffix_len})
Generate a unique temporary file name from @var{pattern}.
@var{pattern} has the form:
@example
   @var{path}/ccXXXXXX@var{suffix}
@end example
@var{suffix_len} tells us how long @var{suffix} is (it can be zero
length).  The last six characters of @var{pattern} before @var{suffix}
must be @samp{XXXXXX}; they are replaced with a string that makes the
filename unique.  Returns a file descriptor open on the file for
reading and writing.
@end deftypefn
*/
int mkstemps(char* tmpl, int suffix_len) {
  int len;
  char* XXXXXX;
  static unsigned long long value;
  unsigned long long random_time_bits;
  unsigned int count;
  int fd = -1;
  int save_errno = errno;

  /* A lower bound on the number of temporary files to attempt to
     generate.  The maximum total number of temporary file names that
     can exist for a given template is 36**6.  It should never be
     necessary to try all these combinations.  Instead if a reasonable
     number of names is tried (we define reasonable as 36**3) fail to
     give the system administrator the chance to remove the problems.  */
#define ATTEMPTS_MIN (36 * 36 * 36)

  /* The number of times to attempt to generate a temporary file.  To
     conform to POSIX, this must be no smaller than TMP_MAX.  */
#if ATTEMPTS_MIN < TMP_MAX
  unsigned int attempts = TMP_MAX;
#else
  unsigned int attempts = ATTEMPTS_MIN;
#endif

  len = strlen(tmpl);
  if (len < 6 + suffix_len ||
      strncmp(&tmpl[len - 6 - suffix_len], "XXXXXX", 6)) {
    return -1;
  }

  /* This is where the Xs start.  */
  XXXXXX = &tmpl[len - 6 - suffix_len];

  std::random_device rd;
  for (count = 0; count < attempts; ++count) {
    unsigned long long r = rd();
    unsigned long long v = (r << 32) | rd();

    /* Fill in the random bits.  */
    XXXXXX[0] = letters[v % 36];
    v /= 36;
    XXXXXX[1] = letters[v % 36];
    v /= 36;
    XXXXXX[2] = letters[v % 36];
    v /= 36;
    XXXXXX[3] = letters[v % 36];
    v /= 36;
    XXXXXX[4] = letters[v % 36];
    v /= 36;
    XXXXXX[5] = letters[v % 36];

    fd = _open(tmpl, _O_RDWR | _O_CREAT | _O_EXCL, _S_IWRITE | _S_IREAD);
    if (fd >= 0) {
      errno = save_errno;
      return fd;
    } else if (errno != EEXIST)
      return -1;
  }

  /* We got out of the loop because we ran out of combinations to try.  */
  errno = EEXIST;
  return -1;
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
    if (!name_.empty() && _access(name_.c_str(), 0) != -1) {
      _unlink(name_.c_str());
    }
    if (file_ != nullptr) {
      fclose(file_);
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
