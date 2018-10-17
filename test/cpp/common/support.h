#pragma once

#include <ATen/Error.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef WIN32
#include <unistd.h>
#endif

namespace torch {
namespace test {

#ifdef WIN32
struct TempFile {
  std::string name{std::tmpnam(nullptr)};
};
#else
struct TempFile {
  TempFile() {
    // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
    char filename[] = "/tmp/fileXXXXXX";
    fd_ = mkstemp(filename);
    AT_CHECK(fd_ != -1, "Error creating tempfile");
    name.assign(filename);
  }

  ~TempFile() {
    close(fd_);
  }

  std::string name;
  int fd_;
};
#endif

#define ASSERT_THROWS_WITH(statement, substring)                        \
  try {                                                                 \
    (void)statement;                                                    \
    FAIL() << "Expected statement `" #statement                         \
              "` to throw an exception, but it did not";                \
  } catch (const std::exception& e) {                                   \
    std::string message = e.what();                                     \
    if (message.find(substring) == std::string::npos) {                 \
      FAIL() << "Error message \"" << message                           \
             << "\" did not contain expected substring \"" << substring \
             << "\"";                                                   \
    }                                                                   \
  }

} // namespace test
} // namespace torch
