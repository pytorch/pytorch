/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/rendezvous/file_store.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>

#include "gloo/common/error.h"
#include "gloo/common/logging.h"

namespace gloo {
namespace rendezvous {

FileStore::FileStore(const std::string& path) {
  basePath_ = realPath(path);
}

std::string FileStore::realPath(const std::string& path) {
#if defined(_MSC_VER)
  std::array<char, _MAX_PATH> buf;
  auto ret = _fullpath(buf.data(), path.c_str(), buf.size());
#else
  std::array<char, PATH_MAX> buf;
  auto ret = realpath(path.c_str(), buf.data());
#endif
  GLOO_ENFORCE_EQ(buf.data(), ret, "realpath: ", strerror(errno));
  return std::string(buf.data());
}

static std::string encodeName(const std::string& name) {
  static std::hash<std::string> hashFn;
  return std::to_string(hashFn(name));
}

std::string FileStore::tmpPath(const std::string& name) {
  return basePath_ + "/." + encodeName(name);
}

std::string FileStore::objectPath(const std::string& name) {
  return basePath_ + "/" + encodeName(name);
}

void FileStore::set(const std::string& key, const std::vector<char>& data) {
  auto tmp = tmpPath(key);
  auto path = objectPath(key);

  {
    // Fail if the key already exists. This implementation is not race free.
    // A race free solution would need to atomically create the file 'path'
    // using an API that fails if the file exists (not provided by STL). If
    // created successfully, rename the temp file as below.
    std::ifstream ifs(path.c_str());
    GLOO_ENFORCE(!ifs.is_open(), "File already exists: ", path);
  }

  {
    std::ofstream ofs(tmp.c_str(), std::ios::out | std::ios::trunc);
    GLOO_ENFORCE(
        ofs.is_open(),
        "File cannot be created: ", tmp, " (", ofs.rdstate(), ")");
    ofs.write(data.data(), data.size());
  }

  // Atomically move result to final location
  auto rv = rename(tmp.c_str(), path.c_str());
  GLOO_ENFORCE_EQ(rv, 0, "rename: ", strerror(errno));
}

std::vector<char> FileStore::get(const std::string& key) {
  auto path = objectPath(key);
  std::vector<char> result;

  // Block until key is set
  wait({key});

  std::ifstream ifs(path.c_str(), std::ios::in);
  if (!ifs) {
    GLOO_ENFORCE(
        false, "File cannot be opened: ", path, " (", ifs.rdstate(), ")");
  }
  ifs.seekg(0, std::ios::end);
  size_t n = ifs.tellg();
  GLOO_ENFORCE_GT(n, 0);
  result.resize(n);
  ifs.seekg(0);
  ifs.read(result.data(), n);
  return result;
}

bool FileStore::check(const std::vector<std::string>& keys) {
  std::vector<std::string> paths;
  for (const auto& key : keys) {
    paths.push_back(objectPath(key));
  }

  for (const auto& path : paths) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      // Only deal with files that don't exist.
      // Anything else is a problem.
      GLOO_ENFORCE_EQ(errno, ENOENT);

      // One of the paths doesn't exist; return early
      return false;
    }

    close(fd);
  }

  return true;
}

void FileStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  // Not using inotify because it doesn't work on many
  // shared filesystems (such as NFS).
  const auto start = std::chrono::steady_clock::now();
  while (!check(keys)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG(
          "Wait timeout for key(s): ", ::gloo::MakeString(keys)));
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace rendezvous
} // namespace gloo
