#include "file_store_handler_op.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <array>
#include <chrono>
#include <iostream>
#include <thread>

#if defined(_MSC_VER)
#include <direct.h> // for _mkdir
#endif

#include "c10/util/StringUtil.h"

#include "caffe2/utils/murmur_hash3.h"

namespace caffe2 {

static std::string encodeName(const std::string& name) {
  std::array<uint64_t, 2> out;
  MurmurHash3_x64_128(name.data(), name.size(), 0xcafef00d, out.data());

  // Size is 33 to have space for final NUL
  std::array<char, 33> buf;
  for (int i = 0; i < 16; i++) {
    snprintf(&buf[i * 2], buf.size() - (i * 2), "%02x", ((char*)out.data())[i]);
  }

  // Return everything but the final NUL
  return std::string(buf.data(), buf.size() - 1);
}

FileStoreHandler::FileStoreHandler(
    const std::string& path,
    const std::string& prefix) {
  basePath_ = realPath(path);
  if (!prefix.empty()) {
    basePath_ = basePath_ + "/" + encodeName(prefix);
  }
#if defined(_MSC_VER)
  auto ret = _mkdir(basePath_.c_str());
#else
  auto ret = mkdir(basePath_.c_str(), 0777);
#endif // defined(_MSC_VER)
  if (ret == -1) {
    CHECK_EQ(errno, EEXIST) << "mkdir: " << strerror(errno);
  }
}

FileStoreHandler::~FileStoreHandler() {}

std::string FileStoreHandler::realPath(const std::string& path) {
#if defined(_MSC_VER)
  std::array<char, _MAX_PATH> buf;
  auto ret = _fullpath(buf.data(), path.c_str(), buf.size());
#else
  std::array<char, PATH_MAX> buf;
  auto ret = realpath(path.c_str(), buf.data());
#endif
  CHECK_EQ(buf.data(), ret) << "realpath: " << strerror(errno);
  return std::string(buf.data());
}

std::string FileStoreHandler::tmpPath(const std::string& name) {
  return basePath_ + "/." + encodeName(name);
}

std::string FileStoreHandler::objectPath(const std::string& name) {
  return basePath_ + "/" + encodeName(name);
}

void FileStoreHandler::set(const std::string& name, const std::string& data) {
  auto tmp = tmpPath(name);
  auto path = objectPath(name);

  {
    std::ofstream ofs(tmp.c_str(), std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
      CAFFE_ENFORCE(
          false, "File cannot be created: ", tmp, " (", ofs.rdstate(), ")");
    }
    ofs << data;
  }

  // Atomically movve result to final location
  auto rv = rename(tmp.c_str(), path.c_str());
  CAFFE_ENFORCE_EQ(rv, 0, "rename: ", strerror(errno));
}

std::string FileStoreHandler::get(
    const std::string& name,
    const std::chrono::milliseconds& timeout) {
  auto path = objectPath(name);
  std::string result;

  // Block until key is set
  wait({name}, timeout);

  std::ifstream ifs(path.c_str(), std::ios::in);
  if (!ifs) {
    CAFFE_ENFORCE(
        false, "File cannot be opened: ", path, " (", ifs.rdstate(), ")");
  }
  ifs.seekg(0, std::ios::end);
  size_t n = ifs.tellg();
  result.resize(n);
  ifs.seekg(0);
  ifs.read(&result[0], n);
  return result;
}

int64_t FileStoreHandler::add(
    const std::string& /* unused */,
    int64_t /* unused */) {
  CHECK(false) << "add not implemented for FileStoreHandler";
  return 0;
}

int64_t FileStoreHandler::getNumKeys() {
  CHECK(false) << "getNumKeys not implemented for FileStoreHandler";
  return 0;
}

bool FileStoreHandler::deleteKey(const std::string& /* unused */) {
  CHECK(false) << "deleteKey not implemented for FileStoreHandler";
  return false;
}

bool FileStoreHandler::check(const std::vector<std::string>& names) {
  std::vector<std::string> paths;
  for (const auto& name : names) {
    paths.push_back(objectPath(name));
  }

  for (const auto& path : paths) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      // Only deal with files that don't exist.
      // Anything else is a problem.
      CHECK_EQ(errno, ENOENT);

      // One of the paths doesn't exist; return early
      return false;
    }

    close(fd);
  }

  return true;
}

void FileStoreHandler::wait(
    const std::vector<std::string>& names,
    const std::chrono::milliseconds& timeout) {
  // Not using inotify because it doesn't work on many
  // shared filesystems (such as NFS).
  const auto start = std::chrono::steady_clock::now();
  while (!check(names)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      STORE_HANDLER_TIMEOUT(
          "Wait timeout for name(s): ", c10::Join(" ", names));
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
}
