#include "file_store_handler_op.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <thread>

#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

FileStoreHandler::FileStoreHandler(std::string& path) {
  basePath_ = realPath(path);
}

FileStoreHandler::~FileStoreHandler() {}

std::string FileStoreHandler::realPath(const std::string& path) {
  std::array<char, PATH_MAX> buf;
  CHECK_EQ(buf.data(), realpath(path.c_str(), buf.data())) << "realpath: "
                                                           << strerror(errno);
  return std::string(buf.data());
}

std::string FileStoreHandler::objectPath(const std::string& name) {
  std::string encoded;
  for (const auto& c : name) {
    // Convert non-alphabetic characters to octal.
    // Means argument cannot collide with encoding.
    // Don't want to take a dependency on SSL for SHA1 here.
    if (!isalpha(c)) {
      // 0-prefix, max 3 numbers, 0-terminator
      std::array<char, 5> buf;
      snprintf(buf.data(), buf.size(), "%#03o", c);
      encoded.append(buf.data());
    } else {
      encoded.append(&c, 1);
    }
  }
  return basePath_ + "/" + encoded;
}

void FileStoreHandler::set(const std::string& name, const std::string& data) {
  auto path = objectPath(name);
  WriteStringToFile(data, path.c_str());
}

std::string FileStoreHandler::get(const std::string& name) {
  // Block until key is set
  wait({name});

  std::string result;
  auto path = objectPath(name);
  ReadStringFromFile(path.c_str(), &result);
  return result;
}

int64_t FileStoreHandler::add(
    const std::string& /* unused */,
    int64_t /* unused */) {
  CHECK(false) << "add not implemented for FileStoreHandler";
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

void FileStoreHandler::wait(const std::vector<std::string>& names) {
  // Not using inotify because it doesn't work on many
  // shared filesystems (such as NFS).
  while (!check(names)) {
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
}
