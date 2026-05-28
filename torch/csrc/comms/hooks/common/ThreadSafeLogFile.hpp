// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace torch::comms {

class ThreadSafeLogFile {
 public:
  void open(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.open(path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open log file: " + path);
    }
  }

  void writeLine(const std::string& line) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.write(line.data(), line.size());
    file_ << '\n';
    file_.flush();
  }

  ~ThreadSafeLogFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
      file_.close();
    }
  }

  ThreadSafeLogFile() = default;
  ThreadSafeLogFile(const ThreadSafeLogFile&) = delete;
  ThreadSafeLogFile& operator=(const ThreadSafeLogFile&) = delete;
  ThreadSafeLogFile(ThreadSafeLogFile&&) = delete;
  ThreadSafeLogFile& operator=(ThreadSafeLogFile&&) = delete;

 private:
  std::ofstream file_;
  std::mutex mutex_;
};

} // namespace torch::comms
