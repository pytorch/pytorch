#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include <gloo/transport/tcp/device.h>

#include "FileStore.hpp"
#include "ProcessGroupGloo.hpp"

class Semaphore {
 public:
  void post(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    n_ += n;
    cv_.notify_all();
  }

  void wait(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    while (n_ < n) {
      cv_.wait(lock);
    }
    n_ -= n;
  }

 protected:
  int n_ = 0;
  std::mutex m_;
  std::condition_variable cv_;
};

std::string tmppath() {
  const char* tmpdir = getenv("TMPDIR");
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";
  }

  // Create template
  std::vector<char> tmp(256);
  auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);
  tmp.resize(len);

  // Create temporary file
  auto fd = mkstemp(&tmp[0]);
  if (fd == -1) {
    throw std::system_error(errno, std::system_category());
  }
  close(fd);
  return std::string(tmp.data(), tmp.size());
}

struct TemporaryFile {
  std::string path;

  TemporaryFile() {
    path = tmppath();
  }

  ~TemporaryFile() {
    unlink(path.c_str());
  }
};

struct Fork {
  pid_t pid;

  Fork() {
    pid = fork();
    if (pid < 0) {
      throw std::system_error(errno, std::system_category(), "fork");
    }
  }

  ~Fork() {
    if (pid > 0) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
  }

  bool isChild() {
    return pid == 0;
  }
};

class SignalTest {
 public:
  SignalTest(const std::string& path)
      : path_(path) {
  }

  ~SignalTest() {
    if (arm_.joinable()) {
      arm_.join();
    }
  }

  // Arms test to send signal to PID when the semaphore unlocks. This
  // happens as soon as the first collective completes successfully.
  void arm(int pid, int signal) {
    arm_ = std::move(std::thread([=] {
          sem_.wait();
          kill(pid, signal);
        }));
  }

  std::shared_ptr<::c10d::ProcessGroup::Work> run(int rank, int size) {
    auto store = std::make_shared<::c10d::FileStore>(path_);

    // Local test; all connections are made through loopback
    std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
    devices.push_back(::gloo::transport::tcp::CreateDevice("localhost"));

    ::c10d::ProcessGroupGloo pg(store, devices, rank, size);
    pg.initialize();

    // Initialize tensor list
    std::vector<at::Tensor> tensors = {
      at::ones(at::CPU(at::kFloat), {16, 16}),
    };

    // Loop until an exception happens
    std::shared_ptr<::c10d::ProcessGroup::Work> work;
    while (true) {
      work = pg.allreduce(tensors);
      if (!work->wait()) {
        break;
      }
      sem_.post();
    }

    pg.destroy();
    return std::move(work);
  }

 protected:
  std::string path_;
  std::thread arm_;
  Semaphore sem_;
};

std::shared_ptr<::c10d::ProcessGroup::Work> testSignal(
    const std::string& path,
    int signal) {
  Fork fork;
  if (fork.isChild()) {
    SignalTest test(path);
    test.run(1, 2);
    exit(1);
  }

  SignalTest test(path);
  test.arm(fork.pid, signal);
  return test.run(0, 2);
}

int main(int argc, char** argv) {
  {
    TemporaryFile file;
    auto work = testSignal(file.path, SIGSTOP);
    auto& ex = work->exception();
    std::cout << "SIGSTOP test got: " << ex.what() << std::endl;
  }

  {
    TemporaryFile file;
    auto work = testSignal(file.path, SIGKILL);
    auto& ex = work->exception();
    std::cout << "SIGKILL test got: " << ex.what() << std::endl;
  }

  return 0;
}
