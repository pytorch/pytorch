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
  SignalTest(const std::string& path) : path_(path) {}

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

    // Use tiny timeout to make this test run fast
    ::c10d::ProcessGroupGloo::Options options;
    options.timeout = std::chrono::milliseconds(50);

    ::c10d::ProcessGroupGloo pg(store, rank, size, options);

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

class CollectiveTest {
 public:
  static std::vector<CollectiveTest> initialize(
      const std::string& path,
      int num) {
    std::vector<CollectiveTest> tests;
    for (auto i = 0; i < num; i++) {
      tests.push_back(std::move(CollectiveTest(path)));
    }

    std::vector<std::thread> threads;
    for (auto i = 0; i < num; i++) {
      threads.push_back(std::move(
          std::thread([i, &tests] { tests[i].start(i, tests.size()); })));
    }
    for (auto& thread : threads) {
      thread.join();
    }

    return std::move(tests);
  }

  CollectiveTest(const std::string& path) : path_(path) {}

  CollectiveTest(CollectiveTest&& other) {
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  ::c10d::ProcessGroupGloo& getProcessGroup() {
    return *pg_;
  }

  void start(int rank, int size) {
    auto store = std::make_shared<::c10d::FileStore>(path_);

    // Use tiny timeout to make this test run fast
    ::c10d::ProcessGroupGloo::Options options;
    options.timeout = std::chrono::milliseconds(50);

    pg_ = std::unique_ptr<::c10d::ProcessGroupGloo>(
        new ::c10d::ProcessGroupGloo(store, rank, size, options));
  }

 protected:
  std::string path_;
  std::unique_ptr<::c10d::ProcessGroupGloo> pg_;
};

void testAllreduce(const std::string& path) {
  const auto size = 4;
  auto tests = CollectiveTest::initialize(path, size);

  // Generate inputs
  std::vector<std::vector<at::Tensor>> inputs(size);
  for (auto i = 0; i < size; i++) {
    auto tensor = at::ones(at::CPU(at::kFloat), {16, 16}) * i;
    inputs[i] = std::vector<at::Tensor>({tensor});
  }

  // Kick off work
  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> work(size);
  for (auto i = 0; i < size; i++) {
    work[i] = tests[i].getProcessGroup().allreduce(inputs[i]);
  }

  // Wait for work to complete
  for (auto i = 0; i < size; i++) {
    if (!work[i]->wait()) {
      throw work[i]->exception();
    }
  }

  // Verify outputs
  const auto expected = (size * (size - 1)) / 2;
  for (auto i = 0; i < size; i++) {
    auto& tensor = inputs[i][0];
    auto data = tensor.data<float>();
    for (auto j = 0; j < tensor.numel(); j++) {
      if (data[j] != expected) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
}

void testBroadcast(const std::string& path) {
  const auto size = 2;
  const auto stride = 2;
  auto tests = CollectiveTest::initialize(path, size);

  std::vector<std::vector<at::Tensor>> inputs(size);
  const auto& type = at::CPU(at::kFloat);

  // Try every permutation of root rank and root tensoro
  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < stride; j++) {
      // Initialize inputs
      for (auto k = 0; k < size; k++) {
        inputs[k].resize(stride);
        for (auto l = 0; l < stride; l++) {
          inputs[k][l] = at::ones(type, {16, 16}) * (k * stride + l);
        }
      }

      ::c10d::BroadcastOptions options;
      options.rootRank = i;
      options.rootTensor = j;

      // Kick off work
      std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> work(size);
      for (auto i = 0; i < size; i++) {
        work[i] = tests[i].getProcessGroup().broadcast(inputs[i], options);
      }

      // Wait for work to complete
      for (auto i = 0; i < size; i++) {
        if (!work[i]->wait()) {
          throw work[i]->exception();
        }
      }

      // Verify outputs
      const auto expected = (i * stride + j);
      for (auto k = 0; k < size; k++) {
        for (auto l = 0; l < stride; l++) {
          auto& tensor = inputs[k][l];
          auto data = tensor.data<float>();
          for (auto n = 0; n < tensor.numel(); n++) {
            if (data[n] != expected) {
              throw std::runtime_error("BOOM!");
            }
          }
        }
      }
    }
  }
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

  {
    TemporaryFile file;
    testAllreduce(file.path);
  }

  {
    TemporaryFile file;
    testBroadcast(file.path);
  }

  return 0;
}
