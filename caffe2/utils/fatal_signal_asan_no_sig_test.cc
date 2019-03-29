#include "caffe2/utils/signal_handler.h"
#if defined(CAFFE2_SUPPORTS_FATAL_SIGNAL_HANDLERS)
#include <gtest/gtest.h>
#include <pthread.h>
#include <unistd.h>

#include <functional>
#include <iostream>
#include <array>

#include "caffe2/core/common.h"

namespace {
void* dummy_thread(void*) {
  while (1) {
  }
  return nullptr;
}

bool forkAndPipe(
    std::string& stderrBuffer,
    std::function<void(void)> callback) {
  std::array<int, 2> stderrPipe;
  if (pipe(stderrPipe.data()) != 0) {
    perror("STDERR pipe");
    return false;
  }
  pid_t child = fork();
  if (child == 0) {
    // Replace this process' stderr so we can read it.
    if (dup2(stderrPipe[1], STDERR_FILENO) < 0) {
      close(stderrPipe[0]);
      close(stderrPipe[1]);
      perror("dup2 STDERR");
      exit(5);
    }

    // This is for the parent to work with.
    close(stderrPipe[0]);
    close(stderrPipe[1]);

    callback();
    exit(7);
  } else if (child > 0) {
    const int bufferSize = 128;
    std::array<char, bufferSize> buffer;

    // We want to close the writing end of the pipe right away so our
    // read actually gets an EOF.
    close(stderrPipe[1]);

    // wait for child to finish crashing.
    int statloc;
    if (wait(&statloc) < 0) {
      close(stderrPipe[0]);
      perror("wait");
      return false;
    }

    ssize_t bytesRead;
    while ((bytesRead = read(stderrPipe[0], buffer.data(), bufferSize)) > 0) {
      const std::string tmp(buffer.data(), bytesRead);
      std::cout << tmp;
      stderrBuffer += tmp;
    }

    // The child should have exited due to signal.
    if (!WIFSIGNALED(statloc)) {
      fprintf(stderr, "Child didn't exit because it received a signal\n");
      if (WIFEXITED(statloc)) {
        fprintf(stderr, "Exited with code: %d\n", WEXITSTATUS(statloc) & 0xff);
      }
      return false;
    }

    if (bytesRead < 0) {
      perror("read");
      return false;
    }

    close(stderrPipe[0]);
    return true;
  } else {
    perror("fork");
    return false;
  }
}
} // namespace

#define _TEST_FATAL_SIGNAL(signum, name, threadCount, print, expected)       \
  do {                                                                       \
    std::string stderrBuffer;                                                \
    ASSERT_TRUE(forkAndPipe(stderrBuffer, [=]() {                            \
      caffe2::setPrintStackTracesOnFatalSignal(print);                       \
      pthread_t pt;                                                          \
      for (int i = 0; i < threadCount; i++) {                                \
        if (pthread_create(&pt, nullptr, ::dummy_thread, nullptr)) {         \
          perror("pthread_create");                                          \
        }                                                                    \
      }                                                                      \
      raise(signum);                                                         \
    }));                                                                     \
    int keyPhraseCount = 0;                                                  \
    std::string keyPhrase =                                                  \
        std::string(name) + "(" + c10::to_string(signum) + "), Thread";      \
    size_t loc = 0;                                                          \
    while ((loc = stderrBuffer.find(keyPhrase, loc)) != std::string::npos) { \
      keyPhraseCount += 1;                                                   \
      loc += 1;                                                              \
    }                                                                        \
    EXPECT_EQ(keyPhraseCount, expected);                                     \
  } while (0)

#define TEST_FATAL_SIGNAL(signum, name, threadCount) \
  _TEST_FATAL_SIGNAL(signum, name, threadCount, true, threadCount + 1)

#define TEST_FATAL_SIGNAL_NO_PRINT(signum, name, threadCount) \
  _TEST_FATAL_SIGNAL(signum, name, threadCount, false, 0)

TEST(fatalSignalTest, SIGABRT8) {
  TEST_FATAL_SIGNAL(SIGABRT, "SIGABRT", 8);
}

TEST(fatalSignalTest, SIGINT8) {
  TEST_FATAL_SIGNAL(SIGINT, "SIGINT", 8);
}

TEST(fatalSignalTest, SIGILL8) {
  TEST_FATAL_SIGNAL(SIGILL, "SIGILL", 8);
}

TEST(fatalSignalTest, SIGFPE8) {
  TEST_FATAL_SIGNAL(SIGFPE, "SIGFPE", 8);
}

TEST(fatalSignalTest, SIGBUS8) {
  TEST_FATAL_SIGNAL(SIGBUS, "SIGBUS", 8);
}

TEST(fatalSignalTest, SIGSEGV8) {
  TEST_FATAL_SIGNAL(SIGSEGV, "SIGSEGV", 8);
}

// Test that if we don't enable printing stack traces then we don't get any.
TEST(fatalSignalTest, SIGABRT8_NOPRINT) {
  TEST_FATAL_SIGNAL_NO_PRINT(SIGABRT, "SIGABRT", 8);
}
#endif // defined(CAFFE2_SUPPORTS_FATAL_SIGNAL_HANDLERS)
