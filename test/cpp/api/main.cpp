#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <torch/cuda.h>

#include <iostream>

// Custom main to disable CUDA tests when they are not available.
// https://github.com/catchorg/Catch2/blob/master/docs/own-main.md

int main(int argc, char* argv[]) {
  Catch::Session session;

  const auto return_code = session.applyCommandLine(argc, argv);
  if (return_code != 0) {
    return return_code;
  }

  // ~ disables tags.
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA not available. Disabling [cuda] and [multi-cuda] tests"
              << std::endl;
    session.configData().testsOrTags.emplace_back("~[cuda]");
    session.configData().testsOrTags.emplace_back("~[multi-cuda]");
  } else if (torch::cuda::device_count() < 2) {
    std::cerr << "Only one CUDA device detected. Disabling [multi-cuda] tests"
              << std::endl;
    session.configData().testsOrTags.emplace_back("~[multi-cuda]");
  }

  return session.run();
}
