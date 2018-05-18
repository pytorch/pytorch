#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <torch/torch.h>

#include <iostream>

// Custom main to disable CUDA tests when they are not available.
// https://github.com/catchorg/Catch2/blob/master/docs/own-main.md

int main(int argc, char* argv[]) {
  Catch::Session session;

  const auto return_code = session.applyCommandLine(argc, argv);
  if (return_code != 0) {
    return return_code;
  }

  if (!torch::hasCuda()) {
    std::cerr << "CUDA not available. Disabling CUDA tests" << std::endl;
    // ~ disables the [cuda] tag.
    session.configData().testsOrTags.push_back("~[cuda]");
  }

  return session.run();
}
