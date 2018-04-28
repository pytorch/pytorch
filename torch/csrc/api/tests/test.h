#pragma once

#include "lest.hpp"
#include <autogradpp/autograd.h>

using namespace autograd;

#define CASE( name ) lest_CASE( specification(), name )

#define CUDA_GUARD if (!hasCuda()) {\
  std::cerr << "No cuda, skipping test" << std::endl; return;\
}

extern lest::tests & specification();
