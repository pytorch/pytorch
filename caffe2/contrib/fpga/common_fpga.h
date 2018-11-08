#pragma once

#include <string>

class FPGAEngine {
 public:
  static std::string name;
};

typedef uint16_t bfloat16;

union bfp_converter {
  bfloat16 bfp[2];
  float fp32;
  unsigned int uint;
};
