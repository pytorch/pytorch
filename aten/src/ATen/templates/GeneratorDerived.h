#pragma once

// ${generated_comment}

#include <$header>

#include <ATen/core/Generator.h>

namespace at {

class Context;
struct ${name}Generator : public Generator {
  CAFFE2_API ${name}Generator(Context * context);
  CAFFE2_API ~${name}Generator();

  CAFFE2_API ${name}Generator& copy(const Generator& from);
  CAFFE2_API ${name}Generator& free();

  CAFFE2_API uint64_t seed();
  CAFFE2_API uint64_t initialSeed();
  CAFFE2_API ${name}Generator& manualSeed(uint64_t seed);
  CAFFE2_API ${name}Generator& manualSeedAll(uint64_t seed) override;
  CAFFE2_API void * unsafeGetTH();

//TODO(zach): figure out friends later
public:
  Context * context;
  ${th_generator}
};

}
