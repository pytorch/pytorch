#pragma once

// ${generated_comment}

#include <$header>

#include "ATen/core/Generator.h"

namespace at {

class Context;
struct ${name}Generator : public Generator {
  ${name}Generator(Context * context);
  CAFFE2_API virtual ~${name}Generator();

  CAFFE2_API virtual ${name}Generator& copy(const Generator& from) override;
  CAFFE2_API virtual ${name}Generator& free() override;

  CAFFE2_API virtual uint64_t seed() override;
  CAFFE2_API virtual uint64_t initialSeed() override;
  CAFFE2_API virtual ${name}Generator& manualSeed(uint64_t seed) override;
  CAFFE2_API virtual ${name}Generator& manualSeedAll(uint64_t seed) override;
  CAFFE2_API virtual void * unsafeGetTH() override;

//TODO(zach): figure out friends later
public:
  Context * context;
  ${th_generator}
};

}
