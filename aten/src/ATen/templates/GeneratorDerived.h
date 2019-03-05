#pragma once

// ${generated_comment}

#include <$header>

#include <ATen/core/Generator.h>

namespace at {

class Context;
struct ${name}Generator : public CloneableGenerator<${name}Generator, Generator> {
  CAFFE2_API ${name}Generator(Context * context);
  CAFFE2_API ~${name}Generator();

  CAFFE2_API ${name}Generator& copy(const Generator& from);
  CAFFE2_API ${name}Generator& free();

  CAFFE2_API uint64_t seed();
  CAFFE2_API uint64_t current_seed() const override;
  CAFFE2_API void set_current_seed(uint64_t seed) override;
  CAFFE2_API static DeviceType device_type();
  CAFFE2_API ${name}Generator& manualSeedAll(uint64_t seed) override;
  CAFFE2_API void * unsafeGetTH();

//TODO(zach): figure out friends later
public:
  Context * context;
  ${th_generator}

private:
  CloneableGenerator<${name}Generator, Generator>* clone_impl() const override;
};

}
