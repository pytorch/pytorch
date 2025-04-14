#pragma once

#include <ATen/Config.h>
#include <ATen/core/Generator.h>
#include <c10/core/GeneratorImpl.h>
#include <cstdint>

#include <mkl.h>

namespace at {

struct TORCH_API MKLGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  MKLGeneratorImpl(uint64_t seed_in = default_rng_seed_val);
  ~MKLGeneratorImpl() override = default;

  // MKLGeneratorImpl methods
  std::shared_ptr<MKLGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t seed() override;
  uint64_t current_seed() const override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  static c10::DeviceType device_type();
  void get_stream_copy(VSLStreamStatePtr &streamCopy);
  void skip_ahead(uint64_t n);
  void set_lock_seed(bool lock_seed);
  bool get_lock_seed() const;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;

 private:
  MKLGeneratorImpl* clone_impl() const override;
  void advance_offset(uint64_t n);
  VSLStreamStatePtr stream_;
  uint64_t seed_;
  uint64_t offset_;
  bool lock_seed_;
};

namespace detail {

TORCH_API const Generator& getDefaultMKLGenerator();
TORCH_API Generator
createMKLGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

} // namespace at
