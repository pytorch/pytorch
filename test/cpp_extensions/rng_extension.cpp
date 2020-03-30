#include <torch/extension.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <memory>

using namespace at;

static size_t instance_count = 0;

struct TestCPUGenerator : public c10::GeneratorImpl {
  TestCPUGenerator(uint64_t value) : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, value_(value) {
    ++instance_count;
  }
  ~TestCPUGenerator() {
    --instance_count;
  }
  uint32_t random() { return static_cast<uint32_t>(value_); }
  uint64_t random64() { return value_; }
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  static DeviceType device_type() { return DeviceType::CPU; }

  uint64_t value_;
};

Tensor& random_(Tensor& self, Generator generator) {
  return at::native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, Generator generator) {
  return at::native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, Generator generator) {
  return random_from_to(self, 0, to, generator);
}

Generator createTestCPUGenerator(uint64_t value) {
  return at::make_generator<TestCPUGenerator>(value);
}

Generator identity(Generator g) {
  return g;
}

size_t getInstanceCount() {
  return instance_count;
}

static auto registry = torch::RegisterOperators()
      .op(torch::RegisterOperators::options()
        .schema("aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(random_from_to), &random_from_to>(DispatchKey::CustomRNGKeyId))
      .op(torch::RegisterOperators::options()
        .schema("aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(random_to), &random_to>(DispatchKey::CustomRNGKeyId))
      .op(torch::RegisterOperators::options()
        .schema("aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(random_), &random_>(DispatchKey::CustomRNGKeyId));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createTestCPUGenerator", &createTestCPUGenerator);
  m.def("getInstanceCount", &getInstanceCount);
  m.def("identity", &identity);
}
