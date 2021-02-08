/**
 * This file contains some general registration test cases.
 * More detailed test cases containing different APIs for registering kernels
 * are found in other files in this directory.
 */

#include <gtest/gtest.h>

// This file intentionally tests some deprecated APIs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <ATen/core/boxing/impl/test_helpers.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <functional>

#include <ATen/core/LegacyTypeDispatch.h>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::OperatorHandle;
using c10::Dispatcher;
using c10::IValue;
using c10::DispatchKey;

using torch::Library;
using torch::CppFunction;

using at::Tensor;

namespace {

struct DummyKernel final : OperatorKernel {
  void operator()(Tensor) {}
};

struct MockKernel final : OperatorKernel {
  MockKernel(bool* called): called_(called) {}

  void operator()(Tensor) {
    *called_ = true;
  }
private:
  bool* called_;
};

TEST(OperatorRegistrationTest, whenRegisteringWithSchemaBeforeKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().schema("_test::dummy(Tensor dummy) -> ()").catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithSchemaAfterKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy(Tensor dummy) -> ()"));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithNameBeforeKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().schema("_test::dummy").catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithNameAfterKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy"));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithoutSchema_thenFails) {
  expectThrows<c10::Error>([] {
    c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<DummyKernel>());
  }, "In operator registration: Tried to register an operator without specifying a schema or operator name.");
}

TEST(OperatorRegistrationTest, whenCallingOpWithWrongDispatchKey_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>(c10::DispatchKey::CPU));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");
}

TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenCallingOp_thenCallsCatchallKernel) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

// TODO Rewrite (since this is now allowed) and reenable
// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernel_thenFails) {
//   bool called = false;
//   auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
//   expectThrows<c10::Error>([&] {
//     c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }

// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernelInSameOpCall_thenFails) {
//   bool called = false;
//   expectThrows<c10::Error>([&] {
//     auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
//       .catchAllKernel<MockKernel>(&called)
//       .kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }

TEST(OperatorRegistrationTest, givenOpWithDispatchedKernelOutOfScope_whenRegisteringCatchallKernelAndCallingOp_thenCallsCatchallKernel) {
  bool called = false;
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));
  }

  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

// TODO Rewrite (since this is now allowed) and reenable
// TEST(OperatorRegistrationTest, givenOpWithDispatchedKernel_whenRegisteringCatchallKernel_thenFails) {
//   bool called = false;
//   auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   expectThrows<c10::Error>([&] {
//     c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
//   }, "Tried to register a catch-all kernel for an operator which already has kernels for dispatch keys CPU. An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is _test::dummy");
// }
//
// TEST(OperatorRegistrationTest, givenOpWithDispatchedKernel_whenRegisteringCatchallKernelInSameOpCall_thenFails) {
//   bool called = false;
//   expectThrows<c10::Error>([&] {
//     auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
//       .kernel<MockKernel>(c10::DispatchKey::CPU, &called)
//       .catchAllKernel<MockKernel>(&called));
//   }, "Tried to register a catch-all kernel for an operator which already has kernels for dispatch keys CPU. An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is _test::dummy");
// }

TEST(OperatorRegistrationTest, givenOpWithCatchallKernelOutOfScope_whenRegisteringDispatchedKernelAndCallingOp_thenCallsCatchallKernel) {
  bool called = false;
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
  }

  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringWithSchema_thenOnlyRegistersSchema) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringWithoutSchema_thenFails) {
  expectThrows<c10::Error>([&] {
    c10::RegisterOperators().op("_test::dummy");
  }, "Cannot infer operator schema in registration of operator _test::dummy because there is no kernel specified.");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRunningOutOfScope_thenSchemaIsGone) {
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  }

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  EXPECT_FALSE(op.has_value());
}

TEST(OperatorRegistrationTest, givenOpWithoutKernelsWithoutTensorInputs_whenRegistering_thenRegisters) {
  // as long as we don't register non-catchall kernels, ops without tensor arguments are fine
  auto registrar = c10::RegisterOperators().op("_test::dummy() -> ()");

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered
}

TEST(OperatorRegistrationTest, givenMultipleKernelsWithSameDispatchKey_whenRegisteringInSameOpCall_thenFails) {
  expectThrows<c10::Error>([&] {
    auto registrar = c10::RegisterOperators()
        .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
            .kernel<DummyKernel>(c10::DispatchKey::CPU)
            .kernel<DummyKernel>(c10::DispatchKey::CPU));
  }, "In operator registration: Tried to register multiple kernels with same dispatch key CPU for operator schema _test::dummy");
}

TEST(OperatorRegistrationTest, givenMultipleCatchallKernels_whenRegisteringInSameOpCall_thenFails) {
  expectThrows<c10::Error>([&] {
    auto registrar = c10::RegisterOperators()
        .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
            .catchAllKernel<DummyKernel>()
            .catchAllKernel<DummyKernel>());
  }, "Tried to register multiple catch-all kernels for operator schema _test::dummy");
}

TEST(OperatorRegistrationTest, whenRegisteringCPUTensorType_thenCanOnlyCallUnboxedWithCPUDispatchKey) {
  bool called_kernel_cpu = false;
  auto registrar= c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<MockKernel>(c10::DispatchKey::CPU, &called_kernel_cpu));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  // Ensure that dispatcher doesn't take the dispatch key from the tensor but from the direct argument instead.
  called_kernel_cpu = false;
  callOpUnboxedWithDispatchKey<void, Tensor>(*op, c10::DispatchKey::CPU, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called_kernel_cpu);

  // Ensure that disptach key from tensor is not used here.
  called_kernel_cpu = false;
  expectThrows<c10::Error>([&] {
    callOpUnboxedWithDispatchKey<void, Tensor>(*op, c10::DispatchKey::CUDA, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsInSameOpCallAndCalling_thenCallsCorrectKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar0 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<MockKernel>(c10::DispatchKey::CPU, &called_kernel1)
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel2));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel1 = called_kernel2 = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_kernel1);
  EXPECT_FALSE(called_kernel2);

  called_kernel1 = called_kernel2 = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_FALSE(called_kernel1);
  EXPECT_TRUE(called_kernel2);

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::XLA));
  }, "Could not run '_test::dummy' with arguments from the 'XLA'"
  " backend.");

  // also assert that the error message contains the available tensor type ids, but don't assert their order
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::XLA));
  }, "CPU");
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::XLA));
  }, "CUDA");
}

bool called_stackbased_kernel = false;
void stackBasedKernel(const OperatorHandle&, c10::Stack* stack) {
  called_stackbased_kernel = true;
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsByNameAndNoneCanInferSchema_thenFails) {
  bool called_kernel = false;
  expectThrows<c10::Error>([&] {
    auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
      .kernel<&stackBasedKernel>(c10::DispatchKey::CUDA)
      .kernel<&stackBasedKernel>(c10::DispatchKey::XLA));
  }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::dummy");
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsBySchemaAndNoneCanInferSchema_thenSucceeds) {
  bool called_kernel = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    .kernel<&stackBasedKernel>(c10::DispatchKey::CUDA)
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::XLA));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsByNameAndOnlyOneCanInferSchema_thenSucceeds) {
  bool called_kernel = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel)
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_FALSE(called_stackbased_kernel);
  EXPECT_TRUE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::XLA));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsBySchemaAndOnlyOneCanInferSchema_thenSucceeds) {
  bool called_kernel = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel)
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_FALSE(called_stackbased_kernel);
  EXPECT_TRUE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::XLA));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);
}

struct DummyKernelWithIntParam final : OperatorKernel {
  void operator()(Tensor, int64_t) {}
};

TEST(OperatorRegistrationTest, whenRegisteringMismatchingKernelsInSameOpCall_thenFails) {
  bool called_kernel = false;
  expectThrows<c10::Error>([&] {
    auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel<DummyKernelWithIntParam>(c10::DispatchKey::CPU)
      .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel));
  }, "Mismatch in kernel C++ signatures");
}

void backend_fallback_kernel(const c10::OperatorHandle& op, c10::Stack* stack) {
  (*stack)[1] = (*stack)[1].toString()->string() + op.schema().name();
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernel_thenCanBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_EQ("hello _test::dummy", stack[1].toString()->string());
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelForWrongBackend_thenCannotBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CUDA, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>([&] {
    auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  }, "Could not run '_test::dummy' with arguments from the 'CPU' backend.");
}

bool called = false;

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenRegularKernelCanBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CUDA, [] (Tensor, std::string) {
        called = true;
      }));
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CUDA), "hello ");
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenFallbackKernelCanBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CUDA, [] (Tensor, std::string) {
        called = true;
      }));
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_FALSE(called);
  EXPECT_EQ("hello _test::dummy", stack[1].toString()->string());
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForSameBackend_thenCallsRegularKernel) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CPU, [] (Tensor, std::string) {
        called = true;
      }));
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_TRUE(called);
}

bool called_autograd = false;
bool called_nonautograd = false;

void nonautograd_kernel(Tensor a) {
  called_nonautograd = true;
}

void autograd_kernel(Tensor a) {
  called_autograd = true;
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernel_thenCanCallAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called_autograd = false;
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");

  op->typed<void(Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernelWithRegularKernel_thenCanCallAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(nonautograd_kernel), nonautograd_kernel>(DispatchKey::CPU)
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_FALSE(called_nonautograd);
  EXPECT_TRUE(called_autograd);
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>()
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  // catchAll now maps to Math which has higher precedence than Autograd
  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallCatchallKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>()
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);
}

TEST(OperatorRegistrationTest, AutogradBackendOverridesAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(DispatchKey::AutogradCPU)
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
  EXPECT_FALSE(called_nonautograd);
}

TEST(OperatorRegistrationTest, AutogradXLAOverridesAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(DispatchKey::AutogradXLA)
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::XLA));
  }, "Could not run '_test::dummy' with arguments from the 'XLA'"
  " backend.");

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
  EXPECT_FALSE(called_nonautograd);
}

TEST(OperatorRegistrationTest, whenRegisterWithXLAKernelAndCatchAll_AutogradXLAIsNotFilled) {
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
      .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

    auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
    ASSERT_TRUE(op.has_value());

    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
    EXPECT_TRUE(called_nonautograd);
    EXPECT_FALSE(called_autograd);

    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA));
    EXPECT_FALSE(called_autograd);
    EXPECT_TRUE(called_nonautograd);
  }
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
      .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::XLA)
      .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

    auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
    ASSERT_TRUE(op.has_value());

    // When there's direct registration to XLA backend, AutogradXLA doesn't pick up catchAll
    // kernel in precompute but just keep fallthrough kernel from backend fallback.
    // Thus it falls through AutogradXLA and reaches the kernel at XLA key.
    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
    EXPECT_FALSE(called_nonautograd);
    EXPECT_TRUE(called_autograd);

    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA));
    EXPECT_TRUE(called_autograd);
    EXPECT_FALSE(called_nonautograd);
  }
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringWithMismatchingCppSignatures_thenFails) {
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {})
      .kernel(DispatchKey::CUDA, [] (int64_t&) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringCatchAllAndBackendWithMismatchingCppSignatures_thenFails) {
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {})
      .catchAllKernel([] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringBackendAndCatchAllWithMismatchingCppSignatures_thenFails) {
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .catchAllKernel([] (const int64_t&) {})
      .kernel(DispatchKey::CPU, [] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenAccessingWithMismatchingCppSignatures_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .kernel(DispatchKey::CPU, [] (int64_t) {}));
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .catchAllKernel([] (int64_t) {}));
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenRegisteringWithMismatchingCppSignatures_thenFails) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("dummy(int a) -> ()");
  m.impl("dummy", DispatchKey::CPU, [] (int64_t) {});
  expectThrows<c10::Error>([&] {
    m.impl("dummy", DispatchKey::CUDA, [] (const int64_t&) {});
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenAccessingWithMismatchingCppSignatures_thenFails) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("dummy(int a) -> ()");
  m.impl("dummy", DispatchKey::CPU, [] (int64_t) {});
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("dummy(int a) -> ()", [] (int64_t) {});
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");
}

/**
 * This is used to check that a given type works correctly when passed as input
 * to or as output from a kernel.
 *
 * Call ArgTypeTestKernel<Input, Output>::test(input, inputExpectation, output, outputExpectation, schema)
 * to test that a kernel with `Input` as input type and `Output` as output types,
 * when called with `input` fulfills `inputExpectation` inside the kernel, then
 * returns `output` and the returned value fulfills `outputExpectation`.
 *
 * `inputExpectation` and `outputExpectation` should be lambdas that run
 * googletest expect macros (or use other ways to assert the expectation is met).
 *
 * Optionally, you can specify the argument list part of a function schema
 * (e.g. "(Tensor a) -> Tensor") as an additional argument to use when
 * registering the kernel. In this case, the operator registration logic will
 * check that the kernel function signature matches the one you specified.
 */
struct TestModernAPI final {};
struct TestLegacyAPI final {};
struct TestModernAndLegacyAPI final {};

template<class InputType, class OutputType = InputType>
struct ArgTypeTestKernel final : OperatorKernel {
  explicit ArgTypeTestKernel(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output)
  : input_(std::move(input)), inputExpectation_(std::move(inputExpectation)), output_(std::move(output)) {}

  OutputType operator()(InputType input) const {
    inputExpectation_(std::move(input));
    return output_;
  }

  static void test(TestModernAndLegacyAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    test(TestModernAPI(), input, inputExpectation, output, outputExpectation, schema);
    test(TestLegacyAPI(), input, inputExpectation, output, outputExpectation, schema);
  }

  static void test(TestModernAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    return test_([&] {
      return c10::RegisterOperators().op("_test::my_op" + schema, c10::RegisterOperators::options().catchAllKernel<ArgTypeTestKernel>(input, inputExpectation, output));
    }, input, inputExpectation, output, outputExpectation, schema);
  }

  static void test(TestLegacyAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    return test_([&] {
      return c10::RegisterOperators().op("_test::my_op" + schema, [=] (InputType input) -> OutputType {
        inputExpectation(std::move(input));
        return output;
      });
    }, input, inputExpectation, output, outputExpectation, schema);
  }

private:
  static void test_(std::function<c10::RegisterOperators()> registration, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    auto registry = registration();
    auto op = Dispatcher::singleton().findSchema({"_test::my_op", ""});
    ASSERT_TRUE(op.has_value()); // assert schema is registered
    auto actualOutput = callOp(*op, input);
    outputExpectation(actualOutput);
  }

  InputType input_;
  std::function<void(const InputType&)> inputExpectation_;
  OutputType output_;
  std::string schema_;
};

template<class InputType, class OutputType = InputType>
struct testArgTypes final {
  template<class APIType = TestModernAndLegacyAPI>
  static void test(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const IValue&)> outputExpectation, const std::string& schema) {
    // Test with explicitly specified schema
    ArgTypeTestKernel<InputType, OutputType>::test(
      APIType(), input, inputExpectation, output, [&] (const c10::Stack& output) {
        EXPECT_EQ(1, output.size());
        outputExpectation(output[0]);
      }, schema
    );

    // Test with inferred schema
    ArgTypeTestKernel<InputType, OutputType>::test(
      APIType(), input, inputExpectation, output, [&] (const c10::Stack& output) {
        EXPECT_EQ(1, output.size());
        outputExpectation(output[0]);
      }, ""
    );

    // Test taking argument and returning nothing
    ArgTypeTestKernel<InputType, std::tuple<>>::test(
      APIType(), input, inputExpectation, {}, [] (const c10::Stack&) {}, ""
    );

    // Test taking argument and returning multiple outputs
    ArgTypeTestKernel<InputType, std::tuple<int64_t, OutputType>>::test(
      APIType(), input, inputExpectation, std::tuple<int64_t, OutputType>{3, output}, [&] (const c10::Stack& output) {
        EXPECT_EQ(2, output.size());
        EXPECT_EQ(3, output[0].toInt());
        outputExpectation(output[1]);
      }, ""
    );
  }
};

TEST(OperatorRegistrationTest, testAvailableArgTypes) {
  // TODO Test Scalar

  // primitive types
  testArgTypes<double>::test(
    1.5, [] (const double& v) {EXPECT_EQ(1.5, v);},
    2.5, [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
    "(float a) -> float");
  testArgTypes<int64_t>::test(
    1, [] (const int64_t& v) {EXPECT_EQ(1, v);},
    2, [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
    "(int a) -> int");
  testArgTypes<bool>::test(
    true, [] (const bool& v) {EXPECT_EQ(true, v);},
    false, [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
    "(bool a) -> bool");
  testArgTypes<bool>::test(
    false, [] (const bool& v) {EXPECT_EQ(false, v);},
    true, [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
    "(bool a) -> bool");
  testArgTypes<std::string>::test(
    "string1", [] (const std::string& v) {EXPECT_EQ("string1", v);},
    "string2", [] (const IValue& v) {EXPECT_EQ("string2", v.toString()->string());},
    "(str a) -> str");
  testArgTypes<Tensor>::test(
    dummyTensor(c10::DispatchKey::CPU), [] (const Tensor& v) {EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v));},
    dummyTensor(c10::DispatchKey::CUDA), [] (const IValue& v) {EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.toTensor()));},
    "(Tensor a) -> Tensor");


  // optional types (with has_value() == true)
  testArgTypes<c10::optional<double>>::test(
    c10::optional<double>(1.5), [] (const c10::optional<double>& v) {EXPECT_EQ(1.5, v.value());},
    c10::optional<double>(2.5), [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
    "(float? a) -> float?");
  testArgTypes<c10::optional<int64_t>>::test(
    c10::optional<int64_t>(1), [] (const c10::optional<int64_t>& v) {EXPECT_EQ(1, v.value());},
    c10::optional<int64_t>(2), [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
    "(int? a) -> int?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(true), [] (const c10::optional<bool>& v) {EXPECT_EQ(true, v.value());},
    c10::optional<bool>(false), [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(false), [] (const c10::optional<bool>& v) {EXPECT_EQ(false, v.value());},
    c10::optional<bool>(true), [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<std::string>>::test(
    c10::optional<std::string>("string1"), [] (const c10::optional<std::string>& v) {EXPECT_EQ("string1", v.value());},
    c10::optional<std::string>("string2"), [] (const IValue& v) {EXPECT_EQ("string2", v.toString()->string());},
    "(str? a) -> str?");
  testArgTypes<c10::optional<Tensor>>::test(
    c10::optional<Tensor>(dummyTensor(c10::DispatchKey::CPU)), [] (const c10::optional<Tensor>& v) {EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.value()));},
    c10::optional<Tensor>(dummyTensor(c10::DispatchKey::CUDA)), [] (const IValue& v) {EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.toTensor()));},
    "(Tensor? a) -> Tensor?");


  // optional types (with has_value() == false)
  testArgTypes<c10::optional<double>>::test(
    c10::optional<double>(c10::nullopt), [] (const c10::optional<double>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<double>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(float? a) -> float?");
  testArgTypes<c10::optional<int64_t>>::test(
    c10::optional<int64_t>(c10::nullopt), [] (const c10::optional<int64_t>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<int64_t>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(int? a) -> int?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(c10::nullopt), [] (const c10::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<bool>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(c10::nullopt), [] (const c10::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<bool>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<std::string>>::test(
    c10::optional<std::string>(c10::nullopt), [] (const c10::optional<std::string>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<std::string>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(str? a) -> str?");
  testArgTypes<c10::optional<Tensor>>::test(
    c10::optional<Tensor>(c10::nullopt), [] (const c10::optional<Tensor>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<Tensor>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(Tensor? a) -> Tensor?");


  // list types (with empty list)
  testArgTypes<c10::List<double>>::test(
    c10::List<double>(), [] (const c10::List<double>& v) {EXPECT_EQ(0, v.size());},
    c10::List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    "(float[] a) -> float[]");
  testArgTypes<c10::List<int64_t>>::test(
    c10::List<int64_t>(), [] (const c10::List<int64_t>& v) {EXPECT_EQ(0, v.size());},
    c10::List<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[] a) -> int[]");
  testArgTypes<c10::List<bool>>::test(
    c10::List<bool>(), [] (const c10::List<bool>& v) {EXPECT_EQ(0, v.size());},
    c10::List<bool>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<bool>>().size());},
    "(bool[] a) -> bool[]");
  testArgTypes<c10::List<std::string>>::test(
    c10::List<std::string>(), [] (const c10::List<std::string>& v) {EXPECT_EQ(0, v.size());},
    c10::List<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    "(str[] a) -> str[]");


  // list types (with non-empty list)
  testArgTypes<c10::List<double>>::test(
    c10::List<double>({1.5, 2.5}), [] (const c10::List<double>& v) {expectListEquals({1.5, 2.5}, v);},
    c10::List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    "(float[] a) -> float[]");
  testArgTypes<c10::List<int64_t>>::test(
    c10::List<int64_t>({1, 2}), [] (const c10::List<int64_t>& v) {expectListEquals({1, 2}, v);},
    c10::List<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[] a) -> int[]");
  testArgTypes<c10::List<bool>>::test(
    c10::List<bool>({true, false}), [] (const c10::List<bool>& v) {expectListEquals({true, false}, v);},
    c10::List<bool>({true, false}), [] (const IValue& v) {expectListEquals({true, false}, v.to<c10::List<bool>>());},
    "(bool[] a) -> bool[]");
  testArgTypes<c10::List<std::string>>::test(
    c10::List<std::string>({"first", "second"}), [] (const c10::List<std::string>& v) {expectListEquals({"first", "second"}, v);},
    c10::List<std::string>({"first", "second"}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toListRef().size());
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  testArgTypes<c10::List<Tensor>>::test(
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CUDA)}), [] (const c10::List<Tensor>& v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.get(0)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.get(1)));
    },
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CUDA), dummyTensor(c10::DispatchKey::CPU)}), [] (const IValue& v) {
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[] a) -> Tensor[]");

  // ArrayRef list types (with empty list)
  testArgTypes<c10::ArrayRef<double>, c10::List<double>>::test(
    c10::ArrayRef<double>(), [] (c10::ArrayRef<double> v) {EXPECT_EQ(0, v.size());},
    c10::List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    "(float[] a) -> float[]");
  testArgTypes<c10::ArrayRef<int64_t>, c10::List<int64_t>>::test(
    c10::ArrayRef<int64_t>(), [] (c10::ArrayRef<int64_t> v) {EXPECT_EQ(0, v.size());},
    c10::List<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[] a) -> int[]");
  testArgTypes<c10::ArrayRef<std::string>, c10::List<std::string>>::test(
    c10::ArrayRef<std::string>(), [] (c10::ArrayRef<std::string> v) {EXPECT_EQ(0, v.size());},
    c10::List<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    "(str[] a) -> str[]");


  // list types (with non-empty list)
  testArgTypes<c10::ArrayRef<double>, c10::List<double>>::test(
    c10::ArrayRef<double>({1.5, 2.5}), [] (c10::ArrayRef<double> v) {expectListEquals({1.5, 2.5}, v);},
    c10::List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    "(float[] a) -> float[]");
  testArgTypes<c10::ArrayRef<int64_t>, c10::List<int64_t>>::test(
    c10::ArrayRef<int64_t>({1, 2}), [] (c10::ArrayRef<int64_t> v) {expectListEquals({1, 2}, v);},
    c10::List<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[] a) -> int[]");
  testArgTypes<c10::ArrayRef<std::string>, c10::List<std::string>>::test(
    c10::ArrayRef<std::string>({"first", "second"}), [] (c10::ArrayRef<std::string> v) {expectListEquals({"first", "second"}, v);},
    c10::List<std::string>({"first", "second"}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toListRef().size());
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  testArgTypes<c10::ArrayRef<Tensor>, c10::List<Tensor>>::test(
    c10::ArrayRef<Tensor>({dummyTensor(c10::DispatchKey::CPUTensorId), dummyTensor(c10::DispatchKey::CUDATensorId)}), [] (c10::ArrayRef<Tensor> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v[0]));
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v[1]));
    },
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CUDATensorId), dummyTensor(c10::DispatchKey::CPUTensorId)}), [] (const IValue& v) {
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[] a) -> Tensor[]");


  // std::array list types (with empty list)
  testArgTypes<std::array<double, 0>>::test(
    std::array<double, 0>(), [] (std::array<double, 0> v) {},
    std::array<double, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<c10::List<double>>().size()));},
    "(float[0] a) -> float[0]");
  testArgTypes<std::array<int64_t, 0>>::test(
    std::array<int64_t, 0>(), [] (std::array<int64_t, 0> v) {},
    std::array<int64_t, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<c10::List<int64_t>>().size()));},
    "(int[0] a) -> int[0]");
  testArgTypes<std::array<bool, 0>>::test(
    std::array<bool, 0>(), [] (std::array<bool, 0> v) {},
    std::array<bool, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<std::array<bool, 0>>().size()));},
    "(bool[0] a) -> bool[0]");
  testArgTypes<std::array<std::string, 0>>::test(
    std::array<std::string, 0>(), [] (std::array<std::string, 0> v) {EXPECT_EQ(0, v.size());},
    std::array<std::string, 0>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    "(str[0] a) -> str[0]");


  // std::array list types (with non-empty list)
  testArgTypes<std::array<double, 2>>::test(
    std::array<double, 2>({1.5, 2.5}), [] (std::array<double, 2> v) {expectListEquals({1.5, 2.5}, v);},
    std::array<double, 2>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<std::array<double, 2>>());},
    "(float[2] a) -> float[2]");
  testArgTypes<std::array<int64_t, 2>>::test(
    std::array<int64_t, 2>({1, 2}), [] (std::array<int64_t, 2> v) {expectListEquals({1, 2}, v);},
    std::array<int64_t, 2>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<std::array<int64_t, 2>>());},
    "(int[2] a) -> int[2]");
  testArgTypes<std::array<bool, 2>>::test(
    std::array<bool, 2>({true, false}), [] (std::array<bool, 2> v) {expectListEquals({true, false}, v);},
    std::array<bool, 2>({true, false}), [] (const IValue& v) {expectListEquals({true, false}, v.to<std::array<bool, 2>>());},
    "(bool[2] a) -> bool[2]");
  testArgTypes<std::array<std::string, 2>>::test(
    std::array<std::string, 2>({"first", "second"}), [] (std::array<std::string, 2> v) {expectListEquals({"first", "second"}, v);},
    std::array<std::string, 2>({"first", "second"}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toListRef().size());
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[2] a) -> str[2]");
  testArgTypes<std::array<Tensor, 2>>::test(
    std::array<Tensor, 2>({dummyTensor(c10::DispatchKey::CPUTensorId), dummyTensor(c10::DispatchKey::CUDATensorId)}), [] (std::array<Tensor, 2> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v[0]));
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v[1]));
    },
    std::array<Tensor, 2>({dummyTensor(c10::DispatchKey::CUDATensorId), dummyTensor(c10::DispatchKey::CPUTensorId)}), [] (const IValue& v) {
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[2] a) -> Tensor[2]");


  // deprecated list types (with empty list)
  testArgTypes<std::vector<double>>::test<TestLegacyAPI>(
    std::vector<double>(), [] (const std::vector<double>& v) {EXPECT_EQ(0, v.size());},
    std::vector<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    "(float[] a) -> float[]");
  testArgTypes<std::vector<int64_t>>::test<TestLegacyAPI>(
    std::vector<int64_t>(), [] (const std::vector<int64_t>& v) {EXPECT_EQ(0, v.size());},
    std::vector<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[] a) -> int[]");
  //Note: vector<bool> is not supported, use List<bool> instead.
  testArgTypes<std::vector<std::string>>::test<TestLegacyAPI>(
    std::vector<std::string>(), [] (const std::vector<std::string>& v) {EXPECT_EQ(0, v.size());},
    std::vector<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    "(str[] a) -> str[]");


  // deprecated list types (with non-empty list)
  testArgTypes<std::vector<double>>::test<TestLegacyAPI>(
    std::vector<double>({1.5, 2.5}), [] (const std::vector<double>& v) {expectListEquals({1.5, 2.5}, v);},
    std::vector<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    "(float[] a) -> float[]");
  testArgTypes<std::vector<int64_t>>::test<TestLegacyAPI>(
    std::vector<int64_t>({1, 2}), [] (const std::vector<int64_t>& v) {expectListEquals({1, 2}, v);},
    std::vector<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[] a) -> int[]");
  //Note: vector<bool> is not supported, use List<bool> instead.
  testArgTypes<std::vector<std::string>>::test<TestLegacyAPI>(
    std::vector<std::string>({"first", "second"}), [] (const std::vector<std::string>& v) {expectListEquals({"first", "second"}, v);},
    std::vector<std::string>({"first", "second"}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toListRef().size());
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  testArgTypes<std::vector<Tensor>>::test<TestLegacyAPI>(
    std::vector<Tensor>({dummyTensor(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CUDA)}), [] (const std::vector<Tensor>& v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.at(0)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.at(1)));
    },
    std::vector<Tensor>({dummyTensor(c10::DispatchKey::CUDA), dummyTensor(c10::DispatchKey::CPU)}), [] (const IValue& v) {
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[] a) -> Tensor[]");

  // Test optional of list (with nullopt)
  testArgTypes<c10::optional<c10::List<int64_t>>>::test(
    c10::optional<c10::List<int64_t>>(c10::nullopt), [] (const c10::optional<c10::List<int64_t>>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<c10::List<int64_t>>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(int[]? a) -> int[]?");

  // Test optional of list (with empty list)
  testArgTypes<c10::optional<c10::List<int64_t>>>::test(
    c10::optional<c10::List<int64_t>>(c10::List<int64_t>({})), [] (const c10::optional<c10::List<int64_t>>& v) {EXPECT_EQ(0, v.value().size());},
    c10::optional<c10::List<int64_t>>(c10::List<int64_t>({})), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[]? a) -> int[]?");

  // Test optional of list (with values)
  testArgTypes<c10::optional<c10::List<int64_t>>>::test(
    c10::optional<c10::List<int64_t>>(c10::List<int64_t>({1, 2})), [] (const c10::optional<c10::List<int64_t>>& v) {expectListEquals({1, 2}, v.value());},
    c10::optional<c10::List<int64_t>>(c10::List<int64_t>({3, 4})), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[]? a) -> int[]?");

  // Test list of optional (with empty list)
  testArgTypes<c10::List<c10::optional<int64_t>>>::test(
    c10::List<c10::optional<int64_t>>(c10::List<c10::optional<int64_t>>({})), [] (const c10::List<c10::optional<int64_t>>& v) {EXPECT_EQ(0, v.size());},
    c10::List<c10::optional<int64_t>>(c10::List<c10::optional<int64_t>>({})), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<c10::optional<int64_t>>>().size());},
    "(int?[] a) -> int?[]");

  // Test list of optional (with values)
  testArgTypes<c10::List<c10::optional<int64_t>>>::test(
    c10::List<c10::optional<int64_t>>(c10::List<c10::optional<int64_t>>({3, c10::nullopt, 2})), [] (const c10::List<c10::optional<int64_t>>& v) {expectListEquals<c10::optional<int64_t>>({3, c10::nullopt, 2}, v);},
    c10::List<c10::optional<int64_t>>(c10::List<c10::optional<int64_t>>({3, c10::nullopt, 2})), [] (const IValue& v) {expectListEquals<c10::optional<int64_t>>({3, c10::nullopt, 2}, v.to<c10::List<c10::optional<int64_t>>>());},
    "(int?[] a) -> int?[]");

  // dict types
  c10::Dict<std::string, std::string> str_dict;
  str_dict.insert("key1", "value1");
  str_dict.insert("key2", "value2");
  testArgTypes<c10::Dict<std::string, std::string>>::test(
    str_dict, [] (c10::Dict<std::string, std::string> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ("value1", v.at("key1"));
      EXPECT_EQ("value2", v.at("key2"));
    },
    str_dict, [] (const IValue& v) {
      c10::Dict<std::string, std::string> dict = c10::impl::toTypedDict<std::string, std::string>(v.toGenericDict());
      EXPECT_EQ(2, dict.size());
      EXPECT_EQ("value1", dict.at("key1"));
      EXPECT_EQ("value2", dict.at("key2"));
    },
    "(Dict(str, str) a) -> Dict(str, str)");
  c10::Dict<int64_t, Tensor> tensor_dict;
  tensor_dict.insert(1, dummyTensor(c10::DispatchKey::CPU));
  tensor_dict.insert(2, dummyTensor(c10::DispatchKey::CUDA));
  testArgTypes<c10::Dict<int64_t, Tensor>>::test(
    tensor_dict, [] (c10::Dict<int64_t, Tensor> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.at(1)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.at(2)));
    },
    tensor_dict, [] (const IValue& v) {
      c10::Dict<int64_t, Tensor> dict = c10::impl::toTypedDict<int64_t, Tensor>(v.toGenericDict());
      EXPECT_EQ(2, dict.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(dict.at(1)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(dict.at(2)));
    },
    "(Dict(int, Tensor) a) -> Dict(int, Tensor)");

  // deprecated dict types
  std::unordered_map<std::string, std::string> str_map;
  str_map.emplace("key1", "value1");
  str_map.emplace("key2", "value2");
  testArgTypes<std::unordered_map<std::string, std::string>>::test<TestLegacyAPI>(
    str_map, [] (std::unordered_map<std::string, std::string> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ("value1", v.at("key1"));
      EXPECT_EQ("value2", v.at("key2"));
    },
    str_map, [] (const IValue& v) {
      c10::Dict<std::string, std::string> dict = c10::impl::toTypedDict<std::string, std::string>(v.toGenericDict());
      EXPECT_EQ(2, dict.size());
      EXPECT_EQ("value1", dict.at("key1"));
      EXPECT_EQ("value2", dict.at("key2"));
    },
    "(Dict(str, str) a) -> Dict(str, str)");
  std::unordered_map<int64_t, Tensor> tensor_map;
  tensor_map.emplace(1, dummyTensor(c10::DispatchKey::CPU));
  tensor_map.emplace(2, dummyTensor(c10::DispatchKey::CUDA));
  testArgTypes<std::unordered_map<int64_t, Tensor>>::test<TestLegacyAPI>(
    tensor_map, [] (std::unordered_map<int64_t, Tensor> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.at(1)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.at(2)));
    },
    tensor_map, [] (const IValue& v) {
      c10::Dict<int64_t, Tensor> dict = c10::impl::toTypedDict<int64_t, Tensor>(v.toGenericDict());
      EXPECT_EQ(2, dict.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(dict.at(1)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(dict.at(2)));
    },
    "(Dict(int, Tensor) a) -> Dict(int, Tensor)");

  // weird deeply nested type
  using DeeplyNestedType = c10::List<c10::Dict<std::string, c10::List<c10::optional<c10::Dict<int64_t, std::string>>>>>;
  auto makeDeeplyNestedObject = [] () -> DeeplyNestedType {
    c10::Dict<int64_t, std::string> inner3;
    inner3.insert(1, "1");
    c10::List<c10::optional<c10::Dict<int64_t, std::string>>> inner2;
    inner2.push_back(std::move(inner3));
    c10::Dict<std::string, c10::List<c10::optional<c10::Dict<int64_t, std::string>>>> inner1;
    inner1.insert("key", std::move(inner2));
    c10::List<c10::Dict<std::string, c10::List<c10::optional<c10::Dict<int64_t, std::string>>>>> result;
    result.push_back(inner1);
    return result;
  };
  testArgTypes<DeeplyNestedType>::test(
    makeDeeplyNestedObject(), [] (const DeeplyNestedType& v) {EXPECT_EQ("1", v.get(0).at("key").get(0).value().at(1));},
    makeDeeplyNestedObject(), [] (const IValue& v) {EXPECT_EQ("1", v.to<DeeplyNestedType>().get(0).at("key").get(0).value().at(1));},
    "(Dict(str, Dict(int, str)?[])[] a) -> Dict(str, Dict(int, str)?[])[]");
}

TEST(NewOperatorRegistrationTest, testBasics) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("dummy(Tensor self) -> Tensor");
  m.def("dummy1(Tensor self) -> Tensor");
  m.def("dummy2(Tensor self) -> Tensor");
  m.def("dummy3(Tensor self, Tensor other) -> Tensor", [](const Tensor& self, const Tensor& other) { return self; });
  m.def("dummy4", [](const Tensor& self, const Tensor& other) { return other; });
  m.impl("dummy", c10::DeviceType::CPU, [](const Tensor& self) { return self; });
  m.impl("dummy", c10::DeviceType::XLA, [](const Tensor& self) { return self; });
  // Internal API
  m.impl("dummy2", c10::DispatchKey::CPU, [](const Tensor& self) { return self; });
  m.impl("dummy2", c10::DispatchKey::XLA, [](const Tensor& self) { return self; });

  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy", ""}).has_value());
  // Should have a schema even if there are no impls
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy3", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy4", ""}).has_value());
}

TEST(NewOperatorRegistrationTest, importTopLevel) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("def1(Tensor self) -> Tensor");
  m.def("def2(Tensor self) -> Tensor", [](const Tensor& x) { return x; });
  m.def("def3", [](const Tensor& x) { return x; });

  auto m2 = MAKE_TORCH_LIBRARY_IMPL(test, CatchAll);
  m2.impl("impl1", [](const Tensor& x) { return x; });

  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def3", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::impl1", ""}).has_value());
}

TEST(NewOperatorRegistrationTest, overload) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn(Tensor self) -> Tensor");
  m.def("fn.overload1(Tensor self, Tensor other) -> Tensor");
  m.def("fn.overload2(Tensor self, Tensor other, Tensor alpha) -> Tensor");

  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", "overload1"}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", "overload2"}).has_value());
}

TEST(NewOperatorRegistrationTest, importNamespace) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("def1(Tensor self) -> Tensor");
  m.def("def2(Tensor self) -> Tensor", [](const Tensor& x) { return x; });
  m.def("def3", [](const Tensor& x) { return x; });
  m.impl("impl1", [](const Tensor& x) { return x; });
  expectThrows<c10::Error>([&] {
    m.def("retest::def1(Tensor self) -> Tensor");
  }, "");

  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::impl1", ""}).has_value());
}

TEST(NewOperatorRegistrationTest, schema) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("def1(Tensor self) -> Tensor");
  m.def(torch::schema("def2(Tensor self) -> Tensor"));
  m.def(torch::schema("def3(Tensor self) -> Tensor", c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(torch::jit::parseSchema("def4(Tensor self) -> Tensor"));

  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4", ""}).has_value());

  EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def1", ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::FROM_SCHEMA);
  EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def2", ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::FROM_SCHEMA);
  EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def3", ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::PURE_FUNCTION);
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4", ""})->schema().isDefaultAliasAnalysisKind());
}

TEST(NewOperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndCatchallKernelForSameBackend_thenCallsFallbackKernel) {
  auto m1 = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
  m1.fallback(CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());

  bool called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn(Tensor t, str input) -> ()");
  m.impl("fn", [&] (Tensor, std::string) { called = true; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  // CatchAll now maps to Math and has higher precedence than backend fallback.
  EXPECT_TRUE(called);
}

TEST(NewOperatorRegistrationTest, whenRegisteringAutogradKernelWithRegularKernel_thenCanCallRegularKernel) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn(Tensor dummy) -> ()");
  m.impl("fn", c10::DispatchKey::CPU, nonautograd_kernel);
  m.impl("fn", c10::DispatchKey::Autograd, autograd_kernel);

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  called_nonautograd = called_autograd = false;
  callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);
}

TEST(NewOperatorRegistrationTest, dispatchWithMathKernel) {
  bool math_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::Math, [&](const Tensor& x) { math_called = true; return x; }));

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    ASSERT_FALSE(math_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(math_called);
  }

  {
    math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(math_called);
  }

  {
    math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::XLA));
    ASSERT_TRUE(math_called);
  }

  {
    math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::XLA, /*requires_grad=*/true));
    ASSERT_TRUE(math_called);
  }

  {
    math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU));
    ASSERT_TRUE(math_called);
  }

  {
    math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU, /*requires_grad=*/true));
    ASSERT_TRUE(math_called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchWithMathAndAutogradKernel) {
  bool math_called = false;
  bool autograd_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::Math, [&](const Tensor& x) { math_called = true; return x; }));
  m.impl("fn", c10::DispatchKey::Autograd, [&](const Tensor& x) { autograd_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  // Math has higher precedence than Autograd
  {
    math_called = autograd_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(autograd_called);
  }

  {
    math_called = autograd_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(autograd_called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchWithMathAndCatchAllKernel) {
  bool math_called = false;
  bool catchall_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::Math, [&](const Tensor& x) { math_called = true; return x; }));
  m.impl("fn", [&](const Tensor& x) { catchall_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  // catchAll now maps to Math, which means we have two registrations to Math key.
  // The last registration is used.
  {
    catchall_called = math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_FALSE(math_called);
    ASSERT_TRUE(catchall_called);
  }

  {
    catchall_called = math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_FALSE(math_called);
    ASSERT_TRUE(catchall_called);
  }
}

TEST(NewOperatorRegistrationTest, AutogradBackendOverridesMathKernel) {
  bool math_called = false;
  bool autograd_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::Math, [&](const Tensor& x) { math_called = true; return x; }));
  m.impl("fn", c10::DispatchKey::AutogradCPU, [&](const Tensor& x) { autograd_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    math_called = autograd_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(autograd_called);
  }

  {
    math_called = autograd_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
    ASSERT_FALSE(math_called);
  }

  {
    math_called = autograd_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(autograd_called);
  }

  {
    math_called = autograd_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(autograd_called);
  }
}

TEST(NewOperatorRegistrationTest, BackendOverridesMathKernel) {
  bool math_called = false;
  bool backend_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::Math, [&](const Tensor& x) { math_called = true; return x; }));
  m.impl("fn", c10::DispatchKey::CPU, [&](const Tensor& x) { backend_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    math_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(math_called);
  }

  {
    // Fallthrough AutogradCPU and reaches CPU
    math_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(math_called);
  }

  {
    math_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(backend_called);
  }

  {
    math_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(backend_called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchWithDefaultBackendKernel) {
  bool called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::DefaultBackend, [&](const Tensor& x) { called = true; return x; }));

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    ASSERT_FALSE(called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(called);
  }

  {
    called = false;
    // AutogradCPU is fallthrough, calls CPU kernel
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(called);
  }

  {
    called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::XLA));
    ASSERT_TRUE(called);
  }

  {
    called = false;
    // AutogradXLA is fallthrough, calls XLA kernel
    callOp(*op, dummyTensor(c10::DispatchKey::XLA, /*requires_grad=*/true));
    ASSERT_TRUE(called);
  }

  {
    called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU));
    ASSERT_TRUE(called);
  }

  {
    called = false;
    // AutogradCPU is fallthrough, calls CPU kernel
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU, /*requires_grad=*/true));
    ASSERT_TRUE(called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchWithDefaultBackendAndMathKernel) {
  bool backend_called = false;
  bool math_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::DefaultBackend, [&](const Tensor& x) { backend_called = true; return x; }));
  m.impl("fn", c10::DispatchKey::Math, [&](const Tensor& x) { math_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    backend_called = math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(math_called);
  }

  {
    backend_called = math_called = false;
    // AutogradCPU is fallthrough, calls CPU kernel
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_FALSE(math_called);
    ASSERT_TRUE(backend_called);
  }

  {
    backend_called = math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::XLA));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(math_called);
  }

  {
    backend_called = math_called = false;
    // AutogradXLA is fallthrough, calls XLA kernel
    callOp(*op, dummyTensor(c10::DispatchKey::XLA, /*requires_grad=*/true));
    ASSERT_FALSE(math_called);
    ASSERT_TRUE(backend_called);
  }

  {
    backend_called = math_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(math_called);
  }

  {
    backend_called = math_called = false;
    // AutogradOther is fallthrough, calls SparseCPU kernel
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU, /*requires_grad=*/true));
    ASSERT_FALSE(math_called);
    ASSERT_TRUE(backend_called);
  }
}

TEST(NewOperatorRegistrationTest, BackendOverridesDefaultBackendKernel) {
  bool default_called = false;
  bool backend_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::DefaultBackend, [&](const Tensor& x) { default_called = true; return x; }));
  m.impl("fn", c10::DispatchKey::CPU, [&](const Tensor& x) { backend_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    default_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(default_called);
  }

  {
    default_called = backend_called = false;
    // AutogradCPU is fallthrough, calls CPU kernel
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(default_called);
  }

  {
    default_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    ASSERT_TRUE(default_called);
    ASSERT_FALSE(backend_called);
  }

  {
    default_called = backend_called = false;
    // AutogradCUDA is fallthrough, calls CUDA kernel
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    ASSERT_TRUE(default_called);
    ASSERT_FALSE(backend_called);
  }
}


TEST(NewOperatorRegistrationTest, dispatch) {
  bool cpu_called = false;
  bool cuda_called = false;
  bool autograd_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn_cpu", torch::dispatch(c10::DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; }));
  m.def("fn_cuda", torch::dispatch(c10::kCUDA, [&](const Tensor& x) { cuda_called = true; return x; }));
  m.def("fn_autograd", torch::dispatch(c10::kAutograd, [&](const Tensor& x) { autograd_called = true; return x; }));

  {
    auto op = Dispatcher::singleton().findSchema({"test::fn_cpu", ""});
    ASSERT_TRUE(op.has_value());
    ASSERT_FALSE(cpu_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(cpu_called);
  }

  {
    auto op = Dispatcher::singleton().findSchema({"test::fn_cuda", ""});
    ASSERT_TRUE(op.has_value());
    ASSERT_FALSE(cuda_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    ASSERT_TRUE(cuda_called);
  }

  {
    auto op = Dispatcher::singleton().findSchema({"test::fn_autograd", ""});
    ASSERT_TRUE(op.has_value());
    ASSERT_FALSE(autograd_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
  }

  {
    autograd_called = false;
    auto op = Dispatcher::singleton().findSchema({"test::fn_autograd", ""});
    ASSERT_TRUE(op.has_value());
    callOp(*op, dummyTensor(c10::DispatchKey::XLA, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchAutogradPrecedence) {
  bool cpu_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; }));

  {
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    ASSERT_FALSE(cpu_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(cpu_called);
  }

  {
    // AutogradCPU is fallthrough, use CPU kernel
    cpu_called = false;
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(cpu_called);
  }

  bool autograd_called = false;
  m.impl("fn", c10::kAutograd, [&](const Tensor& x) { autograd_called = true; return x; });

  {
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
  }

  // Autograd backend kernel has higher precedence than Autograd alias.
  bool autogradcpu_called = false;
  m.impl("fn", c10::DispatchKey::AutogradCPU, [&](const Tensor& x) { autogradcpu_called = true; return x; });

  {
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autogradcpu_called);
  }
}

TEST(NewOperatorRegistrationTest, throwsWhenRegisterToBackendMapsToAutogradOther) {
  bool sparsecpu_called, math_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::SparseCPU, [&](const Tensor& x) { sparsecpu_called = true; return x; }));
  m.impl("fn", c10::DispatchKey::Math, [&](const Tensor& x) { math_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU));
    ASSERT_TRUE(sparsecpu_called);
  }

  {
    expectThrows<c10::Error>([&] {
      callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU, /*requires_grad=*/true));
    }, "test::fn has kernels registered to both Math and a backend mapped to AutogradOther.");
  }
}

TEST(NewOperatorRegistrationTest, dispatchMultipleTensors) {
  bool privateuse1_called = false;
  bool catchall_called = false;
  // Similar to in-tree AutogradCPU/AutogradCUDA etc, out-of-tree backends usually register
  // a fallthrough kernel for AutogradPrivateUse1.
  auto m1 = MAKE_TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1);
  m1.fallback(CppFunction::makeFallthrough());

  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn", torch::dispatch(c10::DispatchKey::PrivateUse1, [&](const Tensor& x, const Tensor& y) { privateuse1_called = true; return x; }));
  m.impl("fn", [&](const Tensor& x, const Tensor& y) { catchall_called = true; return x; });

  {
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    callOp(*op, dummyTensor(c10::DispatchKey::PrivateUse1), dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(privateuse1_called);
  }

  {
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    ASSERT_FALSE(catchall_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(catchall_called);
  }

  {
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    catchall_called = false;
    callOp(*op,
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true),
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(catchall_called);
  }

  {
    // TODO(#43908): currently this will fallthrough AutogradPrivateUse1 then call catchall kernel
    // at AutogradCPU, while backend extenders are indeed expecting to call PrivateUse1 kernel.
    // This confusing behavior is caused by we registering fallthrough as backend fallback for
    // Autograd keys. Note users could always work around this by registering the same kernel to
    // AutogradPrivateUse1 as shown below until we support it.
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    catchall_called = false;
    callOp(*op,
           dummyTensor(c10::DispatchKey::PrivateUse1, /*requires_grad=*/true),
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(catchall_called);
  }

  m.impl("fn", c10::DispatchKey::AutogradPrivateUse1, [&](const Tensor& x, const Tensor& y) { privateuse1_called = true; return x; });

  {
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    privateuse1_called = false;
    callOp(*op,
           dummyTensor(c10::DispatchKey::PrivateUse1, /*requires_grad=*/true),
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(privateuse1_called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchMultiple) {
  bool cpu_called = false;
  bool cuda_called = false;
  bool autograd_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn(Tensor self) -> Tensor");
  // NB: Direct use of DispatchKey is discouraged; use the DeviceType
  // k-synonyms instead
  m.impl("fn", c10::DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; });
  m.impl("fn", c10::kCUDA, [&](const Tensor& x) { cuda_called = true; return x; });
  m.impl("fn", c10::kAutograd, [&](const Tensor& x) { autograd_called = true; return x; });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    ASSERT_FALSE(cpu_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(cpu_called);

    ASSERT_FALSE(cuda_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    ASSERT_TRUE(cuda_called);
  }

  {
    ASSERT_FALSE(autograd_called);
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);

    autograd_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
  }
}

TEST(NewOperatorRegistrationTest, fallback) {
  auto m = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
  m.fallback(CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_EQ("hello _test::dummy", stack[1].toString()->string());
}

TEST(NewOperatorRegistrationTest, BackendSelectRedispatchesToCPU) {
  bool cpu_called = false;
  bool backend_generic_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  auto after_backend_select = c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::BackendSelect);
  m.def("fn(Tensor self) -> Tensor");
  m.impl("fn", c10::kCPU, [&](const Tensor& x) { cpu_called = true; return x; });
  m.impl("fn", c10::DispatchKey::BackendSelect, [&](c10::DispatchKeySet ks, const Tensor& x) {
     backend_generic_called = true;
     auto op = c10::Dispatcher::singleton().findSchema({"test::fn", ""}).value().typed<Tensor (const Tensor&)>();
     return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor&>(op, ks & after_backend_select, x);
   });

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  ASSERT_TRUE(cpu_called);
  ASSERT_TRUE(backend_generic_called);
}

TEST(NewOperatorRegistrationTest, TorchLibraryTwiceIsError) {
  {
    auto m = MAKE_TORCH_LIBRARY(test);
    expectThrows<c10::Error>([] {
      auto m2 = MAKE_TORCH_LIBRARY(test);
    }, "Only a single TORCH_LIBRARY");
  }
  // Ensure it's ok after deregistering
  auto m = MAKE_TORCH_LIBRARY(test);
}

Tensor dummy_fn(const Tensor& x) {
  return x;
}

TEST(NewOperatorRegistrationTest, CppFunction) {
  // Just show off the possible ways to register functions
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn1", &dummy_fn);
  // C++ will implicitly convert function to function pointer
  // c.f. https://en.cppreference.com/w/cpp/language/implicit_conversion#Function_to_pointer
  m.def("fn2", dummy_fn);
  m.def("fn3", [](const Tensor& x) { return x; });
  // These require explicit schema
  m.def("fn4(Tensor x) -> Tensor", CppFunction::makeFallthrough());
  m.def("fn5(Tensor x) -> Tensor", CppFunction::makeFromUnboxedFunction(dummy_fn));
  m.def("fn6(Tensor x) -> Tensor", CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());
}

// Some internal tests that have to be done from C++

struct OpRegistrationListenerForDelayedListenerTest : public c10::OpRegistrationListener {
  int64_t num_registers_ = 0;
  int64_t num_deregisters_ = 0;
  void onOperatorRegistered(const OperatorHandle& op) override {
    num_registers_++;
  }
  void onOperatorDeregistered(const OperatorHandle& op) override {
    num_deregisters_++;
  }
};

TEST(NewOperatorRegistrationTest, testDelayedListener) {
  auto listener = std::make_unique<OpRegistrationListenerForDelayedListenerTest>();
  auto listener_ptr = listener.get();
  auto registry = Dispatcher::singleton().addRegistrationListener(std::move(listener));
  int64_t initial_num_registers = listener_ptr->num_registers_;
  int64_t initial_num_deregisters = listener_ptr->num_deregisters_;
  auto op = Dispatcher::singleton().findOp({"_test::dummy", ""});
  ASSERT_FALSE(op.has_value());
  auto m1 = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
  m1.impl("dummy", [](const Tensor& self) { return self; });
  EXPECT_EQ(initial_num_registers, listener_ptr->num_registers_);
  {
    auto m2 = MAKE_TORCH_LIBRARY(_test);
    m2.def("dummy(Tensor self) -> Tensor");
    EXPECT_EQ(initial_num_registers + 1, listener_ptr->num_registers_);
  }
  EXPECT_EQ(initial_num_deregisters + 1, listener_ptr->num_deregisters_);
}

TEST(NewOperatorRegistrationTest, testImplNoDefGetsCaught) {
  auto danglingImpls = Dispatcher::singleton().findDanglingImpls();
  std::string error_str = "Discovered operators that have been registered through the dispatcher"
                          " without explicitly specifying their schemas. Please do so using"
                          " the TORCH_LIBRARY macro. Suspect operators:\n";
  for (auto& op : danglingImpls) {
      auto& op_name = op.operator_name();
      error_str += "\t" + op_name.name;
      if (op_name.overload_name != "") {
          error_str += "." + op_name.overload_name;
      }
      error_str += "\n";
  }
  ASSERT_EQ(danglingImpls.size(), 0) << error_str;
}

bool called_kernel_cpu = false;
bool called_kernel_autograd = false;
bool called_kernel_tracing = false;

void cpu_kernel(Tensor) {
  called_kernel_cpu = true;
}

// autograd kernel that redispatches. Explicitly takes in and updates the DispatchKeySet
void autograd_kernel_redispatching_with_DispatchKeySet(c10::DispatchKeySet ks, Tensor a) {
  called_kernel_autograd = true;
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  auto updatedDispatchKeySet = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
}

// autograd kernel that redispatches. Does not take in a DispatchKeySet
void autograd_kernel_redispatching_without_DispatchKeySet(c10::DispatchKeySet ks, Tensor a) {
  called_kernel_autograd = true;
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  auto updatedDispatchKeySet = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
}

// tracing kernel that redispatches. Explicitly takes in and updates the DispatchKeySet
void tracing_kernel_redispatching_with_DispatchKeySet(c10::DispatchKeySet ks, Tensor a) {
  called_kernel_tracing = true;
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  auto updatedDispatchKeySet = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer);
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
}

TEST(OperatorRegistrationTest, callKernelsWithDispatchKeySetConvention_call_redispatchesToLowerPriorityKernels) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn(Tensor dummy) -> ()");
  m.impl("fn", c10::DispatchKey::CPU, cpu_kernel);
  m.impl("fn", c10::DispatchKey::AutogradCPU, autograd_kernel_redispatching_with_DispatchKeySet);
  m.impl("fn", c10::DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
  auto tracing_autograd_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::Tracer)
                                    .add(c10::DispatchKey::AutogradCPU)
                                    .add(c10::DispatchKey::CPU);

  // call Tracing -> call Autograd -> call CPU
  callOpUnboxed<void, Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
  EXPECT_TRUE(called_kernel_tracing);
  EXPECT_TRUE(called_kernel_autograd);
  EXPECT_TRUE(called_kernel_cpu);
}

TEST(OperatorRegistrationTest, callKernelsWithDispatchKeySetConvention_callBoxed_redispatchesToLowerPriorityKernels) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn(Tensor dummy) -> ()");
  m.impl("fn", c10::DispatchKey::CPU, cpu_kernel);
  m.impl("fn", c10::DispatchKey::AutogradCPU, autograd_kernel_redispatching_with_DispatchKeySet);
  m.impl("fn", c10::DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
  auto tracing_autograd_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::Tracer)
                                    .add(c10::DispatchKey::AutogradCPU)
                                    .add(c10::DispatchKey::CPU);

  // call Tracing -> call Autograd -> call CPU
  callOp<Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
  EXPECT_TRUE(called_kernel_tracing);
  EXPECT_TRUE(called_kernel_autograd);
  EXPECT_TRUE(called_kernel_cpu);
}

TEST(OperatorRegistrationTest, callKernelsWithDispatchKeySetConvention_mixedCallingConventions_redispatchesToLowerPriorityKernels) {
  auto m = MAKE_TORCH_LIBRARY(test);
  m.def("fn(Tensor dummy) -> ()");
  m.impl("fn", c10::DispatchKey::CPU, cpu_kernel);
  // the tracing kernel takes in a DispatchKeySet, but the autograd kernel does not
  // the dispatcher should handle correctly plumbing its DispatchKeySet to tracing and not autograd.
  m.impl("fn", c10::DispatchKey::AutogradCPU, autograd_kernel_redispatching_without_DispatchKeySet);
  m.impl("fn", c10::DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
  auto tracing_autograd_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::Tracer)
                                    .add(c10::DispatchKey::AutogradCPU)
                                    .add(c10::DispatchKey::CPU);

  // call Tracing -> call Autograd -> call CPU
  callOpUnboxed<void, Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
  EXPECT_TRUE(called_kernel_tracing);
  EXPECT_TRUE(called_kernel_autograd);
  EXPECT_TRUE(called_kernel_cpu);
}

}

#pragma GCC diagnostic pop
