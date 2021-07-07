#include <ATen/Parallel.h>
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <future>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

void compare_torchpy_jit(const char* model_filename, const char* jit_filename) {
  // Test
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.load_package(model_filename);
  auto model = p.load_pickle("model", "model.pkl");
  at::IValue eg;
  {
    auto I = p.acquire_session();
    eg = I.self.attr("load_pickle")({"model", "example.pkl"}).toIValue();
  }

  at::Tensor output = model(eg.toTuple()->elements()).toTensor();

  // Reference
  auto ref_model = torch::jit::load(jit_filename);
  at::Tensor ref_output =
      ref_model.forward(eg.toTuple()->elements()).toTensor();

  ASSERT_TRUE(ref_output.allclose(output, 1e-03, 1e-05));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char* simple = "torch/csrc/deploy/example/generated/simple";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char* simple_jit = "torch/csrc/deploy/example/generated/simple_jit";

const char* path(const char* envname, const char* path) {
  const char* e = getenv(envname);
  return e ? e : path;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, LoadLibrary) {
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.load_package(
      path("LOAD_LIBRARY", "torch/csrc/deploy/example/generated/load_library"));
  auto model = p.load_pickle("fn", "fn.pkl");
  model({});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, SimpleModel) {
  compare_torchpy_jit(path("SIMPLE", simple), path("SIMPLE_JIT", simple_jit));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, ResNet) {
  compare_torchpy_jit(
      path("RESNET", "torch/csrc/deploy/example/generated/resnet"),
      path("RESNET_JIT", "torch/csrc/deploy/example/generated/resnet_jit"));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, Movable) {
  torch::deploy::InterpreterManager m(1);
  torch::deploy::ReplicatedObj obj;
  {
    auto I = m.acquire_one();
    auto model =
        I.global("torch.nn", "Module")(std::vector<torch::deploy::Obj>());
    obj = I.create_movable(model);
  }
  obj.acquire_session();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, MultiSerialSimpleModel) {
  torch::deploy::InterpreterManager manager(3);
  torch::deploy::Package p = manager.load_package(path("SIMPLE", simple));
  auto model = p.load_pickle("model", "model.pkl");
  auto ref_model = torch::jit::load(path("SIMPLE_JIT", simple_jit));

  auto input = torch::ones({10, 20});
  size_t ninterp = 3;
  std::vector<at::Tensor> outputs;

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (const auto i : c10::irange(ninterp)) {
    outputs.push_back(model({input.alias()}).toTensor());
  }

  // Generate reference
  auto ref_output = ref_model.forward({input.alias()}).toTensor();

  // Compare all to reference
  for (const auto i : c10::irange(ninterp)) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }

  // test kwargs api with args
  std::vector<c10::IValue> args;
  args.emplace_back(input);
  std::unordered_map<std::string, c10::IValue> kwargs_empty;
  auto jit_output_args = model.call_kwargs(args, kwargs_empty).toTensor();
  ASSERT_TRUE(ref_output.equal(jit_output_args));

  // and with kwargs only
  std::unordered_map<std::string, c10::IValue> kwargs;
  kwargs["input"] = input;
  auto jit_output_kwargs = model.call_kwargs(kwargs).toTensor();
  ASSERT_TRUE(ref_output.equal(jit_output_kwargs));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, ThreadedSimpleModel) {
  size_t nthreads = 3;
  torch::deploy::InterpreterManager manager(nthreads);

  torch::deploy::Package p = manager.load_package(path("SIMPLE", simple));
  auto model = p.load_pickle("model", "model.pkl");
  auto ref_model = torch::jit::load(path("SIMPLE_JIT", simple_jit));

  auto input = torch::ones({10, 20});

  std::vector<at::Tensor> outputs;

  std::vector<std::future<at::Tensor>> futures;
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (const auto i : c10::irange(nthreads)) {
    futures.push_back(std::async(std::launch::async, [&model]() {
      auto input = torch::ones({10, 20});
      for (const auto i : c10::irange(100)) {
        model({input.alias()}).toTensor();
      }
      auto result = model({input.alias()}).toTensor();
      return result;
    }));
  }
  for (const auto i : c10::irange(nthreads)) {
    outputs.push_back(futures[i].get());
  }

  // Generate reference
  auto ref_output = ref_model.forward({input.alias()}).toTensor();

  // Compare all to reference
  for (const auto i : c10::irange(nthreads)) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, ThrowsSafely) {
  // See explanation in deploy.h
  torch::deploy::InterpreterManager manager(3);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(manager.load_package("some garbage path"), c10::Error);

  torch::deploy::Package p = manager.load_package(path("SIMPLE", simple));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(p.load_pickle("some other", "garbage path"), c10::Error);

  auto model = p.load_pickle("model", "model.pkl");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(model(at::IValue("unexpected input")), c10::Error);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, AcquireMultipleSessionsInTheSamePackage) {
  torch::deploy::InterpreterManager m(1);

  torch::deploy::Package p = m.load_package(path("SIMPLE", simple));
  auto I = p.acquire_session();

  auto I1 = p.acquire_session();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, AcquireMultipleSessionsInDifferentPackages) {
  torch::deploy::InterpreterManager m(1);

  torch::deploy::Package p = m.load_package(path("SIMPLE", simple));
  auto I = p.acquire_session();

  torch::deploy::Package p1 = m.load_package(
      path("RESNET", "torch/csrc/deploy/example/generated/resnet"));
  auto I1 = p1.acquire_session();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, TensorSharingNotAllowed) {
  size_t nthreads = 2;
  torch::deploy::InterpreterManager m(nthreads);
  // generate a tensor from one interpreter
  auto I0 = m.all_instances()[0].acquire_session();
  auto I1 = m.all_instances()[1].acquire_session();
  auto obj = I0.global("torch", "empty")({I0.from_ivalue(2)});
  auto t = obj.toIValue().toTensor();
  // try to feed it to the other interpreter, should error
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(I1.global("torch", "sigmoid")({t}), c10::Error);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, TaggingRace) {
  // At time of writing, this takes about 7s to run on DEBUG=1.  I think
  // this is OK, but feel free to fiddle with the knobs here to reduce the
  // runtime
  constexpr int64_t trials = 4;
  constexpr int64_t nthreads = 16;
  torch::deploy::InterpreterManager m(nthreads);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (const auto n : c10::irange(trials)) {
    at::Tensor t = torch::empty(2);
    std::atomic<int64_t> success(0);
    std::atomic<int64_t> failed(0);
    at::parallel_for(0, nthreads, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        auto I = m.all_instances()[i].acquire_session();
        try {
          I.from_ivalue(t);
          success++;
        } catch (const c10::Error& e) {
          failed++;
        }
      }
    });
    ASSERT_EQ(success, 1);
    ASSERT_EQ(failed, nthreads - 1);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, DisarmHook) {
  at::Tensor t = torch::empty(2);
  {
    torch::deploy::InterpreterManager m(1);
    auto I = m.acquire_one();
    I.from_ivalue(t);
  } // unload the old interpreter
  torch::deploy::InterpreterManager m(1);
  auto I = m.acquire_one();
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(I.from_ivalue(t), c10::Error); // NOT a segfault
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchpyTest, RegisterModule) {
  torch::deploy::InterpreterManager m(2);
  m.register_module_source("foomodule", "def add1(x): return x + 1\n");
  for (const auto& interp : m.all_instances()) {
    auto I = interp.acquire_session();
    AT_ASSERT(3 == I.global("foomodule", "add1")({2}).toIValue().toInt());
  }
}
