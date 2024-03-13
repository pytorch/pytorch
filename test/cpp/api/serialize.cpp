#include <gtest/gtest.h>

#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/tempfile.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace torch::test;
using namespace torch::nn;
using namespace torch::optim;

namespace {
Sequential xor_model() {
  return Sequential(
      Linear(2, 8),
      Functional(at::sigmoid),
      Linear(8, 1),
      Functional(at::sigmoid));
}

torch::Tensor save_and_load(torch::Tensor input) {
  std::stringstream stream;
  torch::save(input, stream);
  torch::Tensor tensor;
  torch::load(tensor, stream);
  return tensor;
}
} // namespace

template <typename DerivedOptions>
void is_optimizer_param_group_equal(
    const OptimizerParamGroup& lhs,
    const OptimizerParamGroup& rhs) {
  const auto& lhs_params = lhs.params();
  const auto& rhs_params = rhs.params();

  ASSERT_TRUE(lhs_params.size() == rhs_params.size());
  for (const auto j : c10::irange(lhs_params.size())) {
    ASSERT_TRUE(torch::equal(lhs_params[j], rhs_params[j]));
  }
  ASSERT_TRUE(
      static_cast<const DerivedOptions&>(lhs.options()) ==
      static_cast<const DerivedOptions&>(rhs.options()));
}

template <typename DerivedOptimizerParamState>
void is_optimizer_state_equal(
    const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
        lhs_state,
    const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
        rhs_state) {
  ASSERT_TRUE(lhs_state.size() == rhs_state.size());
  for (const auto& value : lhs_state) {
    auto found = rhs_state.find(value.first);
    ASSERT_TRUE(found != rhs_state.end());
    const DerivedOptimizerParamState& lhs_curr_state =
        static_cast<const DerivedOptimizerParamState&>(*(value.second.get()));
    const DerivedOptimizerParamState& rhs_curr_state =
        static_cast<const DerivedOptimizerParamState&>(*(found->second.get()));
    ASSERT_TRUE(lhs_curr_state == rhs_curr_state);
  }
}

template <
    typename OptimizerClass,
    typename DerivedOptimizerOptions,
    typename DerivedOptimizerParamState>
void test_serialize_optimizer(
    DerivedOptimizerOptions options,
    bool only_has_global_state = false) {
  torch::manual_seed(0);
  auto model1 = Linear(5, 2);
  auto model2 = Linear(5, 2);
  auto model3 = Linear(5, 2);

  // Models 1, 2, 3 will have the same parameters.
  auto model_tempfile = c10::make_tempfile();
  torch::save(model1, model_tempfile.name);
  torch::load(model2, model_tempfile.name);
  torch::load(model3, model_tempfile.name);

  auto param1 = model1->named_parameters();
  auto param2 = model2->named_parameters();
  auto param3 = model3->named_parameters();
  for (const auto& p : param1) {
    ASSERT_TRUE(p->allclose(param2[p.key()]));
    ASSERT_TRUE(param2[p.key()].allclose(param3[p.key()]));
  }
  // Make some optimizers
  auto optim1 = OptimizerClass(
      {torch::optim::OptimizerParamGroup(model1->parameters())}, options);
  auto optim2 = OptimizerClass(model2->parameters(), options);
  auto optim2_2 = OptimizerClass(model2->parameters(), options);
  auto optim3 = OptimizerClass(model3->parameters(), options);
  auto optim3_2 = OptimizerClass(model3->parameters(), options);

  auto x = torch::ones({10, 5});

  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    auto closure = []() { return torch::tensor({10}); };
    optimizer.step(closure);
  };

  // Do 2 steps of model1
  step(optim1, model1);
  step(optim1, model1);

  // Do 2 steps of model 2 without saving the optimizer
  step(optim2, model2);
  step(optim2_2, model2);

  // Do 1 step of model 3
  step(optim3, model3);

  // save the optimizer
  auto optim_tempfile = c10::make_tempfile();
  torch::save(optim3, optim_tempfile.name);
  torch::load(optim3_2, optim_tempfile.name);

  auto& optim3_2_param_groups = optim3_2.param_groups();
  auto& optim3_param_groups = optim3.param_groups();
  auto& optim3_2_state = optim3_2.state();
  auto& optim3_state = optim3.state();

  // optim3_2 and optim1 should have param_groups and state of size 1 and
  // state_size respectively
  ASSERT_TRUE(optim3_2_param_groups.size() == 1);
  // state_size = 2 for all optimizers except LBFGS as LBFGS only maintains one
  // global state
  unsigned state_size = only_has_global_state ? 1 : 2;
  ASSERT_TRUE(optim3_2_state.size() == state_size);

  // optim3_2 and optim1 should have param_groups and state of same size
  ASSERT_TRUE(optim3_2_param_groups.size() == optim3_param_groups.size());
  ASSERT_TRUE(optim3_2_state.size() == optim3_state.size());

  // checking correctness of serialization logic for optimizer.param_groups_ and
  // optimizer.state_
  for (const auto i : c10::irange(optim3_2_param_groups.size())) {
    is_optimizer_param_group_equal<DerivedOptimizerOptions>(
        optim3_2_param_groups[i], optim3_param_groups[i]);
    is_optimizer_state_equal<DerivedOptimizerParamState>(
        optim3_2_state, optim3_state);
  }

  // Do step2 for model 3
  step(optim3_2, model3);

  param1 = model1->named_parameters();
  param2 = model2->named_parameters();
  param3 = model3->named_parameters();
  for (const auto& p : param1) {
    const auto& name = p.key();
    // Model 1 and 3 should be the same
    ASSERT_TRUE(
        param1[name].norm().item<float>() == param3[name].norm().item<float>());
    ASSERT_TRUE(
        param1[name].norm().item<float>() != param2[name].norm().item<float>());
  }
}

/// Utility function to save a value of `int64_t` type.
void write_int_value(
    torch::serialize::OutputArchive& archive,
    const std::string& key,
    const int64_t& value) {
  archive.write(key, c10::IValue(value));
}
// Utility function to save a vector of buffers.
template <typename BufferContainer>
void write_tensors_to_archive(
    torch::serialize::OutputArchive& archive,
    const std::string& key,
    const BufferContainer& buffers) {
  archive.write(
      key + "/size", torch::tensor(static_cast<int64_t>(buffers.size())));
  for (const auto index : c10::irange(buffers.size())) {
    archive.write(
        key + "/" + std::to_string(index), buffers[index], /*is_buffer=*/true);
  }
}

// Utility function to save a vector of step buffers.
void write_step_buffers(
    torch::serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<int64_t>& steps) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(steps.size());
  for (const auto& step : steps) {
    tensors.push_back(torch::tensor(static_cast<int64_t>(step)));
  }
  write_tensors_to_archive(archive, key, tensors);
}

#define OLD_SERIALIZATION_LOGIC_WARNING_CHECK(funcname, optimizer, filename) \
  {                                                                          \
    WarningCapture warnings;                                                 \
    funcname(optimizer, filename);                                           \
    ASSERT_EQ(                                                               \
        count_substr_occurrences(warnings.str(), "old serialization"), 1);   \
  }

TEST(SerializeTest, KeysFunc) {
  auto tempfile = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  for (const auto i : c10::irange(3)) {
    output_archive.write(
        "element/" + std::to_string(i), c10::IValue(static_cast<int64_t>(i)));
  }
  output_archive.save_to(tempfile.name);
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(tempfile.name);
  std::vector<std::string> keys = input_archive.keys();
  ASSERT_EQ(keys.size(), 3);
  for (const auto i : c10::irange(keys.size())) {
    ASSERT_EQ(keys[i], "element/" + std::to_string(i));
  }
}

TEST(SerializeTest, TryReadFunc) {
  auto tempfile = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  for (const auto i : c10::irange(3)) {
    output_archive.write(
        "element/" + std::to_string(i), c10::IValue(static_cast<int64_t>(i)));
  }
  output_archive.save_to(tempfile.name);
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(tempfile.name);
  c10::IValue ivalue;
  ASSERT_FALSE(input_archive.try_read("1", ivalue));
  ASSERT_TRUE(input_archive.try_read("element/1", ivalue));
  ASSERT_EQ(ivalue.toInt(), 1);
}

TEST(SerializeTest, Basic) {
  torch::manual_seed(0);

  auto x = torch::randn({5, 5});
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, MathBits) {
  torch::manual_seed(0);

  auto options = torch::TensorOptions{}.dtype(torch::kComplexFloat);
  auto x = torch::randn({5, 5}, options);
  {
    auto expected = torch::conj(x);
    auto actual = save_and_load(expected);

    ASSERT_TRUE(actual.defined());
    ASSERT_EQ(actual.sizes().vec(), expected.sizes().vec());
    ASSERT_TRUE(actual.allclose(expected));
  }

  {
    auto expected = torch::_neg_view(x);
    auto actual = save_and_load(expected);

    ASSERT_TRUE(actual.defined());
    ASSERT_EQ(actual.sizes().vec(), expected.sizes().vec());
    ASSERT_TRUE(actual.allclose(expected));
  }

  {
    auto expected = torch::conj(torch::_neg_view(x));
    auto actual = save_and_load(expected);

    ASSERT_TRUE(actual.defined());
    ASSERT_EQ(actual.sizes().vec(), expected.sizes().vec());
    ASSERT_TRUE(actual.allclose(expected));
  }

  {
    // We don't support serializing `ZeroTensor` as it is not public facing yet.
    // If in future, `ZeroTensor` serialization is supported, this test should
    // start failing!
    auto t = torch::_efficientzerotensor({5, 5});
    ASSERT_THROWS_WITH(save_and_load(t), "ZeroTensor is not serializable,");
  }
}

TEST(SerializeTest, BasicToFile) {
  torch::manual_seed(0);

  auto x = torch::randn({5, 5});

  auto tempfile = c10::make_tempfile();
  torch::save(x, tempfile.name);

  torch::Tensor y;
  torch::load(y, tempfile.name);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, BasicViaFunc) {
  torch::manual_seed(0);

  auto x = torch::randn({5, 5});

  std::string serialized;
  torch::save(x, [&](const void* buf, size_t n) {
    serialized.append(reinterpret_cast<const char*>(buf), n);
    return n;
  });
  torch::Tensor y;
  torch::load(y, serialized.data(), serialized.size());

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));

  torch::Tensor z;
  torch::load(
      z,
      [&](uint64_t pos, void* buf, size_t n) -> size_t {
        if (pos >= serialized.size())
          return 0;
        size_t nbytes =
            std::min(static_cast<size_t>(pos) + n, serialized.size()) - pos;
        memcpy(buf, serialized.data() + pos, nbytes);
        return nbytes;
      },
      [&]() -> size_t { return serialized.size(); });
  ASSERT_TRUE(z.defined());
  ASSERT_EQ(x.sizes().vec(), z.sizes().vec());
  ASSERT_TRUE(x.allclose(z));
}

TEST(SerializeTest, Resized) {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x.resize_({5, 5});
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, Sliced) {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x = x.slice(0, 1, 5);
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, NonContiguous) {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x = x.slice(1, 1, 4);
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, ErrorOnMissingKey) {
  struct B : torch::nn::Module {
    B(const std::string& name_c) {
      register_buffer(name_c, torch::ones(5, torch::kFloat));
    }
  };
  struct A : torch::nn::Module {
    A(const std::string& name_b, const std::string& name_c) {
      register_module(name_b, std::make_shared<B>(name_c));
    }
  };
  struct M : torch::nn::Module {
    M(const std::string& name_a,
      const std::string& name_b,
      const std::string& name_c) {
      register_module(name_a, std::make_shared<A>(name_b, name_c));
    }
  };

  // create a hierarchy of models with names differing below the top level
  auto model1 = std::make_shared<M>("a", "b", "c");
  auto model2 = std::make_shared<M>("a", "b", "x");
  auto model3 = std::make_shared<M>("a", "x", "c");

  std::stringstream stream;
  torch::save(model1, stream);
  // We want the errors to contain hierarchy information, too.
  ASSERT_THROWS_WITH(
      torch::load(model2, stream), "No such serialized tensor 'a.b.x'");
  stream.seekg(0, stream.beg);
  ASSERT_THROWS_WITH(
      torch::load(model3, stream), "No such serialized submodule: 'a.x'");
}

TEST(SerializeTest, XOR) {
  // We better be able to save and load an XOR model!
  auto getLoss = [](Sequential model, uint32_t batch_size) {
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    for (const auto i : c10::irange(batch_size)) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
    }
    auto x = model->forward<torch::Tensor>(inputs);
    return torch::binary_cross_entropy(x, labels);
  };

  auto model = xor_model();
  auto model2 = xor_model();
  auto model3 = xor_model();
  auto optimizer = torch::optim::SGD(
      model->parameters(),
      torch::optim::SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(
          1e-6));

  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    torch::Tensor loss = getLoss(model, 4);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    running_loss = running_loss * 0.99 + loss.sum().item<float>() * 0.01;
    ASSERT_LT(epoch, 3000);
    epoch++;
  }

  auto tempfile = c10::make_tempfile();
  torch::save(model, tempfile.name);
  torch::load(model2, tempfile.name);

  auto loss = getLoss(model2, 100);
  ASSERT_LT(loss.item<float>(), 0.1);
}

TEST(SerializeTest, Optim) {
  auto model1 = Linear(5, 2);
  auto model2 = Linear(5, 2);
  auto model3 = Linear(5, 2);

  // Models 1, 2, 3 will have the same parameters.
  auto model_tempfile = c10::make_tempfile();
  torch::save(model1, model_tempfile.name);
  torch::load(model2, model_tempfile.name);
  torch::load(model3, model_tempfile.name);

  auto param1 = model1->named_parameters();
  auto param2 = model2->named_parameters();
  auto param3 = model3->named_parameters();
  for (const auto& p : param1) {
    ASSERT_TRUE(p->allclose(param2[p.key()]));
    ASSERT_TRUE(param2[p.key()].allclose(param3[p.key()]));
  }

  // Make some optimizers with momentum (and thus state)
  auto optim1 = torch::optim::SGD(
      model1->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim2 = torch::optim::SGD(
      model2->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim2_2 = torch::optim::SGD(
      model2->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim3 = torch::optim::SGD(
      model3->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim3_2 = torch::optim::SGD(
      model3->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));

  auto x = torch::ones({10, 5});

  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };

  // Do 2 steps of model1
  step(optim1, model1);
  step(optim1, model1);

  // Do 2 steps of model 2 without saving the optimizer
  step(optim2, model2);
  step(optim2_2, model2);

  // Do 2 steps of model 3 while saving the optimizer
  step(optim3, model3);

  auto optim_tempfile = c10::make_tempfile();
  torch::save(optim3, optim_tempfile.name);
  torch::load(optim3_2, optim_tempfile.name);
  step(optim3_2, model3);

  param1 = model1->named_parameters();
  param2 = model2->named_parameters();
  param3 = model3->named_parameters();
  for (const auto& p : param1) {
    const auto& name = p.key();
    // Model 1 and 3 should be the same
    ASSERT_TRUE(
        param1[name].norm().item<float>() == param3[name].norm().item<float>());
    ASSERT_TRUE(
        param1[name].norm().item<float>() != param2[name].norm().item<float>());
  }
}

TEST(SerializeTest, Optim_Adagrad) {
  test_serialize_optimizer<Adagrad, AdagradOptions, AdagradParamState>(
      AdagradOptions(1e-1));

  // bc compatibility check
  auto model1 = Linear(5, 2);
  auto optim1 = torch::optim::Adagrad(
      model1->parameters(), torch::optim::AdagradOptions(1e-1));

  auto x = torch::ones({10, 5});
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };
  step(optim1, model1);
  auto optim1_2 =
      Adagrad(model1->parameters(), torch::optim::AdagradOptions(1e-1));

  // fill up with optim1 sum_buffers
  std::vector<torch::Tensor> sum_buffers;
  // fill up with optim1 state_buffers
  std::vector<int64_t> step_buffers;
  const auto& params_ = optim1.param_groups()[0].params();
  const auto& optim1_state = optim1.state();
  for (const auto& param : params_) {
    auto key_ = param.unsafeGetTensorImpl();
    const AdagradParamState& curr_state_ =
        static_cast<const AdagradParamState&>(*(optim1_state.at(key_).get()));
    sum_buffers.emplace_back(curr_state_.sum());
    step_buffers.emplace_back(curr_state_.step());
  }
  // write sum_buffers and step_buffers to the file
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  write_tensors_to_archive(output_archive, "sum_buffers", sum_buffers);
  write_step_buffers(output_archive, "step_buffers", step_buffers);
  output_archive.save_to(optim_tempfile_old_format.name);
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  is_optimizer_state_equal<AdagradParamState>(optim1.state(), optim1_2.state());
}

TEST(SerializeTest, Optim_SGD) {
  test_serialize_optimizer<SGD, SGDOptions, SGDParamState>(
      SGDOptions(1e-1).momentum(0.9));

  // bc compatibility check
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  // added a tensor for lazy init check - when all params do not have a momentum
  // buffer entry
  model1_params.emplace_back(torch::randn({2, 3}));
  auto optim1 = torch::optim::SGD(
      model1_params, torch::optim::SGDOptions(0.01).momentum(0.9));

  auto x = torch::ones({10, 5});
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };
  step(optim1, model1);

  std::vector<at::Tensor> momentum_buffers;
  int64_t iteration_{0};
  const auto& params_ = optim1.param_groups()[0].params();
  const auto& optim1_state = optim1.state();
  for (const auto i : c10::irange(params_.size())) {
    if (i != (params_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      const SGDParamState& curr_state_ =
          static_cast<const SGDParamState&>(*(optim1_state.at(key_).get()));
      momentum_buffers.emplace_back(curr_state_.momentum_buffer());
    }
  }
  ASSERT_TRUE(momentum_buffers.size() == (params_.size() - 1));
  // write momentum_buffers to the file
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  write_tensors_to_archive(
      output_archive, "momentum_buffers", momentum_buffers);
  write_int_value(output_archive, "iteration_", iteration_);
  output_archive.save_to(optim_tempfile_old_format.name);
  auto optim1_2 =
      SGD(model1_params, torch::optim::SGDOptions(1e-1).momentum(0.9));
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  is_optimizer_state_equal<SGDParamState>(optim1.state(), optim1_2.state());
}

TEST(SerializeTest, Optim_Adam) {
  test_serialize_optimizer<Adam, AdamOptions, AdamParamState>(
      AdamOptions().lr(0.99999).amsgrad(true).weight_decay(0.5));

  // bc compatibility check
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  // added a tensor for lazy init check - when all params do not have entry in
  // buffers
  model1_params.emplace_back(torch::randn({2, 3}));
  auto optim1 = torch::optim::Adam(
      model1_params, torch::optim::AdamOptions().weight_decay(0.5));

  auto x = torch::ones({10, 5});
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };
  step(optim1, model1);

  std::vector<int64_t> step_buffers;
  std::vector<at::Tensor> exp_average_buffers;
  std::vector<at::Tensor> exp_average_sq_buffers;
  std::vector<at::Tensor> max_exp_average_sq_buffers;
  const auto& params_ = optim1.param_groups()[0].params();
  const auto& optim1_state = optim1.state();
  for (const auto i : c10::irange(params_.size())) {
    if (i != (params_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      const AdamParamState& curr_state_ =
          static_cast<const AdamParamState&>(*(optim1_state.at(key_).get()));
      step_buffers.emplace_back(curr_state_.step());
      exp_average_buffers.emplace_back(curr_state_.exp_avg());
      exp_average_sq_buffers.emplace_back(curr_state_.exp_avg_sq());
      if (curr_state_.max_exp_avg_sq().defined()) {
        max_exp_average_sq_buffers.emplace_back(curr_state_.max_exp_avg_sq());
      }
    }
  }
  // write buffers to the file
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  write_step_buffers(output_archive, "step_buffers", step_buffers);
  write_tensors_to_archive(
      output_archive, "exp_average_buffers", exp_average_buffers);
  write_tensors_to_archive(
      output_archive, "exp_average_sq_buffers", exp_average_sq_buffers);
  write_tensors_to_archive(
      output_archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
  output_archive.save_to(optim_tempfile_old_format.name);
  auto optim1_2 = Adam(model1_params, torch::optim::AdamOptions());
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  is_optimizer_state_equal<AdamParamState>(optim1.state(), optim1_2.state());
}

TEST(SerializeTest, Optim_AdamW) {
  test_serialize_optimizer<AdamW, AdamWOptions, AdamWParamState>(
      AdamWOptions().lr(0.99999).amsgrad(true).betas(
          std::make_tuple(0.999, 0.1)));

  // bc compatibility check
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  // added a tensor for lazy init check - when all params do not have entry in
  // buffers
  model1_params.emplace_back(torch::randn({2, 3}));
  auto optim1 = torch::optim::AdamW(
      model1_params, torch::optim::AdamWOptions().weight_decay(0.5));

  auto x = torch::ones({10, 5});
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };
  step(optim1, model1);

  std::vector<int64_t> step_buffers;
  std::vector<at::Tensor> exp_average_buffers;
  std::vector<at::Tensor> exp_average_sq_buffers;
  std::vector<at::Tensor> max_exp_average_sq_buffers;
  const auto& params_ = optim1.param_groups()[0].params();
  const auto& optim1_state = optim1.state();
  for (const auto i : c10::irange(params_.size())) {
    if (i != (params_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      const AdamWParamState& curr_state_ =
          static_cast<const AdamWParamState&>(*(optim1_state.at(key_).get()));
      step_buffers.emplace_back(curr_state_.step());
      exp_average_buffers.emplace_back(curr_state_.exp_avg());
      exp_average_sq_buffers.emplace_back(curr_state_.exp_avg_sq());
      if (curr_state_.max_exp_avg_sq().defined()) {
        max_exp_average_sq_buffers.emplace_back(curr_state_.max_exp_avg_sq());
      }
    }
  }
  // write buffers to the file
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  write_step_buffers(output_archive, "step_buffers", step_buffers);
  write_tensors_to_archive(
      output_archive, "exp_average_buffers", exp_average_buffers);
  write_tensors_to_archive(
      output_archive, "exp_average_sq_buffers", exp_average_sq_buffers);
  write_tensors_to_archive(
      output_archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
  output_archive.save_to(optim_tempfile_old_format.name);
  auto optim1_2 = AdamW(model1_params, torch::optim::AdamWOptions());
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  is_optimizer_state_equal<AdamWParamState>(optim1.state(), optim1_2.state());
}

TEST(SerializeTest, Optim_RMSprop) {
  auto options = RMSpropOptions(0.1).momentum(0.9).centered(true);
  test_serialize_optimizer<RMSprop, RMSpropOptions, RMSpropParamState>(options);

  // bc compatibility check
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();

  // added a tensor for lazy init check - when all params do not have a momentum
  // buffer entry
  model1_params.emplace_back(torch::randn({2, 3}));
  auto optim1 = torch::optim::RMSprop(model1_params, options);

  auto x = torch::ones({10, 5});
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };
  step(optim1, model1);

  std::vector<at::Tensor> square_average_buffers;
  std::vector<at::Tensor> momentum_buffers;
  std::vector<at::Tensor> grad_average_buffers;
  const auto& params_ = optim1.param_groups()[0].params();
  const auto& optim1_state = optim1.state();
  for (const auto i : c10::irange(params_.size())) {
    if (i != (params_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      const RMSpropParamState& curr_state_ =
          static_cast<const RMSpropParamState&>(*(optim1_state.at(key_).get()));
      square_average_buffers.emplace_back(curr_state_.square_avg());
      if (curr_state_.momentum_buffer().defined()) {
        momentum_buffers.emplace_back(curr_state_.momentum_buffer());
      }
      if (curr_state_.grad_avg().defined()) {
        grad_average_buffers.emplace_back(curr_state_.grad_avg());
      }
    }
  }
  // write buffers to the file
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  write_tensors_to_archive(
      output_archive, "square_average_buffers", square_average_buffers);
  write_tensors_to_archive(
      output_archive, "momentum_buffers", momentum_buffers);
  write_tensors_to_archive(
      output_archive, "grad_average_buffers", grad_average_buffers);
  output_archive.save_to(optim_tempfile_old_format.name);
  auto optim1_2 = RMSprop(model1_params, options);
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  const auto& params1_2_ = optim1_2.param_groups()[0].params();
  auto& optim1_2_state = optim1_2.state();
  // old RMSprop didn't track step value
  for (const auto i : c10::irange(params1_2_.size())) {
    if (i != (params1_2_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      auto key1_2_ = params1_2_[i].unsafeGetTensorImpl();
      const RMSpropParamState& curr_state_ =
          static_cast<const RMSpropParamState&>(*(optim1_state.at(key_).get()));
      RMSpropParamState& curr_state1_2_ =
          static_cast<RMSpropParamState&>(*(optim1_2_state.at(key_).get()));
      curr_state1_2_.step(curr_state_.step());
    }
  }
  is_optimizer_state_equal<RMSpropParamState>(optim1.state(), optim1_2.state());
}

TEST(SerializeTest, Optim_LBFGS) {
  test_serialize_optimizer<LBFGS, LBFGSOptions, LBFGSParamState>(
      LBFGSOptions(), true);
  // bc compatibility check
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  // added a tensor for lazy init check - when all params do not have entry in
  // buffers
  model1_params.emplace_back(torch::randn({2, 3}));
  auto optim1 =
      torch::optim::LBFGS(model1_params, torch::optim::LBFGSOptions());

  auto x = torch::ones({10, 5});
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    auto closure = []() { return torch::tensor({10}); };
    optimizer.step(closure);
  };

  step(optim1, model1);

  at::Tensor d, t, H_diag, prev_flat_grad, prev_loss;
  std::deque<at::Tensor> old_dirs, old_stps;

  const auto& params_ = optim1.param_groups()[0].params();
  auto key_ = params_[0].unsafeGetTensorImpl();
  const auto& optim1_state =
      static_cast<const LBFGSParamState&>(*(optim1.state().at(key_).get()));
  d = optim1_state.d();
  t = at::tensor(optim1_state.t());
  H_diag = optim1_state.H_diag();
  prev_flat_grad = optim1_state.prev_flat_grad();
  prev_loss = at::tensor(optim1_state.prev_loss());
  old_dirs = optim1_state.old_dirs();

  // write buffers to the file
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  output_archive.write("d", d, /*is_buffer=*/true);
  output_archive.write("t", t, /*is_buffer=*/true);
  output_archive.write("H_diag", H_diag, /*is_buffer=*/true);
  output_archive.write("prev_flat_grad", prev_flat_grad, /*is_buffer=*/true);
  output_archive.write("prev_loss", prev_loss, /*is_buffer=*/true);
  write_tensors_to_archive(output_archive, "old_dirs", old_dirs);
  write_tensors_to_archive(output_archive, "old_stps", old_stps);
  output_archive.save_to(optim_tempfile_old_format.name);

  auto optim1_2 = LBFGS(model1_params, torch::optim::LBFGSOptions());
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);

  const auto& params1_2_ = optim1_2.param_groups()[0].params();
  auto param_key = params1_2_[0].unsafeGetTensorImpl();
  auto& optim1_2_state =
      static_cast<LBFGSParamState&>(*(optim1_2.state().at(param_key).get()));

  // old LBFGS didn't track func_evals, n_iter, ro, al values
  optim1_2_state.func_evals(optim1_state.func_evals());
  optim1_2_state.n_iter(optim1_state.n_iter());
  optim1_2_state.ro(optim1_state.ro());
  optim1_2_state.al(optim1_state.al());

  is_optimizer_state_equal<LBFGSParamState>(optim1.state(), optim1_2.state());
}

TEST(SerializeTest, XOR_CUDA) {
  torch::manual_seed(0);
  // We better be able to save and load a XOR model!
  auto getLoss = [](Sequential model,
                    uint32_t batch_size,
                    bool is_cuda = false) {
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    if (is_cuda) {
      inputs = inputs.cuda();
      labels = labels.cuda();
    }
    for (const auto i : c10::irange(batch_size)) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
    }
    auto x = model->forward<torch::Tensor>(inputs);
    return torch::binary_cross_entropy(x, labels);
  };

  auto model = xor_model();
  auto model2 = xor_model();
  auto model3 = xor_model();
  auto optimizer = torch::optim::SGD(
      model->parameters(),
      torch::optim::SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(
          1e-6));

  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    torch::Tensor loss = getLoss(model, 4);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    running_loss = running_loss * 0.99 + loss.sum().item<float>() * 0.01;
    ASSERT_LT(epoch, 3000);
    epoch++;
  }

  auto tempfile = c10::make_tempfile();
  torch::save(model, tempfile.name);
  torch::load(model2, tempfile.name);

  auto loss = getLoss(model2, 100);
  ASSERT_LT(loss.item<float>(), 0.1);

  model2->to(torch::kCUDA);
  loss = getLoss(model2, 100, true);
  ASSERT_LT(loss.item<float>(), 0.1);

  auto tempfile2 = c10::make_tempfile();
  torch::save(model2, tempfile2.name);
  torch::load(model3, tempfile2.name);

  loss = getLoss(model3, 100, true);
  ASSERT_LT(loss.item<float>(), 0.1);
}

TEST(
    SerializeTest,
    CanSerializeModulesWithIntermediateModulesWithoutParametersOrBuffers) {
  struct C : torch::nn::Module {
    C() {
      register_buffer("foo", torch::ones(5, torch::kInt32));
    }
  };
  struct B : torch::nn::Module {};
  struct A : torch::nn::Module {
    A() {
      register_module("b", std::make_shared<B>());
      register_module("c", std::make_shared<C>());
    }
  };
  struct M : torch::nn::Module {
    M() {
      register_module("a", std::make_shared<A>());
    }
  };

  auto out = std::make_shared<M>();
  std::stringstream ss;
  torch::save(out, ss);
  auto in = std::make_shared<M>();
  torch::load(in, ss);

  const int output = in->named_buffers()["a.c.foo"].sum().item<int>();
  ASSERT_EQ(output, 5);
}

TEST(SerializeTest, VectorOfTensors) {
  torch::manual_seed(0);

  std::vector<torch::Tensor> x_vec = {
      torch::randn({1, 2}), torch::randn({3, 4})};

  std::stringstream stream;
  torch::save(x_vec, stream);

  std::vector<torch::Tensor> y_vec;
  torch::load(y_vec, stream);

  for (const auto i : c10::irange(x_vec.size())) {
    auto& x = x_vec[i];
    auto& y = y_vec[i];
    ASSERT_TRUE(y.defined());
    ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
    ASSERT_TRUE(x.allclose(y));
  }
}

TEST(SerializeTest, IValue) {
  c10::IValue ivalue(1);
  auto tempfile = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  output_archive.write("value", ivalue);
  output_archive.save_to(tempfile.name);

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(tempfile.name);
  c10::IValue ivalue_out;
  input_archive.read("value", ivalue_out);
  ASSERT_EQ(ivalue_out.toInt(), 1);

  ASSERT_THROWS_WITH(
      input_archive.read("bad_key", ivalue_out),
      "does not have a field with name");
}

// NOTE: if a `Module` contains unserializable submodules (e.g.
// `nn::Functional`), we expect those submodules to be skipped when the `Module`
// is being serialized.
TEST(SerializeTest, UnserializableSubmoduleIsSkippedWhenSavingModule) {
  struct A : torch::nn::Module {
    A() {
      register_module("relu", torch::nn::Functional(torch::relu));
    }
  };

  auto out = std::make_shared<A>();
  std::stringstream ss;
  torch::save(out, ss);

  torch::serialize::InputArchive archive;
  archive.load_from(ss);
  torch::serialize::InputArchive relu_archive;

  // Submodule with name "relu" should not exist in the `InputArchive`,
  // because the "relu" submodule is an `nn::Functional` and is not
  // serializable.
  ASSERT_FALSE(archive.try_read("relu", relu_archive));
}

// NOTE: If a `Module` contains unserializable submodules (e.g.
// `nn::Functional`), we don't check the existence of those submodules in the
// `InputArchive` when deserializing.
TEST(SerializeTest, UnserializableSubmoduleIsIgnoredWhenLoadingModule) {
  struct B : torch::nn::Module {
    B() {
      register_module("relu1", torch::nn::Functional(torch::relu));
      register_buffer("foo", torch::zeros(5, torch::kInt32));
    }
  };
  struct A : torch::nn::Module {
    A() {
      register_module("b", std::make_shared<B>());
      register_module("relu2", torch::nn::Functional(torch::relu));
    }
  };

  auto out = std::make_shared<A>();
  // Manually change the values of "b.foo", so that we can check whether the
  // buffer contains these values after deserialization.
  out->named_buffers()["b.foo"].fill_(1);
  auto tempfile = c10::make_tempfile();
  torch::save(out, tempfile.name);

  torch::serialize::InputArchive archive;
  archive.load_from(tempfile.name);
  torch::serialize::InputArchive archive_b;
  torch::serialize::InputArchive archive_relu;
  torch::Tensor tensor_foo;

  ASSERT_TRUE(archive.try_read("b", archive_b));
  ASSERT_TRUE(archive_b.try_read("foo", tensor_foo, /*is_buffer=*/true));

  // Submodule with name "relu1" should not exist in `archive_b`, because the
  // "relu1" submodule is an `nn::Functional` and is not serializable.
  ASSERT_FALSE(archive_b.try_read("relu1", archive_relu));

  // Submodule with name "relu2" should not exist in `archive`, because the
  // "relu2" submodule is an `nn::Functional` and is not serializable.
  ASSERT_FALSE(archive.try_read("relu2", archive_relu));

  auto in = std::make_shared<A>();
  // `torch::load(...)` works without error, even though `A` contains the
  // `nn::Functional` submodules while the serialized file doesn't, because the
  // `nn::Functional` submodules are not serializable and thus ignored when
  // deserializing.
  torch::load(in, tempfile.name);

  // Check that the "b.foo" buffer is correctly deserialized from the file.
  const int output = in->named_buffers()["b.foo"].sum().item<int>();
  // `output` should equal to the sum of the values we manually assigned to
  // "b.foo" before serialization.
  ASSERT_EQ(output, 5);
}
