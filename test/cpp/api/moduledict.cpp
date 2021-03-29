#include <gtest/gtest.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ModuleDictTest : torch::test::SeedingFixture {};

TEST_F(ModuleDictTest, ConstructsFromList) {
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list = {
    {"module_1", std::make_shared<M>(1)},
    {"module_2", std::make_shared<M>(2)},
    {"module_3", std::make_shared<M>(3)}
  };
  ModuleDict dict(list);
  ASSERT_EQ(dict->size(), 3);
}

TEST_F(ModuleDictTest, ConstructsFromordereddict) {
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"module_1", std::make_shared<M>(1)},
    {"module_2", std::make_shared<M>(2)},
    {"module_3", std::make_shared<M>(3)},
  };
  ModuleDict dict(ordereddict);
  ASSERT_EQ(dict->size(), 3);
}

TEST_F(ModuleDictTest, UpdatePopClearContains) {
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  ModuleDict dict;
  ASSERT_TRUE(dict->empty());
  // Update by List
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list1 = {
    {"module_1", std::make_shared<M>(1)}
  };
  dict->update(list1);
  ASSERT_EQ(dict->size(), 1);
  ASSERT_TRUE(dict->contains("module_1"));
  // Update by OrderedDict
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"module_2", std::make_shared<M>(2)}
  };
  dict->update(ordereddict);
  ASSERT_EQ(dict->size(), 2);
  ASSERT_TRUE(dict->contains("module_2"));
  // Update by another ModuleDict
  std::vector<std::pair<std::string, std::shared_ptr<Module>>>list2 = {
    {"module_3", std::make_shared<M>(3)}
  };
  ModuleDict updatedict(list2);
  dict->update(*updatedict);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(dict->contains("module_3"));
  // Pop
  dict->pop("module_1");
  ASSERT_EQ(dict->size(), 2);
  // Pop unexist
  ASSERT_THROWS_WITH(dict->pop("module_4"), " 'module_4' is not defined");
  // Clear
  dict->clear();
  ASSERT_EQ(dict->size(), 0);
}

TEST_F(ModuleDictTest, UpdateExist) {
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list1 = {
    {"module_1", std::make_shared<M>(1)},
    {"module_2", std::make_shared<M>(2)}
  };
  ModuleDict dict(list1);
  ASSERT_EQ(dict->at<M>("module_2").value, 2);
  // Update by list
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list2 = {
    {"module_2", std::make_shared<M>(0)},
    {"module_3", std::make_shared<M>(3)}
  };
  dict->update(list2);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_EQ(dict->at<M>("module_2").value, 0);
  // Update by ordereddict
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"module_3", std::make_shared<M>(0)},
    {"module_4", std::make_shared<M>(4)}
  };
  dict->update(ordereddict);
  ASSERT_EQ(dict->size(), 4);
  ASSERT_EQ(dict->at<M>("module_3").value, 0);
  // Update by ModuleDict
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list3 = {
    {"module_4", std::make_shared<M>(0)},
    {"module_1", std::make_shared<M>(0)}
  };
  ModuleDict dict2(list3);
  dict->update(*dict2);
  ASSERT_EQ(dict->size(), 4);
  ASSERT_EQ(dict->at<M>("module_1").value, 0);
  ASSERT_EQ(dict->at<M>("module_4").value, 0);
}

TEST_F(ModuleDictTest, Keys) {
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"linear", Linear(10, 3).ptr()},
    {"conv", Conv2d(1, 2, 3).ptr()},
    {"dropout", Dropout(0.5).ptr()},
  };
  ModuleDict dict(ordereddict);
  const auto& keys = dict->keys();
  std::vector<std::string> expected{"linear", "conv", "dropout"};
  ASSERT_EQ(keys, expected);
  ASSERT_THROWS_WITH(dict["batch"], " 'batch' is not defined");

  ASSERT_TRUE(dict["linear"]->as<Linear>());
  ASSERT_TRUE(dict["conv"]->as<Conv2d>());
  ASSERT_TRUE(dict["dropout"]->as<Dropout>());
}

TEST_F(ModuleDictTest, Values) {
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"module_1", std::make_shared<M>(1)},
    {"module_2", std::make_shared<M>(2)},
  };
  ModuleDict dict(ordereddict);
  const auto& values = dict->values();
  const auto& expected = ordereddict.values();
  ASSERT_EQ(values, expected);
  ASSERT_TRUE(std::equal(
      dict->begin(),
      dict->end(),
      ordereddict.begin(),
      [](const auto& lhs,
         const auto& rhs) {
        return lhs.value().get() == rhs.value().get();
      }));
}

TEST_F(ModuleDictTest, SanityCheckForHoldingStandardModules) {
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"linear", Linear(10, 3).ptr()},
    {"conv", Conv2d(1, 2, 3).ptr()},
    {"dropout", Dropout(0.5).ptr()},
    {"batch", BatchNorm2d(5).ptr()},
    {"embedding", Embedding(4, 10).ptr()},
    {"lstm", LSTM(4, 5).ptr()}
  };
  ModuleDict dict(ordereddict);
}

TEST_F(ModuleDictTest, HasReferenceSemantics) {
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"linear1", Linear(2, 3).ptr()},
    {"linear2", Linear(3, 4).ptr()},
    {"linear3", Linear(4, 5).ptr()},
  };
  ModuleDict first(ordereddict);
  ModuleDict second(ordereddict);

  ASSERT_EQ(first->size(), second->size());
  ASSERT_TRUE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const auto& lhs,
         const auto& rhs) {
        return lhs.value().get() == rhs.value().get();
      }));
}

void iscloneable_helper(torch::Device device) {
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"linear", Linear(2, 3).ptr()},
    {"relu", Functional(torch::relu).ptr()},
    {"batch", BatchNorm1d(3).ptr()},
  };
  ModuleDict dict(ordereddict);
  dict->to(device);
  ModuleDict clone = std::dynamic_pointer_cast<ModuleDictImpl>(dict->clone(device));
  ASSERT_EQ(dict->size(), clone->size());

  for (auto it = dict->begin(), it_c = clone->begin(); it != dict->end(); ++it, ++it_c) {
    // The key should be same
    ASSERT_EQ(it->key(), it_c->key());
    // The modules should be the same kind (type).
    ASSERT_EQ(it->value()->name(), it_c->value()->name());
    // But not pointer-equal (distinct objects).
    ASSERT_NE(it->value(), it_c->value());
  }

  // Verify that the clone is deep, i.e. parameters of modules are cloned too.
  torch::NoGradGuard no_grad;

  auto params1 = dict->named_parameters();
  auto params2 = clone->named_parameters();
  ASSERT_EQ(params1.size(), params2.size());
  for (auto& param : params1) {
    ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
    ASSERT_EQ(param->device(), params2[param.key()].device());
    ASSERT_TRUE(param->allclose(params2[param.key()]));
    param->add_(2);
  }
  for (auto& param : params1) {
    ASSERT_FALSE(param->allclose(params2[param.key()]));
  }
}

TEST_F(ModuleDictTest, IsCloneable) {
  iscloneable_helper(torch::kCPU);
}

TEST_F(ModuleDictTest, IsCloneable_CUDA) {
  iscloneable_helper({torch::kCUDA, 0});
}

TEST_F(ModuleDictTest, RegistersElementsAsSubmodules) {
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict1 = {
    {"linear", Linear(10, 3).ptr()},
    {"conv", Conv2d(1, 2, 3).ptr()},
    {"test", Dropout(0.5).ptr()},
  };
  ModuleDict dict(ordereddict1);

  auto modules = dict->children();
  ASSERT_TRUE(modules[0]->as<Linear>());
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  ASSERT_TRUE(modules[2]->as<Dropout>());

  // Update Existing
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict2 = {
    {"lstm", LSTM(4, 5).ptr()},
    {"test", BatchNorm2d(5).ptr()}
  };
  dict->update(ordereddict2);

  modules = dict->children();
  ASSERT_TRUE(modules[0]->as<Linear>());
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  // Keep Order
  ASSERT_TRUE(modules[2]->as<BatchNorm2d>());
  ASSERT_TRUE(modules[3]->as<LSTM>());
}

TEST_F(ModuleDictTest, CloneToDevice_CUDA) {
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"linear", Linear(2, 3).ptr()},
    {"relu", Functional(torch::relu).ptr()},
    {"batch", BatchNorm1d(3).ptr()},
  };
  ModuleDict dict(ordereddict);
  torch::Device device(torch::kCUDA, 0);
  ModuleDict clone =
      std::dynamic_pointer_cast<ModuleDictImpl>(dict->clone(device));
  for (const auto& p : clone->parameters()) {
    ASSERT_EQ(p.device(), device);
  }
  for (const auto& b : clone->buffers()) {
    ASSERT_EQ(b.device(), device);
  }
}

TEST_F(ModuleDictTest, PrettyPrintModuleDict) {
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
    {"linear", Linear(10, 3).ptr()},
    {"conv", Conv2d(1, 2, 3).ptr()},
    {"dropout", Dropout(0.5).ptr()},
    {"batch", BatchNorm2d(5).ptr()},
    {"embedding", Embedding(4, 10).ptr()},
    {"lstm", LSTM(4, 5).ptr()}
  };
  ModuleDict dict(ordereddict);

  ASSERT_EQ(
      c10::str(dict),
      "torch::nn::ModuleDict(\n"
      "  (linear): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (conv): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (dropout): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (batch): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (embedding): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (lstm): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
}
