#include <gtest/gtest.h>

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ModuleListTest : torch::test::SeedingFixture {};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, ConstructsFromSharedPointer) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  ModuleList list(
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3));
  ASSERT_EQ(list->size(), 3);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, ConstructsFromConcreteType) {
  static int copy_count;

  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    M(const M& other) : torch::nn::Module(other) {
      copy_count++;
    }
    int value;
  };

  copy_count = 0;
  ModuleList list(M(1), M(2), M(3));
  ASSERT_EQ(list->size(), 3);
  // NOTE: The current implementation expects each module to be copied exactly
  // once, which happens when the module is passed into `std::make_shared<T>()`.
  // TODO: Find a way to avoid copying, and then delete the copy constructor of
  // `M`.
  ASSERT_EQ(copy_count, 3);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, ConstructsFromModuleHolder) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };

  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  ModuleList list(M(1), M(2), M(3));
  ASSERT_EQ(list->size(), 3);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, PushBackAddsAnElement) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  ModuleList list;
  ASSERT_EQ(list->size(), 0);
  ASSERT_TRUE(list->is_empty());
  list->push_back(Linear(3, 4));
  ASSERT_EQ(list->size(), 1);
  list->push_back(std::make_shared<M>(1));
  ASSERT_EQ(list->size(), 2);
  list->push_back(M(2));
  ASSERT_EQ(list->size(), 3);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, Insertion) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };
  TORCH_MODULE(M);

  ModuleList list;
  list->push_back(MImpl(1));
  ASSERT_EQ(list->size(), 1);
  list->insert(0, std::make_shared<MImpl>(2));
  ASSERT_EQ(list->size(), 2);
  list->insert(1, M(3));
  ASSERT_EQ(list->size(), 3);
  list->insert(3, M(4));
  ASSERT_EQ(list->size(), 4);
  ASSERT_EQ(list->at<MImpl>(0).value, 2);
  ASSERT_EQ(list->at<MImpl>(1).value, 3);
  ASSERT_EQ(list->at<MImpl>(2).value, 1);
  ASSERT_EQ(list->at<MImpl>(3).value, 4);

  std::unordered_map<size_t, size_t> U = {{0, 2}, {1, 3}, {2, 1}, {3, 4}};
  for (const auto& P : list->named_modules("", false))
    ASSERT_EQ(U[std::stoul(P.key())], P.value()->as<M>()->value);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, AccessWithAt) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  ModuleList list;
  for (auto& module : modules) {
    list->push_back(module);
  }
  ASSERT_EQ(list->size(), 3);

  // returns the correct module for a given index
  for (size_t i = 0; i < modules.size(); ++i) {
    ASSERT_EQ(&list->at<M>(i), modules[i].get());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->at<M>(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(
      list->at<M>(modules.size() + 1000000), "Index out of range");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, AccessWithPtr) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  ModuleList list;
  for (auto& module : modules) {
    list->push_back(module);
  }
  ASSERT_EQ(list->size(), 3);

  // returns the correct module for a given index
  for (size_t i = 0; i < modules.size(); ++i) {
    ASSERT_EQ(list->ptr(i).get(), modules[i].get());
    ASSERT_EQ(list[i].get(), modules[i].get());
    ASSERT_EQ(list->ptr<M>(i).get(), modules[i].get());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->ptr(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(list->ptr(modules.size() + 1000000), "Index out of range");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, SanityCheckForHoldingStandardModules) {
  ModuleList list(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, ExtendPushesModulesFromOtherModuleList) {
  struct A : torch::nn::Module {};
  struct B : torch::nn::Module {};
  struct C : torch::nn::Module {};
  struct D : torch::nn::Module {};
  ModuleList a(A{}, B{});
  ModuleList b(C{}, D{});
  a->extend(*b);

  ASSERT_EQ(a->size(), 4);
  ASSERT_TRUE(a[0]->as<A>());
  ASSERT_TRUE(a[1]->as<B>());
  ASSERT_TRUE(a[2]->as<C>());
  ASSERT_TRUE(a[3]->as<D>());

  ASSERT_EQ(b->size(), 2);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());

  std::vector<std::shared_ptr<A>> c = {std::make_shared<A>(),
                                       std::make_shared<A>()};
  b->extend(c);

  ASSERT_EQ(b->size(), 4);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());
  ASSERT_TRUE(b[2]->as<A>());
  ASSERT_TRUE(b[3]->as<A>());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, HasReferenceSemantics) {
  ModuleList first(Linear(2, 3), Linear(4, 4), Linear(4, 5));
  ModuleList second(first);

  ASSERT_EQ(first.get(), second.get());
  ASSERT_EQ(first->size(), second->size());
  ASSERT_TRUE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const std::shared_ptr<Module>& first,
         const std::shared_ptr<Module>& second) {
        return first.get() == second.get();
      }));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, IsCloneable) {
  ModuleList list(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  ModuleList clone = std::dynamic_pointer_cast<ModuleListImpl>(list->clone());
  ASSERT_EQ(list->size(), clone->size());

  for (size_t i = 0; i < list->size(); ++i) {
    // The modules should be the same kind (type).
    ASSERT_EQ(list[i]->name(), clone[i]->name());
    // But not pointer-equal (distinct objects).
    ASSERT_NE(list[i], clone[i]);
  }

  // Verify that the clone is deep, i.e. parameters of modules are cloned too.

  torch::NoGradGuard no_grad;

  auto params1 = list->named_parameters();
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, RegistersElementsAsSubmodules) {
  ModuleList list(Linear(10, 3), Conv2d(1, 2, 3), Dropout2d(0.5));

  auto modules = list->children();
  ASSERT_TRUE(modules[0]->as<Linear>());
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  ASSERT_TRUE(modules[2]->as<Dropout2d>());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, NestingIsPossible) {
  ModuleList list(
      (ModuleList(Dropout(), Dropout())),
      (ModuleList(Dropout(), Dropout()), Dropout()));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, CloneToDevice_CUDA) {
  ModuleList list(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  torch::Device device(torch::kCUDA, 0);
  ModuleList clone =
      std::dynamic_pointer_cast<ModuleListImpl>(list->clone(device));
  for (const auto& p : clone->parameters()) {
    ASSERT_EQ(p.device(), device);
  }
  for (const auto& b : clone->buffers()) {
    ASSERT_EQ(b.device(), device);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, PrettyPrintModuleList) {
  ModuleList list(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
  ASSERT_EQ(
      c10::str(list),
      "torch::nn::ModuleList(\n"
      "  (0): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (1): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (2): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (3): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (4): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (5): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(ModuleListTest, RangeBasedForLoop) {
  torch::nn::ModuleList mlist(
    torch::nn::Linear(3, 4),
    torch::nn::BatchNorm1d(4),
    torch::nn::Dropout(0.5)
  );

  std::stringstream buffer;
  for (const auto &module : *mlist) {
    module->pretty_print(buffer);
  }
}
