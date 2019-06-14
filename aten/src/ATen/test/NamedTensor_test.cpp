#ifdef NAMEDTENSOR_ENABLED
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NamedTensor.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/memory.h>

using at::Dimname;
using at::NamedTensorMeta;
using at::Symbol;
using torch::make_unique;

TEST(NamedTensorTest, defaultMetadata) {
  int num_names = 4;
  const auto meta = NamedTensorMeta(num_names);
  for (const auto name : meta.names()) {
    ASSERT_EQ(name.type(), at::NameType::WILDCARD);
  }
}

static Dimname dimnameFromString(const std::string& str) {
  return Dimname::fromSymbol(Symbol::dimname(str));
}

TEST(NamedTensorTest, isNamed) {
  auto tensor = at::zeros({3, 2, 5, 7});
  ASSERT_FALSE(tensor.is_named());

  tensor = at::zeros({3, 2, 5, 7});
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(tensor.dim()));
  ASSERT_FALSE(tensor.is_named());

  tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(names));
  ASSERT_TRUE(tensor.is_named());
}

static bool dimnames_equal(at::DimnameList names, at::DimnameList other) {
  if (names.size() != other.size()) {
    return false;
  }
  for (auto i = 0; i < names.size(); i++) {
    const auto& name = names[i];
    const auto& other_name = other[i];
    if (name.type() != other_name.type() || name.name() != other_name.name()) {
      return false;
    }
  }
  return true;
}

TEST(NamedTensorTest, attachMetadata) {
  auto tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };

  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(names));
  
  const auto retrieved_meta = tensor.get_named_tensor_meta();
  ASSERT_TRUE(dimnames_equal(retrieved_meta->names(), names));

  // Test dropping metadata
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(nullptr);
  ASSERT_FALSE(tensor.is_named());
}

TEST(NamedTensorTest, internalSetNamesInplace) {
  auto tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  ASSERT_FALSE(tensor.is_named());

  // Set names
  at::internal_set_names_inplace(tensor, names);
  const auto retrieved_names = tensor.names().value();
  ASSERT_TRUE(dimnames_equal(retrieved_names, names));

  // Drop names
  at::internal_set_names_inplace(tensor, at::nullopt);
  ASSERT_TRUE(tensor.get_named_tensor_meta() == nullptr);
  ASSERT_TRUE(tensor.names() == at::nullopt);
}

TEST(NamedTensorTest, empty) {
  auto N = Dimname::fromSymbol(Symbol::dimname("N"));
  auto C = Dimname::fromSymbol(Symbol::dimname("C"));
  auto H = Dimname::fromSymbol(Symbol::dimname("H"));
  auto W = Dimname::fromSymbol(Symbol::dimname("W"));
  std::vector<Dimname> names = { N, C, H, W };

  auto tensor = at::empty({});
  ASSERT_EQ(tensor.names(), at::nullopt);

  tensor = at::empty({1, 2, 3});
  ASSERT_EQ(tensor.names(), at::nullopt);

  tensor = at::empty({1, 2, 3, 4}, names);
  ASSERT_TRUE(dimnames_equal(tensor.names().value(), names));

  ASSERT_THROW(at::empty({1, 2, 3}, names), c10::Error);
}
#endif
