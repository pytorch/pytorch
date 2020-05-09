#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorNames.h>
#include <c10/util/Exception.h>
#include <c10/util/C++17.h>

using at::Dimname;
using at::DimnameList;
using at::NamedTensorMeta;
using at::Symbol;
using at::namedinference::TensorName;
using at::namedinference::TensorNames;
using std::make_unique;

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
  ASSERT_FALSE(tensor.has_names());

  tensor = at::zeros({3, 2, 5, 7});
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(tensor.dim()));
  ASSERT_FALSE(tensor.has_names());

  tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(names));
  ASSERT_TRUE(tensor.has_names());
}

static bool dimnames_equal(at::DimnameList names, at::DimnameList other) {
  if (names.size() != other.size()) {
    return false;
  }
  for (auto i = 0; i < names.size(); i++) {
    const auto& name = names[i];
    const auto& other_name = other[i];
    if (name.type() != other_name.type() || name.symbol() != other_name.symbol()) {
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
  ASSERT_FALSE(tensor.has_names());
}

TEST(NamedTensorTest, internalSetNamesInplace) {
  auto tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  ASSERT_FALSE(tensor.has_names());

  // Set names
  at::internal_set_names_inplace(tensor, names);
  const auto retrieved_names = tensor.opt_names().value();
  ASSERT_TRUE(dimnames_equal(retrieved_names, names));

  // Drop names
  at::internal_set_names_inplace(tensor, at::nullopt);
  ASSERT_TRUE(tensor.get_named_tensor_meta() == nullptr);
  ASSERT_TRUE(tensor.opt_names() == at::nullopt);
}

TEST(NamedTensorTest, empty) {
  auto N = Dimname::fromSymbol(Symbol::dimname("N"));
  auto C = Dimname::fromSymbol(Symbol::dimname("C"));
  auto H = Dimname::fromSymbol(Symbol::dimname("H"));
  auto W = Dimname::fromSymbol(Symbol::dimname("W"));
  std::vector<Dimname> names = { N, C, H, W };

  auto tensor = at::empty({});
  ASSERT_EQ(tensor.opt_names(), at::nullopt);

  tensor = at::empty({1, 2, 3});
  ASSERT_EQ(tensor.opt_names(), at::nullopt);

  tensor = at::empty({1, 2, 3, 4}, names);
  ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), names));

  ASSERT_THROW(at::empty({1, 2, 3}, names), c10::Error);
}

TEST(NamedTensorTest, dimnameToPosition) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };

  auto tensor = at::empty({1, 1, 1});
  ASSERT_THROW(dimname_to_position(tensor, N), c10::Error);

  tensor = at::empty({1, 1, 1, 1}, names);
  ASSERT_EQ(dimname_to_position(tensor, H), 2);
}

static std::vector<Dimname> tensornames_unify_from_right(
    DimnameList names,
    DimnameList other_names) {
  auto names_wrapper = at::namedinference::TensorNames(names);
  auto other_wrapper = at::namedinference::TensorNames(other_names);
  return names_wrapper.unifyFromRightInplace(other_wrapper).toDimnameVec();
}

static void check_unify(
    DimnameList names,
    DimnameList other_names,
    DimnameList expected) {
  // Check legacy at::unify_from_right
  const auto result = at::unify_from_right(names, other_names);
  ASSERT_TRUE(dimnames_equal(result, expected));

  // Check with TensorNames::unifyFromRight.
  // In the future we'll merge at::unify_from_right and
  // TensorNames::unifyFromRight, but for now, let's test them both.
  const auto also_result = tensornames_unify_from_right(names, other_names);
  ASSERT_TRUE(dimnames_equal(also_result, expected));
}

static void check_unify_error(DimnameList names, DimnameList other_names) {
  // In the future we'll merge at::unify_from_right and
  // TensorNames::unifyFromRight. For now, test them both.
  ASSERT_THROW(at::unify_from_right(names, other_names), c10::Error);
  ASSERT_THROW(tensornames_unify_from_right(names, other_names), c10::Error);
}

TEST(NamedTensorTest, unifyFromRight) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  auto None = dimnameFromString("*");

  std::vector<Dimname> names = { N, C };

  check_unify({ N, C, H, W }, { N, C, H, W }, { N, C, H, W });
  check_unify({ W }, { C, H, W }, { C, H, W });
  check_unify({ None, W }, { C, H, W }, { C, H, W });
  check_unify({ None, None, H, None }, { C, None, W }, { None, C, H, W });

  check_unify_error({ W, H }, { W, C });
  check_unify_error({ W, H }, { C, H });
  check_unify_error({ None, H }, { H, None });
  check_unify_error({ H, None, C }, { H });
}

TEST(NamedTensorTest, alias) {
  // tensor.alias is not exposed in Python so we test its name propagation here
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  std::vector<Dimname> names = { N, C };

  auto tensor = at::empty({2, 3}, std::vector<Dimname>{ N, C });
  auto aliased = tensor.alias();
  ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), aliased.opt_names().value()));
}

TEST(NamedTensorTest, NoNamesGuard) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  std::vector<Dimname> names = { N, C };

  auto tensor = at::empty({2, 3}, names);
  ASSERT_TRUE(at::NamesMode::is_enabled());
  {
    at::NoNamesGuard guard;
    ASSERT_FALSE(at::NamesMode::is_enabled());
    ASSERT_FALSE(tensor.opt_names());
    ASSERT_FALSE(at::impl::get_opt_names(tensor.unsafeGetTensorImpl()));
  }
  ASSERT_TRUE(at::NamesMode::is_enabled());
}

static std::vector<Dimname> nchw() {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  return { N, C, H, W };
}

TEST(NamedTensorTest, TensorNamePrint) {
  auto names = nchw();
  {
    auto N = TensorName(names, 0);
    ASSERT_EQ(
        c10::str(N),
        "'N' (index 0 of ['N', 'C', 'H', 'W'])");
  }
  {
    auto H = TensorName(names, 2);
    ASSERT_EQ(
        c10::str(H),
        "'H' (index 2 of ['N', 'C', 'H', 'W'])");
  }
}

TEST(NamedTensorTest, TensorNamesCheckUnique) {
  auto names = nchw();
  {
    // smoke test to check that this doesn't throw
    TensorNames(names).checkUnique("op_name");
  }
  {
    std::vector<Dimname> nchh = { names[0], names[1], names[2], names[2] };
    auto tensornames = TensorNames(nchh);
    ASSERT_THROW(tensornames.checkUnique("op_name"), c10::Error);
  }
}


