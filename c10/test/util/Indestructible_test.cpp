#include <c10/util/Indestructible.h>
#include <gtest/gtest.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>

namespace {

struct Magic {
  std::function<void()> dtor_;
  std::function<void()> move_;
  Magic(std::function<void()> ctor, std::function<void()> dtor, std::function<void()> move)
      : dtor_(std::move(dtor)), move_(std::move(move)) {
    ctor();
  }
  Magic(Magic&& other) /* may throw */ { *this = std::move(other); }
  Magic& operator=(Magic&& other) {
    dtor_ = std::move(other.dtor_);
    move_ = std::move(other.move_);
    move_();
    return *this;
  }
  ~Magic() { dtor_(); }
};

template <typename T>
struct DeferredDtor {
  c10::Indestructible<T>& obj_;
  explicit constexpr DeferredDtor(c10::Indestructible<T>& obj) noexcept
      : obj_{obj} {}
  ~DeferredDtor() { obj_->~T(); }
};

class IndestructibleTest : public testing::Test {};
} // namespace

TEST(IndestructibleTest, access) {
  c10::Indestructible<std::map<std::string, int>> data{
      std::map<std::string, int>{{"key1", 17}, {"key2", 19}, {"key3", 23}}};
  DeferredDtor s{data};

  auto& m = *data;
  EXPECT_EQ(19, m.at("key2"));
}

TEST(IndestructibleTest, no_destruction) {
  int state = 0;
  int value = 0;

  c10::Indestructible<Magic> sing(
      [&] {
        ++state;
        value = 7;
      },
      [&] { state = -1; },
      [] {});
  EXPECT_EQ(1, state);
  EXPECT_EQ(7, value);

  sing.~Indestructible();
  EXPECT_EQ(1, state);
}

TEST(IndestructibleTest, empty) {
  const c10::Indestructible<std::map<std::string, int>> data;
  auto& m = *data;
  EXPECT_EQ(0, m.size());
}

TEST(IndestructibleTest, disabled_default_ctor) {
  EXPECT_TRUE((std::is_constructible<c10::Indestructible<int>>::value)) << "sanity";

  struct Foo {
    Foo(int) {}
  };
  EXPECT_FALSE((std::is_constructible<c10::Indestructible<Foo>>::value));
  EXPECT_FALSE((std::is_constructible<c10::Indestructible<Foo>, Magic>::value));
  EXPECT_TRUE((std::is_constructible<c10::Indestructible<Foo>, int>::value));
}

TEST(IndestructibleTest, list_initialization) {
  c10::Indestructible<std::map<int, int>> map{{{1, 2}}};
  DeferredDtor s{map};

  EXPECT_EQ(map->at(1), 2);
}

namespace {
class InitializerListConstructible {
 public:
  InitializerListConstructible(InitializerListConstructible&&) = default;
  explicit InitializerListConstructible(std::initializer_list<int>) {}
  InitializerListConstructible(std::initializer_list<double>, double) {}
};
} // namespace

TEST(IndestructibleTest, initializer_list_in_place_initialization) {
  using I = InitializerListConstructible;
  std::ignore = c10::Indestructible<I>{{1, 2, 3, 4}};
  std::ignore = c10::Indestructible<I>{{1.2}, 4.2};
}

namespace {
class ExplicitlyMoveConstructible {
 public:
  ExplicitlyMoveConstructible() = default;
  explicit ExplicitlyMoveConstructible(ExplicitlyMoveConstructible&&) = default;
};
} // namespace

TEST(IndestructibleTest, list_initialization_explicit_implicit) {
  using E = ExplicitlyMoveConstructible;
  using I = std::map<int, int>;
  EXPECT_TRUE((!std::is_convertible<E, c10::Indestructible<E>>::value));
  EXPECT_TRUE((std::is_convertible<I, c10::Indestructible<I>>::value));
}

TEST(IndestructibleTest, conversion) {
  using I = std::map<std::string, std::string>;
  c10::Indestructible<I> map{I{{"foo", "bar"}}};
  DeferredDtor s{map};
  I& r = map;
  EXPECT_EQ(1, r.count("foo"));
  I const& cr = std::as_const(map);
  EXPECT_EQ(1, cr.count("foo"));
}

TEST(IndestructibleTest, factory_method) {
  struct Foo {
    Foo(int x) : value(x) {}
    Foo(const Foo&) = delete;

    const int value;
  };

  auto factory = [] { return Foo(42); };

  c10::Indestructible<Foo> foo(c10::factory_constructor, factory);
  EXPECT_EQ(42, foo->value);
}
