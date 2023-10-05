// Originally taken from
// https://raw.githubusercontent.com/cryfs/cryfs/14ad22570ddacef22d5ff139cdff68a54fc8234d/test/cpp-utils/either_test.cpp

#include <c10/macros/Macros.h>
#include <c10/util/either.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

using c10::either;
using c10::make_left;
using c10::make_right;
using std::ostringstream;
using std::pair;
using std::string;
using std::tuple;
using std::vector;

namespace {
class MovableOnly final {
 public:
  explicit MovableOnly(int value) : _value(value) {}
  MovableOnly(const MovableOnly&) = delete;
  MovableOnly& operator=(const MovableOnly&) = delete;

  MovableOnly(MovableOnly&& rhs) : _value(rhs._value) {
    rhs._value = 0;
  }

  MovableOnly& operator=(MovableOnly&& rhs) {
    _value = rhs._value;
    rhs._value = 0;
    return *this;
  }

  int value() const {
    return _value;
  }

 private:
  int _value;
};

bool operator==(const MovableOnly& lhs, const MovableOnly& rhs) {
  return lhs.value() == rhs.value();
}

template <class T>
void test_with_matrix(
    std::vector<std::function<void(std::function<void(T&)>)>> setups,
    std::vector<std::function<void(T&)>> expectations) {
  for (const auto& setup : setups) {
    for (const auto& expectation : expectations) {
      setup(expectation);
    }
  }
}

template <class Left, class Right>
std::vector<std::function<void(either<Left, Right>&)>> EXPECT_IS_LEFT(
    const Left& expected) {
  return {
      [&](either<Left, Right>& obj) { EXPECT_TRUE(obj.is_left()); },
      [&](either<Left, Right>& obj) { EXPECT_FALSE(obj.is_right()); },
      [&](either<Left, Right>& obj) { EXPECT_EQ(expected, obj.left()); },
      [&](either<Left, Right>& obj) {
        EXPECT_EQ(expected, std::move(obj).left());
      },
      [&](either<Left, Right>& obj) {
        // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
        EXPECT_ANY_THROW(obj.right());
      },
      [&](either<Left, Right>& obj) {
        // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
        EXPECT_ANY_THROW(std::move(obj).right());
      }};
}

template <class Left, class Right>
std::vector<std::function<void(either<Left, Right>&)>> EXPECT_IS_RIGHT(
    const Right& expected) {
  return {
      [&](either<Left, Right>& obj) { EXPECT_FALSE(obj.is_left()); },
      [&](either<Left, Right>& obj) { EXPECT_TRUE(obj.is_right()); },
      [&](either<Left, Right>& obj) { EXPECT_EQ(expected, obj.right()); },
      [&](either<Left, Right>& obj) {
        EXPECT_EQ(expected, std::move(obj).right());
      },
      [&](either<Left, Right>& obj) {
        // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
        EXPECT_ANY_THROW(obj.left());
      },
      [&](either<Left, Right>& obj) {
        // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
        EXPECT_ANY_THROW(std::move(obj).left());
      }};
}

template <class Value>
std::vector<std::function<void(Value&)>> EXPECT_IS(const Value& v) {
  return {[&](Value& obj) { return obj == v; }};
}

template <typename T>
struct StoreWith1ByteFlag {
  T val;
  char flag;
};

template <typename Left, typename Right>
void TestSpaceUsage() {
  EXPECT_EQ(
      std::max(
          sizeof(StoreWith1ByteFlag<Left>), sizeof(StoreWith1ByteFlag<Right>)),
      sizeof(either<Left, Right>));
}
} // namespace

TEST(EitherTest, SpaceUsage) {
  TestSpaceUsage<char, int>();
  TestSpaceUsage<int, short>();
  TestSpaceUsage<char, short>();
  TestSpaceUsage<int, string>();
  TestSpaceUsage<string, vector<string>>();
}

TEST(EitherTest, givenLeft) {
  test_with_matrix(
      {
          [](std::function<void(either<int, string>&)> test) {
            either<int, string> a(4);
            test(a);
          },
          [](std::function<void(either<int, string>&)> test) {
            either<int, string> a = 4;
            test(a);
          },
      },
      EXPECT_IS_LEFT<int, string>(4));
}

TEST(EitherTest, givenRight) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
         either<int, string> a("4");
         test(a);
       },
       [](std::function<void(either<int, string>&)> test) {
         either<int, string> a = string("4");
         test(a);
       }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenMakeLeft) {
  test_with_matrix(
      {
          [](std::function<void(either<int, string>&)> test) {
            either<int, string> a = make_left<int, string>(4);
            test(a);
          },
          [](std::function<void(either<int, string>&)> test) {
            auto a = make_left<int, string>(4);
            test(a);
          },
      },
      EXPECT_IS_LEFT<int, string>(4));
}

TEST(EitherTest, givenMakeLeftWithSameType) {
  test_with_matrix(
      {
          [](std::function<void(either<int, int>&)> test) {
            either<int, int> a = make_left<int, int>(4);
            test(a);
          },
          [](std::function<void(either<int, int>&)> test) {
            auto a = make_left<int, int>(4);
            test(a);
          },
      },
      EXPECT_IS_LEFT<int, int>(4));
}

TEST(EitherTest, givenMakeRight) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
         either<int, string> a = make_right<int, string>("4");
         test(a);
       },
       [](std::function<void(either<int, string>&)> test) {
         auto a = make_right<int, string>("4");
         test(a);
       }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenMakeRightWithSameType) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_right<string, string>("4");
         test(a);
       },
       [](std::function<void(either<string, string>&)> test) {
         auto a = make_right<string, string>("4");
         test(a);
       }},
      EXPECT_IS_RIGHT<string, string>("4"));
}

TEST(EitherTest, givenMovableOnlyMakeLeft) {
  test_with_matrix(
      {
          [](std::function<void(either<MovableOnly, string>&)> test) {
            either<MovableOnly, string> a = make_left<MovableOnly, string>(3);
            test(a);
          },
          [](std::function<void(either<MovableOnly, string>&)> test) {
            auto a = make_left<MovableOnly, string>(3);
            test(a);
          },
      },
      EXPECT_IS_LEFT<MovableOnly, string>(MovableOnly(3)));
}

TEST(EitherTest, givenMovableOnlyMakeRight) {
  test_with_matrix(
      {[](std::function<void(either<int, MovableOnly>&)> test) {
         either<int, MovableOnly> a = make_right<int, MovableOnly>(3);
         test(a);
       },
       [](std::function<void(either<int, MovableOnly>&)> test) {
         auto a = make_right<int, MovableOnly>(3);
         test(a);
       }},
      EXPECT_IS_RIGHT<int, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenMultiParamMakeLeft) {
  test_with_matrix(
      {
          [](std::function<void(either<pair<int, int>, string>&)> test) {
            either<pair<int, int>, string> a =
                make_left<pair<int, int>, string>(5, 6);
            test(a);
          },
          [](std::function<void(either<pair<int, int>, string>&)> test) {
            auto a = make_left<pair<int, int>, string>(5, 6);
            test(a);
          },
      },
      EXPECT_IS_LEFT<pair<int, int>, string>(pair<int, int>(5, 6)));
}

TEST(EitherTest, givenMultiParamMakeRight) {
  test_with_matrix(
      {[](std::function<void(either<int, pair<int, int>>&)> test) {
         either<int, pair<int, int>> a = make_right<int, pair<int, int>>(5, 6);
         test(a);
       },
       [](std::function<void(either<int, pair<int, int>>&)> test) {
         auto a = make_right<int, pair<int, int>>(5, 6);
         test(a);
       }},
      EXPECT_IS_RIGHT<int, pair<int, int>>(pair<int, int>(5, 6)));
}

TEST(EitherTest, givenLeftCopyConstructedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, int>&)> test) {
        string a = "4";
        either<string, int> b(a);
        test(b);
      }},
      EXPECT_IS_LEFT<string, int>("4"));
}

TEST(EitherTest, givenLeftCopyConstructedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(string&)> test) {
        string a = "4";
        either<string, int> b(a);
        test(a);
      }},
      EXPECT_IS<string>("4"));
}

TEST(EitherTest, givenRightCopyConstructedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
        string a = "4";
        either<int, string> b(a);
        test(b);
      }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenRightCopyConstructedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(string&)> test) {
        string a = "4";
        either<int, string> b(a);
        test(a);
      }},
      EXPECT_IS<string>("4"));
}

TEST(EitherTest, givenLeftMoveConstructedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, int>&)> test) {
        MovableOnly a(3);
        either<MovableOnly, int> b(std::move(a));
        test(b);
      }},
      EXPECT_IS_LEFT<MovableOnly, int>(MovableOnly(3)));
}

TEST(EitherTest, givenLeftMoveConstructedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(MovableOnly&)> test) {
        MovableOnly a(3);
        either<MovableOnly, int> b(std::move(a));
        test(a); // NOLINT(bugprone-use-after-move)
      }},
      EXPECT_IS<MovableOnly>(MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenRightMoveConstructedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, MovableOnly>&)> test) {
        MovableOnly a(3);
        either<int, MovableOnly> b(std::move(a));
        test(b);
      }},
      EXPECT_IS_RIGHT<int, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenRightMoveConstructedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(MovableOnly&)> test) {
        MovableOnly a(3);
        either<int, MovableOnly> b(std::move(a));
        test(a); // NOLINT(bugprone-use-after-move)
      }},
      EXPECT_IS<MovableOnly>(MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenLeftCopyAssignedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, int>&)> test) {
         string a = "4";
         either<string, int> b(2);
         b = a;
         test(b);
       },
       [](std::function<void(either<string, int>&)> test) {
         string a = "4";
         either<string, int> b("2");
         b = a;
         test(b);
       }},
      EXPECT_IS_LEFT<string, int>("4"));
}

TEST(EitherTest, givenLeftCopyAssignedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(string&)> test) {
         string a = "4";
         either<string, int> b(2);
         b = a;
         test(a);
       },
       [](std::function<void(string&)> test) {
         string a = "4";
         either<string, int> b("2");
         b = a;
         test(a);
       }},
      EXPECT_IS<string>("4"));
}

TEST(EitherTest, givenRightCopyAssignedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
         string a = "4";
         either<int, string> b(2);
         b = a;
         test(b);
       },
       [](std::function<void(either<int, string>&)> test) {
         string a = "4";
         either<int, string> b("2");
         b = a;
         test(b);
       }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenRightCopyAssignedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(string&)> test) {
         string a = "4";
         either<int, string> b(2);
         b = a;
         test(a);
       },
       [](std::function<void(string&)> test) {
         string a = "4";
         either<int, string> b("2");
         b = a;
         test(a);
       }},
      EXPECT_IS<string>("4"));
}

TEST(EitherTest, givenLeftMoveAssignedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, string>&)> test) {
         MovableOnly a(3);
         either<MovableOnly, string> b(2);
         b = std::move(a);
         test(b);
       },
       [](std::function<void(either<MovableOnly, string>&)> test) {
         MovableOnly a(3);
         either<MovableOnly, string> b(MovableOnly(2));
         b = std::move(a);
         test(b);
       }},
      EXPECT_IS_LEFT<MovableOnly, string>(MovableOnly(3)));
}

TEST(EitherTest, givenLeftMoveAssignedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(MovableOnly&)> test) {
         MovableOnly a(3);
         either<MovableOnly, string> b("2");
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       },
       [](std::function<void(MovableOnly&)> test) {
         MovableOnly a(3);
         either<MovableOnly, string> b(MovableOnly(0));
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       }},
      EXPECT_IS<MovableOnly>(MovableOnly(0)));
}

TEST(EitherTest, givenRightMoveAssignedFromValue_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, MovableOnly>&)> test) {
         MovableOnly a(3);
         either<string, MovableOnly> b("2");
         b = std::move(a);
         test(b);
       },
       [](std::function<void(either<string, MovableOnly>&)> test) {
         MovableOnly a(3);
         either<string, MovableOnly> b(MovableOnly(2));
         b = std::move(a);
         test(b);
       }},
      EXPECT_IS_RIGHT<string, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenRightMoveAssignedFromValue_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(MovableOnly&)> test) {
         MovableOnly a(3);
         either<string, MovableOnly> b("2");
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       },
       [](std::function<void(MovableOnly&)> test) {
         MovableOnly a(3);
         either<string, MovableOnly> b(MovableOnly(2));
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       }},
      EXPECT_IS<MovableOnly>(MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenLeftCopyConstructed_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, int>&)> test) {
        either<string, int> a("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<string, int> b(a);
        test(b);
      }},
      EXPECT_IS_LEFT<string, int>("4"));
}

TEST(EitherTest, givenLeftCopyConstructed_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, int>&)> test) {
        either<string, int> a("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<string, int> b(a);
        test(a);
      }},
      EXPECT_IS_LEFT<string, int>("4"));
}

TEST(EitherTest, givenLeftCopyConstructed_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
        either<string, string> a = make_left<string, string>("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<string, string> b(a);
        test(b);
      }},
      EXPECT_IS_LEFT<string, string>("4"));
}

TEST(EitherTest, givenLeftCopyConstructed_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
        either<string, string> a = make_left<string, string>("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<string, string> b(a);
        test(a);
      }},
      EXPECT_IS_LEFT<string, string>("4"));
}

TEST(EitherTest, givenRightCopyConstructed_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
        either<int, string> a("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<int, string> b(a);
        test(b);
      }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenRightCopyConstructed_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
        either<int, string> a("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<int, string> b(a);
        test(a);
      }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenRightCopyConstructed_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
        either<string, string> a = make_right<string, string>("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<string, string> b(a);
        test(b);
      }},
      EXPECT_IS_RIGHT<string, string>("4"));
}

TEST(EitherTest, givenRightCopyConstructed_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
        either<string, string> a = make_right<string, string>("4");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        either<string, string> b(a);
        test(a);
      }},
      EXPECT_IS_RIGHT<string, string>("4"));
}

TEST(EitherTest, givenLeftMoveConstructed_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, int>&)> test) {
        either<MovableOnly, int> a(MovableOnly(3));
        either<MovableOnly, int> b(std::move(a));
        test(b);
      }},
      EXPECT_IS_LEFT<MovableOnly, int>(MovableOnly(3)));
}

TEST(EitherTest, givenLeftMoveConstructed_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, int>&)> test) {
        either<MovableOnly, int> a(MovableOnly(3));
        either<MovableOnly, int> b(std::move(a));
        test(a); // NOLINT(bugprone-use-after-move)
      }},
      EXPECT_IS_LEFT<MovableOnly, int>(MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenLeftMoveConstructed_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
        either<MovableOnly, MovableOnly> a =
            make_left<MovableOnly, MovableOnly>(MovableOnly(3));
        either<MovableOnly, MovableOnly> b(std::move(a));
        test(b);
      }},
      EXPECT_IS_LEFT<MovableOnly, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenLeftMoveConstructed_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
        either<MovableOnly, MovableOnly> a =
            make_left<MovableOnly, MovableOnly>(MovableOnly(3));
        either<MovableOnly, MovableOnly> b(std::move(a));
        test(a); // NOLINT(bugprone-use-after-move)
      }},
      EXPECT_IS_LEFT<MovableOnly, MovableOnly>(
          MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenRightMoveConstructed_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, MovableOnly>&)> test) {
        either<int, MovableOnly> a(MovableOnly(3));
        either<int, MovableOnly> b(std::move(a));
        test(b);
      }},
      EXPECT_IS_RIGHT<int, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenRightMoveConstructed_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, MovableOnly>&)> test) {
        either<int, MovableOnly> a(MovableOnly(3));
        either<int, MovableOnly> b(std::move(a));
        test(a); // NOLINT(bugprone-use-after-move)
      }},
      EXPECT_IS_RIGHT<int, MovableOnly>(MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenRightMoveConstructed_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
        either<MovableOnly, MovableOnly> a =
            make_right<MovableOnly, MovableOnly>(MovableOnly(3));
        either<MovableOnly, MovableOnly> b(std::move(a));
        test(b);
      }},
      EXPECT_IS_RIGHT<MovableOnly, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenRightMoveConstructed_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
        either<MovableOnly, MovableOnly> a =
            make_right<MovableOnly, MovableOnly>(MovableOnly(3));
        either<MovableOnly, MovableOnly> b(std::move(a));
        test(a); // NOLINT(bugprone-use-after-move)
      }},
      EXPECT_IS_RIGHT<MovableOnly, MovableOnly>(
          MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenLeftCopyAssigned_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, int>&)> test) {
         either<string, int> a("4");
         either<string, int> b(2);
         b = a;
         test(b);
       },
       [](std::function<void(either<string, int>&)> test) {
         either<string, int> a("4");
         either<string, int> b("2");
         b = a;
         test(b);
       }},
      EXPECT_IS_LEFT<string, int>("4"));
}

TEST(EitherTest, givenLeftCopyAssigned_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, int>&)> test) {
         either<string, int> a("4");
         either<string, int> b(2);
         b = a;
         test(a);
       },
       [](std::function<void(either<string, int>&)> test) {
         either<string, int> a("4");
         either<string, int> b("2");
         b = a;
         test(a);
       }},
      EXPECT_IS_LEFT<string, int>("4"));
}

TEST(EitherTest, givenLeftCopyAssigned_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_left<string, string>("4");
         either<string, string> b = make_right<string, string>("2");
         b = a;
         test(b);
       },
       [](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_left<string, string>("4");
         either<string, string> b = make_left<string, string>("2");
         b = a;
         test(b);
       }},
      EXPECT_IS_LEFT<string, string>("4"));
}

TEST(EitherTest, givenLeftCopyAssigned_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_left<string, string>("4");
         either<string, string> b = make_right<string, string>("2");
         b = a;
         test(a);
       },
       [](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_left<string, string>("4");
         either<string, string> b = make_left<string, string>("2");
         b = a;
         test(a);
       }},
      EXPECT_IS_LEFT<string, string>("4"));
}

TEST(EitherTest, givenRightCopyAssigned_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
         either<int, string> a("4");
         either<int, string> b(2);
         b = a;
         test(b);
       },
       [](std::function<void(either<int, string>&)> test) {
         either<int, string> a("4");
         either<int, string> b("2");
         b = a;
         test(b);
       }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenRightCopyAssigned_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
         either<int, string> a("4");
         either<int, string> b(2);
         b = a;
         test(a);
       },
       [](std::function<void(either<int, string>&)> test) {
         either<int, string> a("4");
         either<int, string> b("2");
         b = a;
         test(a);
       }},
      EXPECT_IS_RIGHT<int, string>("4"));
}

TEST(EitherTest, givenRightCopyAssigned_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_right<string, string>("4");
         either<string, string> b = make_left<string, string>("2");
         b = a;
         test(b);
       },
       [](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_right<string, string>("4");
         either<string, string> b = make_right<string, string>("2");
         b = a;
         test(b);
       }},
      EXPECT_IS_RIGHT<string, string>("4"));
}

TEST(EitherTest, givenRightCopyAssigned_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_right<string, string>("4");
         either<string, string> b = make_left<string, string>("2");
         b = a;
         test(a);
       },
       [](std::function<void(either<string, string>&)> test) {
         either<string, string> a = make_right<string, string>("4");
         either<string, string> b = make_right<string, string>("2");
         b = a;
         test(a);
       }},
      EXPECT_IS_RIGHT<string, string>("4"));
}

TEST(EitherTest, givenLeftMoveAssigned_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, string>&)> test) {
         either<MovableOnly, string> a(MovableOnly(3));
         either<MovableOnly, string> b(2);
         b = std::move(a);
         test(b);
       },
       [](std::function<void(either<MovableOnly, string>&)> test) {
         either<MovableOnly, string> a(MovableOnly(3));
         either<MovableOnly, string> b(MovableOnly(2));
         b = std::move(a);
         test(b);
       }},
      EXPECT_IS_LEFT<MovableOnly, string>(MovableOnly(3)));
}

TEST(EitherTest, givenLeftMoveAssigned_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, string>&)> test) {
         either<MovableOnly, string> a(MovableOnly(3));
         either<MovableOnly, string> b(2);
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       },
       [](std::function<void(either<MovableOnly, string>&)> test) {
         either<MovableOnly, string> a(MovableOnly(3));
         either<MovableOnly, string> b(MovableOnly(2));
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       }},
      EXPECT_IS_LEFT<MovableOnly, string>(
          MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenLeftMoveAssigned_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_left<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_right<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(b);
       },
       [](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_left<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_left<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(b);
       }},
      EXPECT_IS_LEFT<MovableOnly, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenLeftMoveAssigned_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_left<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_right<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       },
       [](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_left<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_left<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       }},
      EXPECT_IS_LEFT<MovableOnly, MovableOnly>(
          MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenRightMoveAssigned_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, MovableOnly>&)> test) {
         either<string, MovableOnly> a(MovableOnly(3));
         either<string, MovableOnly> b("2");
         b = std::move(a);
         test(b);
       },
       [](std::function<void(either<string, MovableOnly>&)> test) {
         either<string, MovableOnly> a(MovableOnly(3));
         either<string, MovableOnly> b(MovableOnly(2));
         b = std::move(a);
         test(b);
       }},
      EXPECT_IS_RIGHT<string, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenRightMoveAssigned_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<string, MovableOnly>&)> test) {
         either<string, MovableOnly> a(MovableOnly(3));
         either<string, MovableOnly> b("2");
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       },
       [](std::function<void(either<string, MovableOnly>&)> test) {
         either<string, MovableOnly> a(MovableOnly(3));
         either<string, MovableOnly> b(MovableOnly(2));
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       }},
      EXPECT_IS_RIGHT<string, MovableOnly>(
          MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenRightMoveAssigned_withSameType_thenNewIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_right<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_left<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(b);
       },
       [](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_right<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_right<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(b);
       }},
      EXPECT_IS_RIGHT<MovableOnly, MovableOnly>(MovableOnly(3)));
}

TEST(EitherTest, givenRightMoveAssigned_withSameType_thenOldIsCorrect) {
  test_with_matrix(
      {[](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_right<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_left<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       },
       [](std::function<void(either<MovableOnly, MovableOnly>&)> test) {
         either<MovableOnly, MovableOnly> a =
             make_right<MovableOnly, MovableOnly>(3);
         either<MovableOnly, MovableOnly> b =
             make_right<MovableOnly, MovableOnly>(2);
         b = std::move(a);
         test(a); // NOLINT(bugprone-use-after-move)
       }},
      EXPECT_IS_RIGHT<MovableOnly, MovableOnly>(
          MovableOnly(0)) // 0 is moved-from value
  );
}

TEST(EitherTest, givenLeft_whenModified_thenValueIsChanged) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
         either<int, string> a(4);
         a.left() = 5;
         test(a);
       },
       [](std::function<void(either<int, string>&)> test) {
         either<int, string> a(4);
         a.left() = 5;
         test(a);
       }},
      EXPECT_IS_LEFT<int, string>(5));
}

TEST(EitherTest, givenRight_whenModified_thenValueIsChanged) {
  test_with_matrix(
      {[](std::function<void(either<int, string>&)> test) {
         either<int, string> a("4");
         a.right() = "5";
         test(a);
       },
       [](std::function<void(either<int, string>&)> test) {
         either<int, string> a("4");
         a.right() = "5";
         test(a);
       }},
      EXPECT_IS_RIGHT<int, string>("5"));
}

TEST(EitherTest, canEmplaceConstructLeft) {
  test_with_matrix(
      {[](std::function<void(either<tuple<int, int>, tuple<int, string, int>>&)>
              test) {
        either<tuple<int, int>, tuple<int, string, int>> a(2, 3);
        test(a);
      }},
      EXPECT_IS_LEFT<tuple<int, int>, tuple<int, string, int>>(
          tuple<int, int>(2, 3)));
}

TEST(EitherTest, canEmplaceConstructRight) {
  test_with_matrix(
      {[](std::function<void(either<tuple<int, int>, tuple<int, string, int>>&)>
              test) {
        either<tuple<int, int>, tuple<int, string, int>> a(2, "3", 4);
        test(a);
      }},
      EXPECT_IS_RIGHT<tuple<int, int>, tuple<int, string, int>>(
          tuple<int, string, int>(2, "3", 4)));
}

TEST(EitherTest, givenEqualLefts_thenAreEqual) {
  either<string, int> a("3");
  either<string, int> b("3");
  EXPECT_TRUE(a == b);
}

TEST(EitherTest, givenEqualLefts_thenAreNotUnequal) {
  either<string, int> a("3");
  either<string, int> b("3");
  EXPECT_FALSE(a != b);
}

TEST(EitherTest, givenEqualRights_thenAreEqual) {
  either<string, int> a(3);
  either<string, int> b(3);
  EXPECT_TRUE(a == b);
}

TEST(EitherTest, givenEqualRights_thenAreNotUnequal) {
  either<string, int> a(3);
  either<string, int> b(3);
  EXPECT_FALSE(a != b);
}

TEST(EitherTest, givenLeftAndRight_thenAreNotEqual) {
  either<string, int> a("3");
  either<string, int> b(3);
  EXPECT_FALSE(a == b);
  EXPECT_FALSE(b == a);
}

TEST(EitherTest, givenLeftAndRight_thenAreUnequal) {
  either<string, int> a("3");
  either<string, int> b(3);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(b != a);
}

TEST(EitherTest, OutputLeft) {
  ostringstream str;
  str << either<string, int>("mystring");
  EXPECT_EQ("Left(mystring)", str.str());
}

TEST(EitherTest, OutputRight) {
  ostringstream str;
  str << either<int, string>("mystring");
  EXPECT_EQ("Right(mystring)", str.str());
}

TEST(EitherTest, givenLeftAndRightWithSameType_thenAreNotEqual) {
  either<string, string> a = make_left<string, string>("3");
  either<string, string> b = make_right<string, string>("3");
  EXPECT_FALSE(a == b);
  EXPECT_FALSE(b == a);
}

TEST(EitherTest, givenLeftAndRightWithSameType_thenAreUnequal) {
  either<string, string> a = make_left<string, string>("3");
  either<string, string> b = make_right<string, string>("3");
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(b != a);
}

namespace {
class DestructorCallback {
 public:
  MOCK_CONST_METHOD0(call, void());

  void EXPECT_CALLED(int times = 1) {
    EXPECT_CALL(*this, call()).Times(times);
  }
};
class ClassWithDestructorCallback {
 public:
  ClassWithDestructorCallback(const DestructorCallback* destructorCallback)
      : _destructorCallback(destructorCallback) {}

  ~ClassWithDestructorCallback() {
    _destructorCallback->call();
  }

  ClassWithDestructorCallback& operator=(
      const ClassWithDestructorCallback& rhs) = delete;

 private:
  const DestructorCallback* _destructorCallback;
};
class OnlyMoveableClassWithDestructorCallback {
 public:
  OnlyMoveableClassWithDestructorCallback(
      const DestructorCallback* destructorCallback)
      : _destructorCallback(destructorCallback) {}
  OnlyMoveableClassWithDestructorCallback(
      OnlyMoveableClassWithDestructorCallback&& source)
      : _destructorCallback(source._destructorCallback) {}

  ~OnlyMoveableClassWithDestructorCallback() {
    _destructorCallback->call();
  }

 private:
  C10_DISABLE_COPY_AND_ASSIGN(OnlyMoveableClassWithDestructorCallback);
  const DestructorCallback* _destructorCallback;
};

} // namespace

TEST(EitherTestDestructor, LeftDestructorIsCalled) {
  DestructorCallback destructorCallback;
  destructorCallback.EXPECT_CALLED(
      2); // Once for the temp object, once when the either class destructs

  ClassWithDestructorCallback temp(&destructorCallback);
  either<ClassWithDestructorCallback, string> var = temp;
}

TEST(EitherTestDestructor, RightDestructorIsCalled) {
  DestructorCallback destructorCallback;
  destructorCallback.EXPECT_CALLED(
      2); // Once for the temp object, once when the either class destructs

  ClassWithDestructorCallback temp(&destructorCallback);
  either<string, ClassWithDestructorCallback> var = temp;
}

TEST(EitherTestDestructor, LeftDestructorIsCalledAfterCopying) {
  DestructorCallback destructorCallback;
  destructorCallback.EXPECT_CALLED(
      3); // Once for the temp object, once for var1 and once for var2

  ClassWithDestructorCallback temp(&destructorCallback);
  either<ClassWithDestructorCallback, string> var1 = temp;
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  either<ClassWithDestructorCallback, string> var2 = var1;
}

TEST(EitherTestDestructor, RightDestructorIsCalledAfterCopying) {
  DestructorCallback destructorCallback;
  destructorCallback.EXPECT_CALLED(
      3); // Once for the temp object, once for var1 and once for var2

  ClassWithDestructorCallback temp(&destructorCallback);
  either<string, ClassWithDestructorCallback> var1 = temp;
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  either<string, ClassWithDestructorCallback> var2 = var1;
}

TEST(EitherTestDestructor, LeftDestructorIsCalledAfterMoving) {
  DestructorCallback destructorCallback;
  destructorCallback.EXPECT_CALLED(
      3); // Once for the temp object, once for var1 and once for var2

  OnlyMoveableClassWithDestructorCallback temp(&destructorCallback);
  either<OnlyMoveableClassWithDestructorCallback, string> var1 =
      std::move(temp);
  either<OnlyMoveableClassWithDestructorCallback, string> var2 =
      std::move(var1);
}

TEST(EitherTestDestructor, RightDestructorIsCalledAfterMoving) {
  DestructorCallback destructorCallback;
  destructorCallback.EXPECT_CALLED(
      3); // Once for the temp object, once for var1 and once for var2

  OnlyMoveableClassWithDestructorCallback temp(&destructorCallback);
  either<string, OnlyMoveableClassWithDestructorCallback> var1 =
      std::move(temp);
  either<string, OnlyMoveableClassWithDestructorCallback> var2 =
      std::move(var1);
}

TEST(EitherTestDestructor, LeftDestructorIsCalledAfterAssignment) {
  DestructorCallback destructorCallback1;
  DestructorCallback destructorCallback2;
  destructorCallback1.EXPECT_CALLED(
      2); // Once for the temp1 object, once at the assignment
  destructorCallback2.EXPECT_CALLED(
      3); // Once for the temp2 object, once in destructor of var2, once in
          // destructor of var1

  ClassWithDestructorCallback temp1(&destructorCallback1);
  either<ClassWithDestructorCallback, string> var1 = temp1;
  ClassWithDestructorCallback temp2(&destructorCallback2);
  either<ClassWithDestructorCallback, string> var2 = temp2;
  var1 = var2;
}

TEST(EitherTestDestructor, RightDestructorIsCalledAfterAssignment) {
  DestructorCallback destructorCallback1;
  DestructorCallback destructorCallback2;
  destructorCallback1.EXPECT_CALLED(
      2); // Once for the temp1 object, once at the assignment
  destructorCallback2.EXPECT_CALLED(
      3); // Once for the temp2 object, once in destructor of var2, once in
          // destructor of var1

  ClassWithDestructorCallback temp1(&destructorCallback1);
  either<string, ClassWithDestructorCallback> var1 = temp1;
  ClassWithDestructorCallback temp2(&destructorCallback2);
  either<string, ClassWithDestructorCallback> var2 = temp2;
  var1 = var2;
}

TEST(EitherTestDestructor, LeftDestructorIsCalledAfterMoveAssignment) {
  DestructorCallback destructorCallback1;
  DestructorCallback destructorCallback2;
  destructorCallback1.EXPECT_CALLED(
      2); // Once for the temp1 object, once at the assignment
  destructorCallback2.EXPECT_CALLED(
      3); // Once for the temp2 object, once in destructor of var2, once in
          // destructor of var1

  OnlyMoveableClassWithDestructorCallback temp1(&destructorCallback1);
  either<OnlyMoveableClassWithDestructorCallback, string> var1 =
      std::move(temp1);
  OnlyMoveableClassWithDestructorCallback temp2(&destructorCallback2);
  either<OnlyMoveableClassWithDestructorCallback, string> var2 =
      std::move(temp2);
  var1 = std::move(var2);
}

TEST(EitherTestDestructor, RightDestructorIsCalledAfterMoveAssignment) {
  DestructorCallback destructorCallback1;
  DestructorCallback destructorCallback2;
  destructorCallback1.EXPECT_CALLED(
      2); // Once for the temp1 object, once at the assignment
  destructorCallback2.EXPECT_CALLED(
      3); // Once for the temp2 object, once in destructor of var2, once in
          // destructor of var1

  OnlyMoveableClassWithDestructorCallback temp1(&destructorCallback1);
  either<string, OnlyMoveableClassWithDestructorCallback> var1 =
      std::move(temp1);
  OnlyMoveableClassWithDestructorCallback temp2(&destructorCallback2);
  either<string, OnlyMoveableClassWithDestructorCallback> var2 =
      std::move(temp2);
  var1 = std::move(var2);
}
