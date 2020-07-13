#include <c10/util/Metaprogramming.h>
#include <c10/test/util/Macros.h>
#include <gtest/gtest.h>


using namespace c10::guts;

namespace {

namespace test_function_traits {
    static_assert(std::is_same<void, typename function_traits<void(int, float)>::return_type>::value, "");
    static_assert(std::is_same<int, typename function_traits<int(int, float)>::return_type>::value, "");
    static_assert(std::is_same<typelist::typelist<int, float>, typename function_traits<void(int, float)>::parameter_types>::value, "");
    static_assert(std::is_same<typelist::typelist<int, float>, typename function_traits<int(int, float)>::parameter_types>::value, "");
}

struct MovableOnly {
    constexpr MovableOnly(int val_): val(val_) {/* no default constructor */}
    MovableOnly(const MovableOnly&) = delete;
    MovableOnly(MovableOnly&&) = default;
    MovableOnly& operator=(const MovableOnly&) = delete;
    MovableOnly& operator=(MovableOnly&&) = default;

    friend bool operator==(const MovableOnly& lhs, const MovableOnly& rhs) {return lhs.val == rhs.val;}
private:
    int val;
};

template<class T> using is_my_movable_only_class = std::is_same<MovableOnly, std::remove_cv_t<std::remove_reference_t<T>>>;

struct CopyCounting {
    int move_count;
    int copy_count;

    CopyCounting(): move_count(0), copy_count(0) {}
    CopyCounting(const CopyCounting& rhs): move_count(rhs.move_count), copy_count(rhs.copy_count + 1) {}
    CopyCounting(CopyCounting&& rhs): move_count(rhs.move_count + 1), copy_count(rhs.copy_count) {}
    CopyCounting& operator=(const CopyCounting& rhs) {
        move_count = rhs.move_count;
        copy_count = rhs.copy_count + 1;
        return *this;
    }
    CopyCounting& operator=(CopyCounting&& rhs) {
        move_count = rhs.move_count + 1;
        copy_count = rhs.copy_count;
        return *this;
    }
};

template<class T> using is_my_copy_counting_class = std::is_same<CopyCounting, std::remove_cv_t<std::remove_reference_t<T>>>;

namespace test_extract_arg_by_filtered_index {
    class MyClass {};

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex) {
        auto a1 = extract_arg_by_filtered_index<std::is_integral, 0>(3, "bla", MyClass(), 4, nullptr, 5);
        auto a2 = extract_arg_by_filtered_index<std::is_integral, 1>(3, "bla", MyClass(), 4, nullptr, 5);
        auto a3 = extract_arg_by_filtered_index<std::is_integral, 2>(3, "bla", MyClass(), 4, nullptr, 5);
        EXPECT_EQ(3, a1);
        EXPECT_EQ(4, a2);
        EXPECT_EQ(5, a3);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_singleInput) {
        auto a1 = extract_arg_by_filtered_index<std::is_integral, 0>(3);
        EXPECT_EQ(3, a1);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_movableOnly) {
        MovableOnly a1 = extract_arg_by_filtered_index<is_my_movable_only_class, 0>(3, MovableOnly(3), "test", MovableOnly(1));
        MovableOnly a2 = extract_arg_by_filtered_index<is_my_movable_only_class, 1>(3, MovableOnly(3), "test", MovableOnly(1));
        EXPECT_EQ(MovableOnly(3), a1);
        EXPECT_EQ(MovableOnly(1), a2);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_onlyCopiesIfNecessary) {
        CopyCounting source;
        CopyCounting source2;
        CopyCounting a1 = extract_arg_by_filtered_index<is_my_copy_counting_class, 0>(3, CopyCounting(), "test", source, std::move(source2));
        CopyCounting a2 = extract_arg_by_filtered_index<is_my_copy_counting_class, 1>(3, CopyCounting(), "test", source, std::move(source2));
        CopyCounting a3 = extract_arg_by_filtered_index<is_my_copy_counting_class, 2>(3, CopyCounting(), "test", source, std::move(source2));
        EXPECT_EQ(1, a1.move_count);
        EXPECT_EQ(0, a1.copy_count);
        EXPECT_EQ(0, a2.move_count);
        EXPECT_EQ(1, a3.move_count);
        EXPECT_EQ(0, a3.copy_count);
        EXPECT_EQ(1, a2.copy_count);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_onlyMovesIfNecessary) {
        CopyCounting source;
        CopyCounting source2;
        CopyCounting&& a1 = extract_arg_by_filtered_index<is_my_copy_counting_class , 0>(3, std::move(source), "test", std::move(source2));
        CopyCounting a2 = extract_arg_by_filtered_index<is_my_copy_counting_class , 1>(3, std::move(source), "test", std::move(source2));
        EXPECT_EQ(0, a1.move_count);
        EXPECT_EQ(0, a1.copy_count);
        EXPECT_EQ(1, a2.move_count);
        EXPECT_EQ(0, a2.copy_count);
    }

    template<class T> using is_true = std::true_type;

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_keepsLValueReferencesIntact) {
        MyClass obj;
        MyClass& a1 = extract_arg_by_filtered_index<is_true, 1>(3, obj, "test", obj);
        EXPECT_EQ(&obj, &a1);
    }
}

namespace test_filter_map {
    class MyClass {};

    struct map_to_double {
      template<class T> constexpr double operator()(T a) const {
        return static_cast<double>(a);
      }
    };

    TEST(MetaprogrammingTest, FilterMap) {
        auto result = filter_map<double, std::is_integral>(map_to_double(), 3, "bla", MyClass(), 4, nullptr, 5);
        static_assert(std::is_same<array<double, 3>, decltype(result)>::value, "");
        constexpr array<double, 3> expected{{3.0, 4.0, 5.0}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_emptyInput) {
        auto result = filter_map<double, std::is_integral>(map_to_double());
        static_assert(std::is_same<array<double, 0>, decltype(result)>::value, "");
        constexpr array<double, 0> expected{{}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_emptyOutput) {
        auto result = filter_map<double, std::is_integral>(map_to_double(), "bla", MyClass(), nullptr);
        static_assert(std::is_same<array<double, 0>, decltype(result)>::value, "");
        constexpr array<double, 0> expected{{}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_movableOnly_byRValue) {
        struct map_movable_by_rvalue {
          MovableOnly operator()(MovableOnly&& a) const {
            return std::move(a);
          }
        };

        auto result = filter_map<MovableOnly, is_my_movable_only_class>(map_movable_by_rvalue(), MovableOnly(5), "bla", nullptr, 3, MovableOnly(2));
        static_assert(std::is_same<array<MovableOnly, 2>, decltype(result)>::value, "");
        constexpr array<MovableOnly, 2> expected {{MovableOnly(5), MovableOnly(2)}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_movableOnly_byValue) {
        struct map_movable_by_lvalue {
          MovableOnly operator()(MovableOnly a) const {
            return a;
          }
        };

        auto result = filter_map<MovableOnly, is_my_movable_only_class>(map_movable_by_lvalue(), MovableOnly(5), "bla", nullptr, 3, MovableOnly(2));
        static_assert(std::is_same<array<MovableOnly, 2>, decltype(result)>::value, "");
        constexpr array<MovableOnly, 2> expected {{MovableOnly(5), MovableOnly(2)}};
        EXPECT_EQ(expected, result);
    }

    // See https://github.com/pytorch/pytorch/issues/35546
    TEST(MetaprogrammingTest, DISABLED_ON_WINDOWS(FilterMap_onlyCopiesIfNecessary)) {
        struct map_copy_counting_by_copy {
          CopyCounting operator()(CopyCounting v) const {
            return v;
          }
        };

        CopyCounting source;
        CopyCounting source2;
        auto result = filter_map<CopyCounting, is_my_copy_counting_class>(map_copy_counting_by_copy(), CopyCounting(), "bla", nullptr, 3, source, std::move(source2));
        static_assert(std::is_same<array<CopyCounting, 3>, decltype(result)>::value, "");
        EXPECT_EQ(0, result[0].copy_count);
        EXPECT_EQ(2, result[0].move_count);
        EXPECT_EQ(1, result[1].copy_count);
        EXPECT_EQ(1, result[1].move_count);
        EXPECT_EQ(0, result[2].copy_count);
        EXPECT_EQ(2, result[2].move_count);
    }

    TEST(MetaprogrammingTest, DISABLED_ON_WINDOWS(FilterMap_onlyMovesIfNecessary_1)) {
        struct map_copy_counting_by_move {
          CopyCounting operator()(CopyCounting&& v) const {
            return std::move(v);
          }
        };

        CopyCounting source;
        auto result = filter_map<CopyCounting, is_my_copy_counting_class>(map_copy_counting_by_move(), CopyCounting(), "bla", nullptr, 3, std::move(source));
        static_assert(std::is_same<array<CopyCounting, 2>, decltype(result)>::value, "");
        EXPECT_EQ(0, result[0].copy_count);
        EXPECT_EQ(1, result[0].move_count);
        EXPECT_EQ(0, result[1].copy_count);
        EXPECT_EQ(1, result[1].move_count);
    }

    TEST(MetaprogrammingTest, FilterMap_onlyMovesIfNecessary_2) {
        struct map_copy_counting_by_pointer {
          const CopyCounting* operator()(const CopyCounting& v) const {
            return &v;
          }
        };

        CopyCounting source1;
        CopyCounting source2;
        auto result = filter_map<const CopyCounting*, is_my_copy_counting_class>(map_copy_counting_by_pointer(), "bla", nullptr, 3, source1, std::move(source2));
        static_assert(std::is_same<array<const CopyCounting*, 2>, decltype(result)>::value, "");
        EXPECT_EQ(0, result[0]->copy_count);
        EXPECT_EQ(0, result[0]->move_count);
        EXPECT_EQ(0, result[1]->copy_count);
        EXPECT_EQ(0, result[1]->move_count);
    }
}

namespace test_tuple_elements {
  TEST(MetaprogrammingTest, TupleElements_emptyInput) {
    auto x = std::make_tuple();
    auto y = tuple_elements(x, std::index_sequence<>());
    EXPECT_EQ(x, y);
  }

  TEST(MetaprogrammingTest, TupleElements_emptySelection) {
    auto x = std::make_tuple(0, "HEY", 2.0);
    auto y = tuple_elements(x, std::index_sequence<>());
    auto z = std::make_tuple();
    EXPECT_EQ(y, z);
  }

  TEST(MetaprogrammingTest, TupleElements_subsetSelection) {
    auto x = std::make_tuple(0, "HEY", 2.0);
    auto y = tuple_elements(x, std::index_sequence<0, 2>());
    auto z = std::make_tuple(0, 2.0);
    EXPECT_EQ(y, z);
  }

  TEST(MetaprogrammingTest, TupleElements_reorderSelection) {
    auto x = std::make_tuple(0, "HEY", 2.0);
    auto y = tuple_elements(x, std::index_sequence<0, 2, 1>());
    auto z = std::make_tuple(0, 2.0, "HEY");
    EXPECT_EQ(y, z);
  }
}

namespace test_tuple_take {
  TEST(MetaprogrammingTest, TupleTake_emptyInput) {
    auto x = std::make_tuple();
    auto y = tuple_take<std::tuple<>, 0>(x);
    EXPECT_EQ(x, y);
  }

  TEST(MetaprogrammingTest, TupleTake_emptyPrefix) {
    auto x = std::make_tuple(0, "HEY", 2.0);
    auto y = tuple_take<std::tuple<int, const char*, double>, 0>(x);
    auto z = std::make_tuple();
    EXPECT_EQ(y, z);
  }

  TEST(MetaprogrammingTest, TupleTake_nonemptyPrefix) {
    auto x = std::make_tuple(0, "HEY", 2.0);
    auto y = tuple_take<std::tuple<int, const char*, double>, 2>(x);
    auto z = std::make_tuple(0, "HEY");
    EXPECT_EQ(y, z);
  }

  TEST(MetaprogrammingTest, TupleTake_fullPrefix) {
    auto x = std::make_tuple(0, "HEY", 2.0);
    auto y = tuple_take<std::tuple<int, const char*, double>, 3>(x);
    EXPECT_EQ(x, y);
  }
}

}
