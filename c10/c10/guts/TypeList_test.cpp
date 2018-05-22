#include "TypeList.h"
#include <gtest/gtest.h>

using namespace c10::guts::typelist;

namespace test_from_tuple {
    class MyClass {};
    static_assert(std::is_same<typelist<int, float&, const MyClass&&>, from_tuple_t<std::tuple<int, float&, const MyClass&&>>>::value, "");
    static_assert(std::is_same<typelist<>, from_tuple_t<std::tuple<>>>::value, "");
}

namespace test_concat {
    class MyClass {};
    static_assert(std::is_same<typelist<>, concat_t<>>::value, "");
    static_assert(std::is_same<typelist<>, concat_t<typelist<>>>::value, "");
    static_assert(std::is_same<typelist<>, concat_t<typelist<>, typelist<>>>::value, "");
    static_assert(std::is_same<typelist<int>, concat_t<typelist<int>>>::value, "");
    static_assert(std::is_same<typelist<int>, concat_t<typelist<int>, typelist<>>>::value, "");
    static_assert(std::is_same<typelist<int>, concat_t<typelist<>, typelist<int>>>::value, "");
    static_assert(std::is_same<typelist<int>, concat_t<typelist<>, typelist<int>, typelist<>>>::value, "");
    static_assert(std::is_same<typelist<int, float&>, concat_t<typelist<int>, typelist<float&>>>::value, "");
    static_assert(std::is_same<typelist<int, float&>, concat_t<typelist<>, typelist<int, float&>, typelist<>>>::value, "");
    static_assert(std::is_same<typelist<int, float&, const MyClass&&>, concat_t<typelist<>, typelist<int, float&>, typelist<const MyClass&&>>>::value, "");
}

namespace test_filter {
    class MyClass {};
    static_assert(std::is_same<typelist<>, filter_t<std::is_reference, typelist<>>>::value, "");
    static_assert(std::is_same<typelist<>, filter_t<std::is_reference, typelist<int, float, double, MyClass>>>::value, "");
    static_assert(std::is_same<typelist<float&, const MyClass&&>, filter_t<std::is_reference, typelist<int, float&, double, const MyClass&&>>>::value, "");
}

namespace test_count_if {
    class MyClass final {};
    static_assert(count_if<std::is_reference, typelist<int, bool&, const MyClass&&, float, double>>::value == 2, "");
    static_assert(count_if<std::is_reference, typelist<int, bool>>::value == 0, "");
    static_assert(count_if<std::is_reference, typelist<>>::value == 0, "");
}

namespace test_true_for_each_type {
    class MyClass {};
    static_assert(true_for_each_type<std::is_reference, typelist<int&, const float&&, const MyClass&>>::value, "");
    static_assert(!true_for_each_type<std::is_reference, typelist<int&, const float, const MyClass&>>::value, "");
    static_assert(true_for_each_type<std::is_reference, typelist<>>::value, "");
}

namespace test_map {
    class MyClass {};
    static_assert(std::is_same<typelist<>, map_t<std::add_lvalue_reference_t, typelist<>>>::value, "");
    static_assert(std::is_same<typelist<int&>, map_t<std::add_lvalue_reference_t, typelist<int>>>::value, "");
    static_assert(std::is_same<typelist<int&, double&, const MyClass&>, map_t<std::add_lvalue_reference_t, typelist<int, double, const MyClass>>>::value, "");
}

namespace test_head {
    class MyClass {};
    static_assert(std::is_same<int, head_t<typelist<int, double>>>::value, "");
    static_assert(std::is_same<const MyClass&, head_t<typelist<const MyClass&, double>>>::value, "");
    static_assert(std::is_same<MyClass&&, head_t<typelist<MyClass&&, MyClass>>>::value, "");
    static_assert(std::is_same<bool, head_t<typelist<bool>>>::value, "");
}

namespace test_reverse {
    class MyClass {};
    static_assert(std::is_same<
            typelist<int, double, MyClass*, const MyClass&&>,
            reverse_t<typelist<const MyClass&&, MyClass*, double, int>>
    >::value, "");
    static_assert(std::is_same<
            typelist<>,
            reverse_t<typelist<>>
    >::value, "");
}

namespace test_map_types_to_values {
    TEST(TypeListTest, MapTypesToValues_sametype) {
        auto sizes =
            map_types_to_values<typelist<int64_t, bool, uint32_t>>(
                    [](auto type) -> size_t { return sizeof(typename decltype(type)::type); }
            );
        std::tuple<size_t, size_t, size_t> expected(8, 1, 4);
        static_assert(std::is_same<decltype(expected), decltype(sizes)>::value, "");
        EXPECT_EQ(expected, sizes);
    }

    TEST(TypeListTest, MapTypesToValues_differenttypes) {
        auto shared_ptrs =
                map_types_to_values<typelist<int, double>>(
                        [](auto type) { return std::make_shared<typename decltype(type)::type>(); }
                );
        static_assert(std::is_same<std::tuple<std::shared_ptr<int>, std::shared_ptr<double>>, decltype(shared_ptrs)>::value, "");
    }

    struct Class1 {static int func() {return 3;}};
    struct Class2 {static double func() {return 2.0;}};

    TEST(TypeListTest, MapTypesToValues_members) {
        auto result =
                map_types_to_values<typelist<Class1, Class2>>(
                        [](auto type) { return decltype(type)::type::func(); }
                );
        std::tuple<int, double> expected(3, 2.0);
        static_assert(std::is_same<decltype(expected), decltype(result)>::value, "");
        EXPECT_EQ(expected, result);
    }

    TEST(TypeListTest, MapTypesToValues_empty) {
        auto result =
                map_types_to_values<typelist<>>(
                        [](auto type) { return decltype(type)::type::this_doesnt_exist(); }
                );
        std::tuple<> expected;
        static_assert(std::is_same<decltype(expected), decltype(result)>::value, "");
        EXPECT_EQ(expected, result);
    }

}
