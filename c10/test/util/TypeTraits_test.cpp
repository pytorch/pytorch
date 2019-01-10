#include <c10/util/TypeTraits.h>
#include <gtest/gtest.h>

using namespace c10::guts;

namespace {

namespace test_is_equality_comparable {
    class NotEqualityComparable {};
    class EqualityComparable {};

    inline bool operator==(const EqualityComparable &, const EqualityComparable &) { return false; }

    static_assert(!is_equality_comparable<NotEqualityComparable>::value, "");
    static_assert(is_equality_comparable<EqualityComparable>::value, "");
    static_assert(is_equality_comparable<int>::value, "");

    // v_ just exists to silence a compiler warning about operator==(EqualityComparable, EqualityComparable) not being needed
    const bool v_ = EqualityComparable() == EqualityComparable();
}

namespace test_is_hashable {
    class NotHashable {};
    class Hashable {};
}
}
namespace std {
    template<> struct hash<test_is_hashable::Hashable> final {
        size_t operator()(const test_is_hashable::Hashable &) { return 0; }
    };
}
namespace {
namespace test_is_hashable {
    static_assert(is_hashable<int>::value, "");
    static_assert(is_hashable<Hashable>::value, "");
    static_assert(!is_hashable<NotHashable>::value, "");
}

namespace test_is_function_type {
    class MyClass {};
    struct Functor {
        void operator()() {}
    };
    auto lambda = [] () {};
    // func() and func__ just exists to silence a compiler warning about lambda being unused
    bool func() {
        lambda();
        return true;
    }
    bool func__ = func();

    static_assert(is_function_type<void()>::value, "");
    static_assert(is_function_type<int()>::value, "");
    static_assert(is_function_type<MyClass()>::value, "");
    static_assert(is_function_type<void(MyClass)>::value, "");
    static_assert(is_function_type<void(int)>::value, "");
    static_assert(is_function_type<void(void*)>::value, "");
    static_assert(is_function_type<int()>::value, "");
    static_assert(is_function_type<int(MyClass)>::value, "");
    static_assert(is_function_type<int(const MyClass&)>::value, "");
    static_assert(is_function_type<int(MyClass&&)>::value, "");
    static_assert(is_function_type<MyClass&&()>::value, "");
    static_assert(is_function_type<MyClass&&(MyClass&&)>::value, "");
    static_assert(is_function_type<const MyClass&(int, float, MyClass)>::value, "");

    static_assert(!is_function_type<void>::value, "");
    static_assert(!is_function_type<int>::value, "");
    static_assert(!is_function_type<MyClass>::value, "");
    static_assert(!is_function_type<void*>::value, "");
    static_assert(!is_function_type<const MyClass&>::value, "");
    static_assert(!is_function_type<MyClass&&>::value, "");

    static_assert(!is_function_type<void (*)()>::value, "function pointers aren't plain functions");
    static_assert(!is_function_type<Functor>::value, "Functors aren't plain functions");
    static_assert(!is_function_type<decltype(lambda)>::value, "Lambdas aren't plain functions");
}

namespace test_is_instantiation_of {
    class MyClass {};
    template<class T> class Single {};
    template<class T1, class T2> class Double {};
    template<class... T> class Multiple {};

    static_assert(is_instantiation_of<Single, Single<void>>::value, "");
    static_assert(is_instantiation_of<Single, Single<MyClass>>::value, "");
    static_assert(is_instantiation_of<Single, Single<int>>::value, "");
    static_assert(is_instantiation_of<Single, Single<void*>>::value, "");
    static_assert(is_instantiation_of<Single, Single<int*>>::value, "");
    static_assert(is_instantiation_of<Single, Single<const MyClass&>>::value, "");
    static_assert(is_instantiation_of<Single, Single<MyClass&&>>::value, "");
    static_assert(is_instantiation_of<Double, Double<int, void>>::value, "");
    static_assert(is_instantiation_of<Double, Double<const int&, MyClass*>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<int>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<MyClass&, int>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<MyClass&, int, MyClass>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<MyClass&, int, MyClass, void*>>::value, "");

    static_assert(!is_instantiation_of<Single, Double<int, int>>::value, "");
    static_assert(!is_instantiation_of<Single, Double<int, void>>::value, "");
    static_assert(!is_instantiation_of<Single, Multiple<int>>::value, "");
    static_assert(!is_instantiation_of<Double, Single<int>>::value, "");
    static_assert(!is_instantiation_of<Double, Multiple<int, int>>::value, "");
    static_assert(!is_instantiation_of<Double, Multiple<>>::value, "");
    static_assert(!is_instantiation_of<Multiple, Double<int, int>>::value, "");
    static_assert(!is_instantiation_of<Multiple, Single<int>>::value, "");
}

namespace test_is_type_condition {
    template<class> class NotATypeCondition {};
    static_assert(is_type_condition<std::is_reference>::value, "");
    static_assert(!is_type_condition<NotATypeCondition>::value, "");
}
}
