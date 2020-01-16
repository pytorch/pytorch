#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeIndex.h>
#include <gtest/gtest.h>

using c10::string_view;
using c10::util::get_fully_qualified_type_name;
using c10::util::get_type_index;

namespace {

namespace test_simple_types {
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(get_type_index<int>() == get_type_index<int>(), "");
static_assert(get_type_index<float>() == get_type_index<float>(), "");
static_assert(get_type_index<int>() != get_type_index<float>(), "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int(double, double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() != get_type_index<int(double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int (*)(double, double)>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() ==
        get_type_index<std::function<int(double, double)>>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() !=
        get_type_index<std::function<int(double)>>(),
    "");
#endif
TEST(TypeIndex, SimpleTypes) {
    EXPECT_EQ(get_type_index<int>(), get_type_index<int>());
    EXPECT_EQ(get_type_index<float>(), get_type_index<float>());
    EXPECT_NE(get_type_index<int>(), get_type_index<float>());
    EXPECT_EQ(
        get_type_index<int(double, double)>(),
        get_type_index<int(double, double)>());
    EXPECT_NE(
        get_type_index<int(double, double)>(),
        get_type_index<int(double)>());
    EXPECT_EQ(
        get_type_index<int(double, double)>(),
        get_type_index<int (*)(double, double)>());
    EXPECT_EQ(
        get_type_index<std::function<int(double, double)>>(),
        get_type_index<std::function<int(double, double)>>());
    EXPECT_NE(
        get_type_index<std::function<int(double, double)>>(),
        get_type_index<std::function<int(double)>>());
}
}

namespace test_references_and_pointers {
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(get_type_index<int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() == get_type_index<int&&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int>(), "");
static_assert(get_type_index<const int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() != get_type_index<int*>(), "");
static_assert(get_type_index<int*>() != get_type_index<int**>(), "");
static_assert(
    get_type_index<int(double&, double)>() !=
        get_type_index<int(double, double)>(),
    "");
#endif
TEST(TypeIndex, ReferencesAndPointers) {
    EXPECT_EQ(get_type_index<int>(), get_type_index<int&>());
    EXPECT_EQ(get_type_index<int>(), get_type_index<int&&>());
    EXPECT_EQ(get_type_index<int>(), get_type_index<const int&>());
    EXPECT_EQ(get_type_index<int>(), get_type_index<const int>());
    EXPECT_EQ(get_type_index<const int>(), get_type_index<int&>());
    EXPECT_NE(get_type_index<int>(), get_type_index<int*>());
    EXPECT_NE(get_type_index<int*>(), get_type_index<int**>());
    EXPECT_NE(
        get_type_index<int(double&, double)>(),
        get_type_index<int(double, double)>());
}
}

namespace test_function_traits {
struct Dummy final {};
struct Functor final {
  int64_t operator()(uint32_t, Dummy&&, const Dummy&) const;
};
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    get_type_index<int64_t(uint32_t, Dummy&&, const Dummy&)>() ==
        get_type_index<
            c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");
#endif
TEST(TypeIndex, FunctionTraits) {
    EXPECT_EQ(
        get_type_index<int64_t(uint32_t, Dummy&&, const Dummy&)>(),
        get_type_index<
            c10::guts::infer_function_traits_t<Functor>::func_type>());
}
}

namespace test_top_level_name {
class Dummy {};
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos != get_fully_qualified_type_name<Dummy>().find("Dummy"),
    "");
#endif
TEST(TypeIndex, TopLevelName) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Dummy>().find("Dummy")
    );
}
}

namespace test_nested_name {
struct Dummy final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy"),
    "");
#endif
TEST(TypeIndex, NestedName) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy")
    );
}
} // namespace test_nested_name

namespace test_type_template_parameter {
template <class T>
struct Outer final {};
struct Inner final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Outer"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Inner"),
    "");
#endif
TEST(TypeIndex, TypeTemplateParameter) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Outer")
    );
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Inner")
    );
}
} // namespace test_type_template_parameter

namespace test_nontype_template_parameter {
template <size_t N>
struct Class final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Class<38474355>>().find("38474355"),
    "");
#endif
TEST(TypeIndex, NonTypeTemplateParameter) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Class<38474355>>().find("38474355")
    );
}
} // namespace test_nontype_template_parameter

namespace test_type_computations_are_resolved {
template <class T>
struct Type final {
  using type = const T*;
};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find("int"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find("*"),
    "");

// but with remove_pointer applied, there is no '*' in the type name anymore
static_assert(
    string_view::npos ==
        get_fully_qualified_type_name<
            typename std::remove_pointer<typename Type<int>::type>::type>()
            .find("*"),
    "");
#endif
TEST(TypeIndex, TypeComputationsAreResolved) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<typename Type<int>::type>().find("int")
    );
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<typename Type<int>::type>().find("*")
    );
    // but with remove_pointer applied, there is no '*' in the type name anymore
    EXPECT_EQ(
        string_view::npos,
        get_fully_qualified_type_name<
            typename std::remove_pointer<typename Type<int>::type>::type>()
            .find("*")
    );
}

struct Functor final {
  std::string operator()(int64_t a, const Type<int>& b) const;
};
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>() ==
        get_fully_qualified_type_name<
            typename c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");
#endif
TEST(TypeIndex, FunctionTypeComputationsAreResolved) {
    EXPECT_EQ(
        get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>(),
        get_fully_qualified_type_name<
            typename c10::guts::infer_function_traits_t<Functor>::func_type>()
    );
}
} // namespace test_type_computations_are_resolved

namespace test_function_arguments_and_returns {
class Dummy final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Dummy(int)>().find(
            "test_function_arguments_and_returns::Dummy"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<void(Dummy)>().find(
            "test_function_arguments_and_returns::Dummy"),
    "");
#endif
TEST(TypeIndex, FunctionArgumentsAndReturns) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Dummy(int)>().find(
            "test_function_arguments_and_returns::Dummy")
    );
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<void(Dummy)>().find(
            "test_function_arguments_and_returns::Dummy")
    );
}
} // namespace test_function_arguments_and_returns
} // namespace

namespace test_type_names_are_well_defined {
// Our type_index computation works by taking the fully qualified type name and hashing it.
// Type names are compiler specific and different compilers can have different ways of printing
// a name for the same type. To make sure they get the same type_index, we need to make sure
// they get the same type name. This test case here ensures that the type name is what we expect.
// If one of these tests fails, we have to change the type name computation for that compiler
// so that it aligns with what the other compilers output for this type.
struct Dummy final {};
struct Functor final {
  int64_t operator()(uint32_t, Dummy&&, const Dummy&) const;
};
template <class T>
struct Outer final {};
struct Inner final {};
template <size_t N>
struct Class final {};
template <class T>
struct Type final {
  using type = const T*;
};
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert("int" == get_fully_qualified_type_name<int>(), "");
static_assert("float" == get_fully_qualified_type_name<float>(), "");
static_assert("double" == get_fully_qualified_type_name<double>(), "");
static_assert("int (double, double)" == get_fully_qualified_type_name<int(double, double)>(), "");
static_assert("int (*)(double, double)" == get_fully_qualified_type_name<int (*)(double, double)>(), "");
static_assert("std::function<int (double, double)>" == get_fully_qualified_type_name<std::function<int(double, double)>>(), "");
static_assert("int &" == get_fully_qualified_type_name<int&>(), "");
static_assert("int &&" == get_fully_qualified_type_name<int&&>(), "");
static_assert("const int &" == get_fully_qualified_type_name<const int&>(), "");
static_assert("const int" == get_fully_qualified_type_name<const int>(), "");
static_assert("int *" == get_fully_qualified_type_name<int*>(), "");
static_assert("int **" == get_fully_qualified_type_name<int**>(), "");
static_assert("int (double &, double)" ==
        get_fully_qualified_type_name<int(double&, double)>(),
    "");
static_assert(
    "long (unsigned int, test_type_names_are_well_defined::Dummy &&, const test_type_names_are_well_defined::Dummy &)" ==
        get_fully_qualified_type_name<c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");
static_assert(
    "test_type_names_are_well_defined::Outer<test_type_names_are_well_defined::Inner>" ==
        get_fully_qualified_type_name<Outer<Inner>>(),
    "");
static_assert(
    "test_type_names_are_well_defined::Class<38474355>" ==
        get_fully_qualified_type_name<Class<38474355>>(),
    "");
static_assert(
    "const int *" ==
        get_fully_qualified_type_name<typename Type<int>::type>(),
    "");
static_assert(
    "const int" ==
        get_fully_qualified_type_name<
            typename std::remove_pointer<typename Type<int>::type>::type>(),
    "");
#endif
TEST(TypeIndex, TypeNamesAreWellDefined) {
    EXPECT_EQ("int", get_fully_qualified_type_name<int>());
    EXPECT_EQ("float", get_fully_qualified_type_name<float>());
    EXPECT_EQ("double", get_fully_qualified_type_name<double>());
    EXPECT_EQ("int (double, double)", get_fully_qualified_type_name<int(double, double)>());
    EXPECT_EQ("int (*)(double, double)", get_fully_qualified_type_name<int (*)(double, double)>());
    EXPECT_EQ("std::function<int (double, double)>", get_fully_qualified_type_name<std::function<int(double, double)>>());
    EXPECT_EQ("int &", get_fully_qualified_type_name<int&>());
    EXPECT_EQ("int &&", get_fully_qualified_type_name<int&&>());
    EXPECT_EQ("const int &", get_fully_qualified_type_name<const int&>());
    EXPECT_EQ("const int", get_fully_qualified_type_name<const int>());
    EXPECT_EQ("int *", get_fully_qualified_type_name<int*>());
    EXPECT_EQ("int **", get_fully_qualified_type_name<int**>());
    EXPECT_EQ("int (double &, double)",
            get_fully_qualified_type_name<int(double&, double)>());
    EXPECT_EQ("long (unsigned int, test_type_names_are_well_defined::Dummy &&, const test_type_names_are_well_defined::Dummy &)",
        get_fully_qualified_type_name<
            c10::guts::infer_function_traits_t<Functor>::func_type>());
    EXPECT_EQ(
        "test_type_names_are_well_defined::Outer<test_type_names_are_well_defined::Inner>",
            get_fully_qualified_type_name<Outer<Inner>>());
    EXPECT_EQ(
        "test_type_names_are_well_defined::Class<38474355>",
            get_fully_qualified_type_name<Class<38474355>>());
    EXPECT_EQ(
        "const int *",
            get_fully_qualified_type_name<typename Type<int>::type>());
    EXPECT_EQ(
        "const int",
            get_fully_qualified_type_name<
                typename std::remove_pointer<typename Type<int>::type>::type>());
}
}
