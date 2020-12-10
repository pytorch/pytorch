#include <c10/util/IntegerSequence.h>

using namespace c10::guts;
using namespace c10::guts::iseq;

using std::index_sequence;
using std::integer_sequence;

namespace test_iseq_to_typelist {
    // empty iseq
    static_assert(std::is_same<typelist::typelist<>, iseq_to_typelist_t<index_sequence<>>>::value, "");
    static_assert(std::is_same<typelist::typelist<>, iseq_to_typelist_t<integer_sequence<int8_t>>>::value, "");

    // nonempty iseq
    static_assert(std::is_same<
        typelist::typelist<CompileTimeInteger<size_t, 2>>,
        iseq_to_typelist_t<index_sequence<2>>
        >::value, "");
    static_assert(std::is_same<
        typelist::typelist<CompileTimeInteger<size_t, 2>, CompileTimeInteger<size_t, 1>>,
        iseq_to_typelist_t<index_sequence<2, 1>>
        >::value, "");
    static_assert(std::is_same<
        typelist::typelist<CompileTimeInteger<int8_t, 2>>,
        iseq_to_typelist_t<integer_sequence<int8_t, 2>>
        >::value, "");
    static_assert(std::is_same<
        typelist::typelist<CompileTimeInteger<int8_t, 2>, CompileTimeInteger<int8_t, 1>>,
        iseq_to_typelist_t<integer_sequence<int8_t, 2, 1>>
        >::value, "");
}

namespace test_typelist_to_iseq {
    // empty typelist
    static_assert(std::is_same<index_sequence<>, typelist_to_iseq_t<typelist::typelist<>>>::value, "");

    // nonempty typelist
    static_assert(std::is_same<
        index_sequence<2>,
        typelist_to_iseq_t<typelist::typelist<CompileTimeInteger<size_t, 2>>>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<2, 1>,
        typelist_to_iseq_t<typelist::typelist<CompileTimeInteger<size_t, 2>, CompileTimeInteger<size_t, 1>>>
        >::value, "");
    static_assert(std::is_same<
        integer_sequence<int8_t, 2>,
        typelist_to_iseq_t<typelist::typelist<CompileTimeInteger<int8_t, 2>>>
        >::value, "");
    static_assert(std::is_same<
        integer_sequence<int8_t, 2, 1>,
        typelist_to_iseq_t<typelist::typelist<CompileTimeInteger<int8_t, 2>, CompileTimeInteger<int8_t, 1>>>
        >::value, "");
}

namespace test_drop {
    static_assert(std::is_same<
        index_sequence<>,
        drop_t<index_sequence<>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>, // if we go to zero elements, then IntType cannot be preserved and it goes to index_sequence
        drop_t<integer_sequence<int8_t>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>,
        drop_t<index_sequence<1>, 1>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>, // if we go to zero elements, then IntType cannot be preserved and it goes to index_sequence
        drop_t<integer_sequence<int8_t, 1>, 1>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>,
        drop_t<index_sequence<1, 5>, 2>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>, // if we go to zero elements, then IntType cannot be preserved and it goes to index_sequence
        drop_t<integer_sequence<int8_t, 1, 5>, 2>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<3, 4>,
        drop_t<index_sequence<1, 5, 3, 4>, 2>
        >::value, "");
    static_assert(std::is_same<
        integer_sequence<int8_t, 3, 4>,
        drop_t<integer_sequence<int8_t, 1, 5, 3, 4>, 2>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<1, 5>,
        drop_t<index_sequence<1, 5>, 0>
        >::value, "");
    static_assert(std::is_same<
        integer_sequence<int8_t, 1, 5>,
        drop_t<integer_sequence<int8_t, 1, 5>, 0>
        >::value, "");
}

namespace test_concat {
    static_assert(std::is_same<index_sequence<>, concat_t<>>::value, "");
    static_assert(std::is_same<index_sequence<>, concat_t<index_sequence<>>>::value, "");
    static_assert(std::is_same<index_sequence<>, concat_t<index_sequence<>, index_sequence<>>>::value, "");
    static_assert(std::is_same<index_sequence<4>, concat_t<index_sequence<4>>>::value, "");
    static_assert(std::is_same<index_sequence<4>, concat_t<index_sequence<4>, index_sequence<>>>::value, "");
    static_assert(std::is_same<index_sequence<4>, concat_t<index_sequence<>, index_sequence<4>>>::value, "");
    static_assert(std::is_same<index_sequence<4>, concat_t<index_sequence<>, index_sequence<4>, index_sequence<>>>::value, "");
    static_assert(std::is_same<index_sequence<4, 2>, concat_t<index_sequence<4>, index_sequence<2>>>::value, "");
    static_assert(std::is_same<index_sequence<4, 2>, concat_t<index_sequence<>, index_sequence<4, 2>, index_sequence<>>>::value, "");
    static_assert(std::is_same<index_sequence<4, 2, 9>, concat_t<index_sequence<>, index_sequence<4, 2>, index_sequence<9>>>::value, "");

    static_assert(std::is_same<integer_sequence<int8_t, -5, -3>, concat_t<integer_sequence<int8_t, -5>, integer_sequence<int8_t, -3>>>::value, "");
}
