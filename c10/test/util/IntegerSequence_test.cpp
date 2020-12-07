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

namespace test_take {
    static_assert(std::is_same<
        index_sequence<>,
        take_t<index_sequence<>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>, // if we go to zero elements, then IntType cannot be preserved and it goes to index_sequence
        take_t<integer_sequence<int8_t>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>,
        take_t<index_sequence<1>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>, // if we go to zero elements, then IntType cannot be preserved and it goes to index_sequence
        take_t<integer_sequence<int8_t, 1>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>,
        take_t<index_sequence<1, 5>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<>, // if we go to zero elements, then IntType cannot be preserved and it goes to index_sequence
        take_t<integer_sequence<int8_t, 1, 5>, 0>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<1, 5>,
        take_t<index_sequence<1, 5, 3, 4>, 2>
        >::value, "");
    static_assert(std::is_same<
        integer_sequence<int8_t, 1, 5>,
        take_t<integer_sequence<int8_t, 1, 5, 3, 4>, 2>
        >::value, "");
    static_assert(std::is_same<
        index_sequence<1, 5>,
        take_t<index_sequence<1, 5>, 2>
        >::value, "");
    static_assert(std::is_same<
        integer_sequence<int8_t, 1, 5>,
        take_t<integer_sequence<int8_t, 1, 5>, 2>
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

namespace range_test {
    // empty
    static_assert(std::is_same<
        index_sequence<>,
        range_t<0, 0>
    >::value, "");
    static_assert(std::is_same<
        index_sequence<>,
        range_t<5, 5>
    >::value, "");

    // non-empty
    static_assert(std::is_same<
        index_sequence<5>,
        range_t<5, 6>
    >::value, "");
    static_assert(std::is_same<
        index_sequence<5, 6, 7, 8, 9>,
        range_t<5, 10>
    >::value, "");
}

namespace test_zip_with_index {
    using typelist::typelist;
    static_assert(std::is_same<typelist<>, zip_with_index_t<index_sequence<>>>::value, "");
    static_assert(std::is_same<typelist<index_sequence<0, 3>>, zip_with_index_t<index_sequence<3>>>::value, "");
    static_assert(std::is_same<typelist<index_sequence<0, 3>, index_sequence<1, 95>>, zip_with_index_t<index_sequence<3, 95>>>::value, "");
    static_assert(std::is_same<typelist<index_sequence<0, 3>, index_sequence<1, 95>, index_sequence<2, 20>>, zip_with_index_t<index_sequence<3, 95, 20>>>::value, "");

    static_assert(std::is_same<typelist<integer_sequence<int8_t, 0, 3>, integer_sequence<int8_t, 1, -95>, integer_sequence<int8_t, 2, 20>>, zip_with_index_t<integer_sequence<int8_t, 3, -95, 20>>>::value, "");
}

namespace test_insert {
    static_assert(std::is_same<index_sequence<100>, insert_t<index_sequence<>, 0, 100>>::value, "");
    static_assert(std::is_same<index_sequence<100, 2>, insert_t<index_sequence<2>, 0, 100>>::value, "");
    static_assert(std::is_same<index_sequence<2, 100>, insert_t<index_sequence<2>, 1, 100>>::value, "");
    static_assert(std::is_same<index_sequence<100, 2, 5>, insert_t<index_sequence<2, 5>, 0, 100>>::value, "");
    static_assert(std::is_same<index_sequence<2, 100, 5>, insert_t<index_sequence<2, 5>, 1, 100>>::value, "");
    static_assert(std::is_same<index_sequence<2, 5, 100>, insert_t<index_sequence<2, 5>, 2, 100>>::value, "");

    static_assert(std::is_same<integer_sequence<int8_t, 2, 100, 5>, insert_t<integer_sequence<int8_t, 2, 5>, 1, 100>>::value, "");
}

namespace test_insert_all {
    using typelist::typelist;
    static_assert(std::is_same<index_sequence<>, insert_all_t<index_sequence<>, typelist<>>>::value, "");
    static_assert(std::is_same<index_sequence<3>, insert_all_t<index_sequence<3>, typelist<>>>::value, "");
    static_assert(std::is_same<index_sequence<3>, insert_all_t<index_sequence<>, typelist<index_sequence<0, 3>>>>::value, "");
    static_assert(std::is_same<index_sequence<4, 3, 5>, insert_all_t<index_sequence<>,
                        typelist<index_sequence<0, 3>, index_sequence<0, 4>, index_sequence<2, 5>>>>::value, "");
    static_assert(std::is_same<index_sequence<4, 3, 5>, insert_all_t<index_sequence<3>,
                        typelist<index_sequence<0, 4>, index_sequence<2, 5>>>>::value, "");

    static_assert(std::is_same<integer_sequence<int8_t, 4, 3, -5>, insert_all_t<integer_sequence<int8_t, 3>,
                        typelist<integer_sequence<int8_t, 0, 4>, integer_sequence<int8_t, 2, -5>>>>::value, "");
}

namespace test_remove_by_index {
    static_assert(std::is_same<index_sequence<>, remove_by_index_t<index_sequence<100>, 0>>::value, "");
    static_assert(std::is_same<index_sequence<2>, remove_by_index_t<index_sequence<100, 2>, 0>>::value, "");
    static_assert(std::is_same<index_sequence<2>, remove_by_index_t<index_sequence<2, 100>, 1>>::value, "");
    static_assert(std::is_same<index_sequence<2, 5>, remove_by_index_t<index_sequence<100, 2, 5>, 0>>::value, "");
    static_assert(std::is_same<index_sequence<2, 5>, remove_by_index_t<index_sequence<2, 100, 5>, 1>>::value, "");
    static_assert(std::is_same<index_sequence<2, 5>, remove_by_index_t<index_sequence<2, 5, 100>, 2>>::value, "");

    static_assert(std::is_same<integer_sequence<int8_t, 2, 5>, remove_by_index_t<integer_sequence<int8_t, 2, 100, 5>, 1>>::value, "");
}

namespace test_remove_all_by_index {
    static_assert(std::is_same<index_sequence<>, remove_all_by_index_t<index_sequence<>, index_sequence<>>>::value, "");
    static_assert(std::is_same<index_sequence<3>, remove_all_by_index_t<index_sequence<3>, index_sequence<>>>::value, "");
    static_assert(std::is_same<index_sequence<>, remove_all_by_index_t<index_sequence<3>, index_sequence<0>>>::value, "");
    static_assert(std::is_same<index_sequence<>, remove_all_by_index_t<index_sequence<4, 3, 5>, index_sequence<2, 0, 0>>>::value, "");
    static_assert(std::is_same<index_sequence<3>, remove_all_by_index_t<index_sequence<4, 3, 5>, index_sequence<2, 0>>>::value, "");

    static_assert(std::is_same<integer_sequence<int8_t, -5>, remove_all_by_index_t<integer_sequence<int8_t, 4, 3, -5>, index_sequence<0, 0>>>::value, "");
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

namespace test_reverse {
    static_assert(std::is_same<
            std::index_sequence<5, 2, 10, 94>,
            reverse_t<std::index_sequence<94, 10, 2, 5>>
    >::value, "");
    static_assert(std::is_same<
            std::index_sequence<>,
            reverse_t<std::index_sequence<>>
    >::value, "");

    static_assert(std::is_same<
            integer_sequence<int8_t, 5, -2, 10, 94>,
            reverse_t<integer_sequence<int8_t, 94, 10, -2, 5>>
    >::value, "");
}

namespace test_set {
    static_assert(std::is_same<
        set_t<index_sequence<0>, 0, 100>,
        index_sequence<100>
    >::value, "");
    static_assert(std::is_same<
        set_t<index_sequence<0, 5, 2>, 0, 100>,
        index_sequence<100, 5, 2>
    >::value, "");
        static_assert(std::is_same<
        set_t<index_sequence<0, 5, 2>, 1, 100>,
        index_sequence<0, 100, 2>
    >::value, "");
    static_assert(std::is_same<
        set_t<index_sequence<0, 5, 2>, 2, 100>,
        index_sequence<0, 5, 100>
    >::value, "");

    static_assert(std::is_same<
        set_t<integer_sequence<int8_t, 0, 5, 2>, 0, 100>,
        integer_sequence<int8_t, 100, 5, 2>
    >::value, "");
}

namespace test_set_all {
    static_assert(std::is_same<
        set_all_t<index_sequence<>, typelist::typelist<>>,
        index_sequence<>
    >::value, "");
    static_assert(std::is_same<
        set_all_t<index_sequence<5>, typelist::typelist<std::index_sequence<0, 100>>>,
        index_sequence<100>
    >::value, "");
    static_assert(std::is_same<
        set_all_t<index_sequence<5, 7, 8>, typelist::typelist<std::index_sequence<1, 700>, std::index_sequence<0, 500>, std::index_sequence<2, 800>>>,
        index_sequence<500, 700, 800>
    >::value, "");

    static_assert(std::is_same<
        set_all_t<integer_sequence<int8_t, 5>, typelist::typelist<std::integer_sequence<int8_t, 0, -100>>>,
        integer_sequence<int8_t, -100>
    >::value, "");
}

namespace test_permutation {
    namespace test_apply_to_typelist_t {
        static_assert(std::is_same<
            typelist::typelist<>,
            permutation<index_sequence<>>::apply_to_typelist_t<typelist::typelist<>>
        >::value, "");
        static_assert(std::is_same<
            typelist::typelist<int>,
            permutation<index_sequence<0>>::apply_to_typelist_t<typelist::typelist<int>>
        >::value, "");
        static_assert(std::is_same<
            typelist::typelist<int, char>,
            permutation<index_sequence<1, 0>>::apply_to_typelist_t<typelist::typelist<char, int>>
        >::value, "");
        static_assert(std::is_same<
            typelist::typelist<int, bool, char>,
            permutation<index_sequence<1, 2, 0>>::apply_to_typelist_t<typelist::typelist<char, int, bool>>
        >::value, "");
    }
    namespace test_apply_to_tuple {
        static_assert(std::make_tuple() == permutation<index_sequence<>>::apply_to_tuple(std::make_tuple()), "");
        static_assert(std::make_tuple(1) == permutation<index_sequence<0>>::apply_to_tuple(std::make_tuple(1)), "");
        static_assert(std::make_tuple(5, 4) == permutation<index_sequence<1, 0>>::apply_to_tuple(std::make_tuple(4, 5)), "");
        static_assert(std::make_tuple(5, 2.3f, 4) == permutation<index_sequence<1, 2, 0>>::apply_to_tuple(std::make_tuple(4, 5, 2.3f)), "");
    }
    namespace test_andthen {
        static_assert(std::is_same<
            permutation<index_sequence<>>,
            typename permutation<index_sequence<>>::and_then_t<permutation<index_sequence<>>>
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<0>>,
            typename permutation<index_sequence<0>>::and_then_t<permutation<index_sequence<0>>>
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<1, 2, 0>>,
            typename permutation<index_sequence<1, 2, 0>>::and_then_t<permutation<index_sequence<0, 1, 2>>>
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<1, 2, 0>>,
            typename permutation<index_sequence<0, 1, 2>>::and_then_t<permutation<index_sequence<1, 2, 0>>>
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<0, 4, 2, 3, 1>>,
            typename permutation<index_sequence<3, 2, 4, 0, 1>>
                ::and_then_t<permutation<index_sequence<3, 1, 4, 0, 2>>>
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<0, 2, 1, 3, 4>>,
            typename permutation<index_sequence<3, 1, 4, 0, 2>>
                ::and_then_t<permutation<index_sequence<3, 2, 4, 0, 1>>>
        >::value, "");
    }

    namespace test_inverted {
        static_assert(std::is_same<
            permutation<index_sequence<>>,
            typename permutation<index_sequence<>>::inverted_t
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<0>>,
            typename permutation<index_sequence<0>>::inverted_t
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<0, 1, 2, 3, 4, 5>>,
            typename permutation<index_sequence<0, 1, 2, 3, 4, 5>>::inverted_t
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<5, 4, 3, 2, 1, 0>>,
            typename permutation<index_sequence<5, 4, 3, 2, 1, 0>>::inverted_t
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<1, 2, 0>>,
            typename permutation<index_sequence<2, 0, 1>>::inverted_t
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<2, 0, 1>>,
            typename permutation<index_sequence<1, 2, 0>>::inverted_t
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<0, 1, 2, 3, 4, 5>>,
            typename permutation<index_sequence<1, 4, 2, 3, 5, 0>>::inverted_t
                ::and_then_t<permutation<index_sequence<1, 4, 2, 3, 5, 0>>>
        >::value, "");
        static_assert(std::is_same<
            permutation<index_sequence<0, 1, 2, 3, 4, 5>>,
            typename permutation<index_sequence<1, 4, 2, 3, 5, 0>>
                ::and_then_t<permutation<index_sequence<1, 4, 2, 3, 5, 0>>::inverted_t>
        >::value, "");
    }
}
