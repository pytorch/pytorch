/*
    tests/test_builtin_casters.cpp -- Casters available without any additional headers

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/complex.h>

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

TEST_SUBMODULE(builtin_casters, m) {
    // test_simple_string
    m.def("string_roundtrip", [](const char *s) { return s; });

    // test_unicode_conversion
    // Some test characters in utf16 and utf32 encodings.  The last one (the 𝐀) contains a null byte
    char32_t a32 = 0x61 /*a*/, z32 = 0x7a /*z*/, ib32 = 0x203d /*‽*/, cake32 = 0x1f382 /*🎂*/,              mathbfA32 = 0x1d400 /*𝐀*/;
    char16_t b16 = 0x62 /*b*/, z16 = 0x7a,       ib16 = 0x203d,       cake16_1 = 0xd83c, cake16_2 = 0xdf82, mathbfA16_1 = 0xd835, mathbfA16_2 = 0xdc00;
    std::wstring wstr;
    wstr.push_back(0x61); // a
    wstr.push_back(0x2e18); // ⸘
    if (sizeof(wchar_t) == 2) { wstr.push_back(mathbfA16_1); wstr.push_back(mathbfA16_2); } // 𝐀, utf16
    else { wstr.push_back((wchar_t) mathbfA32); } // 𝐀, utf32
    wstr.push_back(0x7a); // z

    m.def("good_utf8_string", []() { return std::string(u8"Say utf8\u203d \U0001f382 \U0001d400"); }); // Say utf8‽ 🎂 𝐀
    m.def("good_utf16_string", [=]() { return std::u16string({ b16, ib16, cake16_1, cake16_2, mathbfA16_1, mathbfA16_2, z16 }); }); // b‽🎂𝐀z
    m.def("good_utf32_string", [=]() { return std::u32string({ a32, mathbfA32, cake32, ib32, z32 }); }); // a𝐀🎂‽z
    m.def("good_wchar_string", [=]() { return wstr; }); // a‽𝐀z
    m.def("bad_utf8_string", []()  { return std::string("abc\xd0" "def"); });
    m.def("bad_utf16_string", [=]() { return std::u16string({ b16, char16_t(0xd800), z16 }); });
    // Under Python 2.7, invalid unicode UTF-32 characters don't appear to trigger UnicodeDecodeError
    if (PY_MAJOR_VERSION >= 3)
        m.def("bad_utf32_string", [=]() { return std::u32string({ a32, char32_t(0xd800), z32 }); });
    if (PY_MAJOR_VERSION >= 3 || sizeof(wchar_t) == 2)
        m.def("bad_wchar_string", [=]() { return std::wstring({ wchar_t(0x61), wchar_t(0xd800) }); });
    m.def("u8_Z", []() -> char { return 'Z'; });
    m.def("u8_eacute", []() -> char { return '\xe9'; });
    m.def("u16_ibang", [=]() -> char16_t { return ib16; });
    m.def("u32_mathbfA", [=]() -> char32_t { return mathbfA32; });
    m.def("wchar_heart", []() -> wchar_t { return 0x2665; });

    // test_single_char_arguments
    m.attr("wchar_size") = py::cast(sizeof(wchar_t));
    m.def("ord_char", [](char c) -> int { return static_cast<unsigned char>(c); });
    m.def("ord_char_lv", [](char &c) -> int { return static_cast<unsigned char>(c); });
    m.def("ord_char16", [](char16_t c) -> uint16_t { return c; });
    m.def("ord_char16_lv", [](char16_t &c) -> uint16_t { return c; });
    m.def("ord_char32", [](char32_t c) -> uint32_t { return c; });
    m.def("ord_wchar", [](wchar_t c) -> int { return c; });

    // test_bytes_to_string
    m.def("strlen", [](char *s) { return strlen(s); });
    m.def("string_length", [](std::string s) { return s.length(); });

    // test_string_view
#ifdef PYBIND11_HAS_STRING_VIEW
    m.attr("has_string_view") = true;
    m.def("string_view_print",   [](std::string_view s)    { py::print(s, s.size()); });
    m.def("string_view16_print", [](std::u16string_view s) { py::print(s, s.size()); });
    m.def("string_view32_print", [](std::u32string_view s) { py::print(s, s.size()); });
    m.def("string_view_chars",   [](std::string_view s)    { py::list l; for (auto c : s) l.append((std::uint8_t) c); return l; });
    m.def("string_view16_chars", [](std::u16string_view s) { py::list l; for (auto c : s) l.append((int) c); return l; });
    m.def("string_view32_chars", [](std::u32string_view s) { py::list l; for (auto c : s) l.append((int) c); return l; });
    m.def("string_view_return",   []() { return std::string_view(u8"utf8 secret \U0001f382"); });
    m.def("string_view16_return", []() { return std::u16string_view(u"utf16 secret \U0001f382"); });
    m.def("string_view32_return", []() { return std::u32string_view(U"utf32 secret \U0001f382"); });
#endif

    // test_integer_casting
    m.def("i32_str", [](std::int32_t v) { return std::to_string(v); });
    m.def("u32_str", [](std::uint32_t v) { return std::to_string(v); });
    m.def("i64_str", [](std::int64_t v) { return std::to_string(v); });
    m.def("u64_str", [](std::uint64_t v) { return std::to_string(v); });

    // test_tuple
    m.def("pair_passthrough", [](std::pair<bool, std::string> input) {
        return std::make_pair(input.second, input.first);
    }, "Return a pair in reversed order");
    m.def("tuple_passthrough", [](std::tuple<bool, std::string, int> input) {
        return std::make_tuple(std::get<2>(input), std::get<1>(input), std::get<0>(input));
    }, "Return a triple in reversed order");
    m.def("empty_tuple", []() { return std::tuple<>(); });
    static std::pair<RValueCaster, RValueCaster> lvpair;
    static std::tuple<RValueCaster, RValueCaster, RValueCaster> lvtuple;
    static std::pair<RValueCaster, std::tuple<RValueCaster, std::pair<RValueCaster, RValueCaster>>> lvnested;
    m.def("rvalue_pair", []() { return std::make_pair(RValueCaster{}, RValueCaster{}); });
    m.def("lvalue_pair", []() -> const decltype(lvpair) & { return lvpair; });
    m.def("rvalue_tuple", []() { return std::make_tuple(RValueCaster{}, RValueCaster{}, RValueCaster{}); });
    m.def("lvalue_tuple", []() -> const decltype(lvtuple) & { return lvtuple; });
    m.def("rvalue_nested", []() {
        return std::make_pair(RValueCaster{}, std::make_tuple(RValueCaster{}, std::make_pair(RValueCaster{}, RValueCaster{}))); });
    m.def("lvalue_nested", []() -> const decltype(lvnested) & { return lvnested; });

    // test_builtins_cast_return_none
    m.def("return_none_string", []() -> std::string * { return nullptr; });
    m.def("return_none_char",   []() -> const char *  { return nullptr; });
    m.def("return_none_bool",   []() -> bool *        { return nullptr; });
    m.def("return_none_int",    []() -> int *         { return nullptr; });
    m.def("return_none_float",  []() -> float *       { return nullptr; });

    // test_none_deferred
    m.def("defer_none_cstring", [](char *) { return false; });
    m.def("defer_none_cstring", [](py::none) { return true; });
    m.def("defer_none_custom", [](UserType *) { return false; });
    m.def("defer_none_custom", [](py::none) { return true; });
    m.def("nodefer_none_void", [](void *) { return true; });
    m.def("nodefer_none_void", [](py::none) { return false; });

    // test_void_caster
    m.def("load_nullptr_t", [](std::nullptr_t) {}); // not useful, but it should still compile
    m.def("cast_nullptr_t", []() { return std::nullptr_t{}; });

    // test_bool_caster
    m.def("bool_passthrough", [](bool arg) { return arg; });
    m.def("bool_passthrough_noconvert", [](bool arg) { return arg; }, py::arg().noconvert());

    // test_reference_wrapper
    m.def("refwrap_builtin", [](std::reference_wrapper<int> p) { return 10 * p.get(); });
    m.def("refwrap_usertype", [](std::reference_wrapper<UserType> p) { return p.get().value(); });
    // Not currently supported (std::pair caster has return-by-value cast operator);
    // triggers static_assert failure.
    //m.def("refwrap_pair", [](std::reference_wrapper<std::pair<int, int>>) { });

    m.def("refwrap_list", [](bool copy) {
        static IncType x1(1), x2(2);
        py::list l;
        for (auto &f : {std::ref(x1), std::ref(x2)}) {
            l.append(py::cast(f, copy ? py::return_value_policy::copy
                                      : py::return_value_policy::reference));
        }
        return l;
    }, "copy"_a);

    m.def("refwrap_iiw", [](const IncType &w) { return w.value(); });
    m.def("refwrap_call_iiw", [](IncType &w, py::function f) {
        py::list l;
        l.append(f(std::ref(w)));
        l.append(f(std::cref(w)));
        IncType x(w.value());
        l.append(f(std::ref(x)));
        IncType y(w.value());
        auto r3 = std::ref(y);
        l.append(f(r3));
        return l;
    });

    // test_complex
    m.def("complex_cast", [](float x) { return "{}"_s.format(x); });
    m.def("complex_cast", [](std::complex<float> x) { return "({}, {})"_s.format(x.real(), x.imag()); });

    // test int vs. long (Python 2)
    m.def("int_cast", []() {return (int) 42;});
    m.def("long_cast", []() {return (long) 42;});
    m.def("longlong_cast", []() {return  ULLONG_MAX;});

    /// test void* cast operator
    m.def("test_void_caster", []() -> bool {
        void *v = (void *) 0xabcd;
        py::object o = py::cast(v);
        return py::cast<void *>(o) == v;
    });
}
