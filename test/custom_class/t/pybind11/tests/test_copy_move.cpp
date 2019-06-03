/*
    tests/test_copy_move_policies.cpp -- 'copy' and 'move' return value policies
                                         and related tests

    Copyright (c) 2016 Ben North <ben@redfrontdoor.org>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/stl.h>

template <typename derived>
struct empty {
    static const derived& get_one() { return instance_; }
    static derived instance_;
};

struct lacking_copy_ctor : public empty<lacking_copy_ctor> {
    lacking_copy_ctor() {}
    lacking_copy_ctor(const lacking_copy_ctor& other) = delete;
};

template <> lacking_copy_ctor empty<lacking_copy_ctor>::instance_ = {};

struct lacking_move_ctor : public empty<lacking_move_ctor> {
    lacking_move_ctor() {}
    lacking_move_ctor(const lacking_move_ctor& other) = delete;
    lacking_move_ctor(lacking_move_ctor&& other) = delete;
};

template <> lacking_move_ctor empty<lacking_move_ctor>::instance_ = {};

/* Custom type caster move/copy test classes */
class MoveOnlyInt {
public:
    MoveOnlyInt() { print_default_created(this); }
    MoveOnlyInt(int v) : value{std::move(v)} { print_created(this, value); }
    MoveOnlyInt(MoveOnlyInt &&m) { print_move_created(this, m.value); std::swap(value, m.value); }
    MoveOnlyInt &operator=(MoveOnlyInt &&m) { print_move_assigned(this, m.value); std::swap(value, m.value); return *this; }
    MoveOnlyInt(const MoveOnlyInt &) = delete;
    MoveOnlyInt &operator=(const MoveOnlyInt &) = delete;
    ~MoveOnlyInt() { print_destroyed(this); }

    int value;
};
class MoveOrCopyInt {
public:
    MoveOrCopyInt() { print_default_created(this); }
    MoveOrCopyInt(int v) : value{std::move(v)} { print_created(this, value); }
    MoveOrCopyInt(MoveOrCopyInt &&m) { print_move_created(this, m.value); std::swap(value, m.value); }
    MoveOrCopyInt &operator=(MoveOrCopyInt &&m) { print_move_assigned(this, m.value); std::swap(value, m.value); return *this; }
    MoveOrCopyInt(const MoveOrCopyInt &c) { print_copy_created(this, c.value); value = c.value; }
    MoveOrCopyInt &operator=(const MoveOrCopyInt &c) { print_copy_assigned(this, c.value); value = c.value; return *this; }
    ~MoveOrCopyInt() { print_destroyed(this); }

    int value;
};
class CopyOnlyInt {
public:
    CopyOnlyInt() { print_default_created(this); }
    CopyOnlyInt(int v) : value{std::move(v)} { print_created(this, value); }
    CopyOnlyInt(const CopyOnlyInt &c) { print_copy_created(this, c.value); value = c.value; }
    CopyOnlyInt &operator=(const CopyOnlyInt &c) { print_copy_assigned(this, c.value); value = c.value; return *this; }
    ~CopyOnlyInt() { print_destroyed(this); }

    int value;
};
NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)
template <> struct type_caster<MoveOnlyInt> {
    PYBIND11_TYPE_CASTER(MoveOnlyInt, _("MoveOnlyInt"));
    bool load(handle src, bool) { value = MoveOnlyInt(src.cast<int>()); return true; }
    static handle cast(const MoveOnlyInt &m, return_value_policy r, handle p) { return pybind11::cast(m.value, r, p); }
};

template <> struct type_caster<MoveOrCopyInt> {
    PYBIND11_TYPE_CASTER(MoveOrCopyInt, _("MoveOrCopyInt"));
    bool load(handle src, bool) { value = MoveOrCopyInt(src.cast<int>()); return true; }
    static handle cast(const MoveOrCopyInt &m, return_value_policy r, handle p) { return pybind11::cast(m.value, r, p); }
};

template <> struct type_caster<CopyOnlyInt> {
protected:
    CopyOnlyInt value;
public:
    static constexpr auto name = _("CopyOnlyInt");
    bool load(handle src, bool) { value = CopyOnlyInt(src.cast<int>()); return true; }
    static handle cast(const CopyOnlyInt &m, return_value_policy r, handle p) { return pybind11::cast(m.value, r, p); }
    static handle cast(const CopyOnlyInt *src, return_value_policy policy, handle parent) {
        if (!src) return none().release();
        return cast(*src, policy, parent);
    }
    operator CopyOnlyInt*() { return &value; }
    operator CopyOnlyInt&() { return value; }
    template <typename T> using cast_op_type = pybind11::detail::cast_op_type<T>;
};
NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

TEST_SUBMODULE(copy_move_policies, m) {
    // test_lacking_copy_ctor
    py::class_<lacking_copy_ctor>(m, "lacking_copy_ctor")
        .def_static("get_one", &lacking_copy_ctor::get_one,
                    py::return_value_policy::copy);
    // test_lacking_move_ctor
    py::class_<lacking_move_ctor>(m, "lacking_move_ctor")
        .def_static("get_one", &lacking_move_ctor::get_one,
                    py::return_value_policy::move);

    // test_move_and_copy_casts
    m.def("move_and_copy_casts", [](py::object o) {
        int r = 0;
        r += py::cast<MoveOrCopyInt>(o).value; /* moves */
        r += py::cast<MoveOnlyInt>(o).value; /* moves */
        r += py::cast<CopyOnlyInt>(o).value; /* copies */
        MoveOrCopyInt m1(py::cast<MoveOrCopyInt>(o)); /* moves */
        MoveOnlyInt m2(py::cast<MoveOnlyInt>(o)); /* moves */
        CopyOnlyInt m3(py::cast<CopyOnlyInt>(o)); /* copies */
        r += m1.value + m2.value + m3.value;

        return r;
    });

    // test_move_and_copy_loads
    m.def("move_only", [](MoveOnlyInt m) { return m.value; });
    m.def("move_or_copy", [](MoveOrCopyInt m) { return m.value; });
    m.def("copy_only", [](CopyOnlyInt m) { return m.value; });
    m.def("move_pair", [](std::pair<MoveOnlyInt, MoveOrCopyInt> p) {
        return p.first.value + p.second.value;
    });
    m.def("move_tuple", [](std::tuple<MoveOnlyInt, MoveOrCopyInt, MoveOnlyInt> t) {
        return std::get<0>(t).value + std::get<1>(t).value + std::get<2>(t).value;
    });
    m.def("copy_tuple", [](std::tuple<CopyOnlyInt, CopyOnlyInt> t) {
        return std::get<0>(t).value + std::get<1>(t).value;
    });
    m.def("move_copy_nested", [](std::pair<MoveOnlyInt, std::pair<std::tuple<MoveOrCopyInt, CopyOnlyInt, std::tuple<MoveOnlyInt>>, MoveOrCopyInt>> x) {
        return x.first.value + std::get<0>(x.second.first).value + std::get<1>(x.second.first).value +
            std::get<0>(std::get<2>(x.second.first)).value + x.second.second.value;
    });
    m.def("move_and_copy_cstats", []() {
        ConstructorStats::gc();
        // Reset counts to 0 so that previous tests don't affect later ones:
        auto &mc = ConstructorStats::get<MoveOrCopyInt>();
        mc.move_assignments = mc.move_constructions = mc.copy_assignments = mc.copy_constructions = 0;
        auto &mo = ConstructorStats::get<MoveOnlyInt>();
        mo.move_assignments = mo.move_constructions = mo.copy_assignments = mo.copy_constructions = 0;
        auto &co = ConstructorStats::get<CopyOnlyInt>();
        co.move_assignments = co.move_constructions = co.copy_assignments = co.copy_constructions = 0;
        py::dict d;
        d["MoveOrCopyInt"] = py::cast(mc, py::return_value_policy::reference);
        d["MoveOnlyInt"] = py::cast(mo, py::return_value_policy::reference);
        d["CopyOnlyInt"] = py::cast(co, py::return_value_policy::reference);
        return d;
    });
#ifdef PYBIND11_HAS_OPTIONAL
    // test_move_and_copy_load_optional
    m.attr("has_optional") = true;
    m.def("move_optional", [](std::optional<MoveOnlyInt> o) {
        return o->value;
    });
    m.def("move_or_copy_optional", [](std::optional<MoveOrCopyInt> o) {
        return o->value;
    });
    m.def("copy_optional", [](std::optional<CopyOnlyInt> o) {
        return o->value;
    });
    m.def("move_optional_tuple", [](std::optional<std::tuple<MoveOrCopyInt, MoveOnlyInt, CopyOnlyInt>> x) {
        return std::get<0>(*x).value + std::get<1>(*x).value + std::get<2>(*x).value;
    });
#else
    m.attr("has_optional") = false;
#endif

    // #70 compilation issue if operator new is not public
    struct PrivateOpNew {
        int value = 1;
    private:
#if defined(_MSC_VER)
#  pragma warning(disable: 4822) // warning C4822: local class member function does not have a body
#endif
        void *operator new(size_t bytes);
    };
    py::class_<PrivateOpNew>(m, "PrivateOpNew").def_readonly("value", &PrivateOpNew::value);
    m.def("private_op_new_value", []() { return PrivateOpNew(); });
    m.def("private_op_new_reference", []() -> const PrivateOpNew & {
        static PrivateOpNew x{};
        return x;
    }, py::return_value_policy::reference);

    // test_move_fallback
    // #389: rvp::move should fall-through to copy on non-movable objects
    struct MoveIssue1 {
        int v;
        MoveIssue1(int v) : v{v} {}
        MoveIssue1(const MoveIssue1 &c) = default;
        MoveIssue1(MoveIssue1 &&) = delete;
    };
    py::class_<MoveIssue1>(m, "MoveIssue1").def(py::init<int>()).def_readwrite("value", &MoveIssue1::v);

    struct MoveIssue2 {
        int v;
        MoveIssue2(int v) : v{v} {}
        MoveIssue2(MoveIssue2 &&) = default;
    };
    py::class_<MoveIssue2>(m, "MoveIssue2").def(py::init<int>()).def_readwrite("value", &MoveIssue2::v);

    m.def("get_moveissue1", [](int i) { return new MoveIssue1(i); }, py::return_value_policy::move);
    m.def("get_moveissue2", [](int i) { return MoveIssue2(i); }, py::return_value_policy::move);
}
