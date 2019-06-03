/*
    tests/test_tagbased_polymorphic.cpp -- test of polymorphic_type_hook

    Copyright (c) 2018 Hudson River Trading LLC <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/stl.h>

struct Animal
{
    enum class Kind {
        Unknown = 0,
        Dog = 100, Labrador, Chihuahua, LastDog = 199,
        Cat = 200, Panther, LastCat = 299
    };
    static const std::type_info* type_of_kind(Kind kind);
    static std::string name_of_kind(Kind kind);

    const Kind kind;
    const std::string name;

  protected:
    Animal(const std::string& _name, Kind _kind)
        : kind(_kind), name(_name)
    {}
};

struct Dog : Animal
{
    Dog(const std::string& _name, Kind _kind = Kind::Dog) : Animal(_name, _kind) {}
    std::string bark() const { return name_of_kind(kind) + " " + name + " goes " + sound; }
    std::string sound = "WOOF!";
};

struct Labrador : Dog
{
    Labrador(const std::string& _name, int _excitement = 9001)
        : Dog(_name, Kind::Labrador), excitement(_excitement) {}
    int excitement;
};

struct Chihuahua : Dog
{
    Chihuahua(const std::string& _name) : Dog(_name, Kind::Chihuahua) { sound = "iyiyiyiyiyi"; }
    std::string bark() const { return Dog::bark() + " and runs in circles"; }
};

struct Cat : Animal
{
    Cat(const std::string& _name, Kind _kind = Kind::Cat) : Animal(_name, _kind) {}
    std::string purr() const { return "mrowr"; }
};

struct Panther : Cat
{
    Panther(const std::string& _name) : Cat(_name, Kind::Panther) {}
    std::string purr() const { return "mrrrRRRRRR"; }
};

std::vector<std::unique_ptr<Animal>> create_zoo()
{
    std::vector<std::unique_ptr<Animal>> ret;
    ret.emplace_back(new Labrador("Fido", 15000));

    // simulate some new type of Dog that the Python bindings
    // haven't been updated for; it should still be considered
    // a Dog, not just an Animal.
    ret.emplace_back(new Dog("Ginger", Dog::Kind(150)));

    ret.emplace_back(new Chihuahua("Hertzl"));
    ret.emplace_back(new Cat("Tiger", Cat::Kind::Cat));
    ret.emplace_back(new Panther("Leo"));
    return ret;
}

const std::type_info* Animal::type_of_kind(Kind kind)
{
    switch (kind) {
        case Kind::Unknown: break;

        case Kind::Dog: break;
        case Kind::Labrador: return &typeid(Labrador);
        case Kind::Chihuahua: return &typeid(Chihuahua);
        case Kind::LastDog: break;

        case Kind::Cat: break;
        case Kind::Panther: return &typeid(Panther);
        case Kind::LastCat: break;
    }

    if (kind >= Kind::Dog && kind <= Kind::LastDog) return &typeid(Dog);
    if (kind >= Kind::Cat && kind <= Kind::LastCat) return &typeid(Cat);
    return nullptr;
}

std::string Animal::name_of_kind(Kind kind)
{
    std::string raw_name = type_of_kind(kind)->name();
    py::detail::clean_type_id(raw_name);
    return raw_name;
}

namespace pybind11 {
    template <typename itype>
    struct polymorphic_type_hook<itype, detail::enable_if_t<std::is_base_of<Animal, itype>::value>>
    {
        static const void *get(const itype *src, const std::type_info*& type)
        { type = src ? Animal::type_of_kind(src->kind) : nullptr; return src; }
    };
}

TEST_SUBMODULE(tagbased_polymorphic, m) {
    py::class_<Animal>(m, "Animal")
        .def_readonly("name", &Animal::name);
    py::class_<Dog, Animal>(m, "Dog")
        .def(py::init<std::string>())
        .def_readwrite("sound", &Dog::sound)
        .def("bark", &Dog::bark);
    py::class_<Labrador, Dog>(m, "Labrador")
        .def(py::init<std::string, int>(), "name"_a, "excitement"_a = 9001)
        .def_readwrite("excitement", &Labrador::excitement);
    py::class_<Chihuahua, Dog>(m, "Chihuahua")
        .def(py::init<std::string>())
        .def("bark", &Chihuahua::bark);
    py::class_<Cat, Animal>(m, "Cat")
        .def(py::init<std::string>())
        .def("purr", &Cat::purr);
    py::class_<Panther, Cat>(m, "Panther")
        .def(py::init<std::string>())
        .def("purr", &Panther::purr);
    m.def("create_zoo", &create_zoo);
};
