Classes
#######

This section presents advanced binding code for classes and it is assumed
that you are already familiar with the basics from :doc:`/classes`.

.. _overriding_virtuals:

Overriding virtual functions in Python
======================================

Suppose that a C++ class or interface has a virtual function that we'd like to
to override from within Python (we'll focus on the class ``Animal``; ``Dog`` is
given as a specific example of how one would do this with traditional C++
code).

.. code-block:: cpp

    class Animal {
    public:
        virtual ~Animal() { }
        virtual std::string go(int n_times) = 0;
    };

    class Dog : public Animal {
    public:
        std::string go(int n_times) override {
            std::string result;
            for (int i=0; i<n_times; ++i)
                result += "woof! ";
            return result;
        }
    };

Let's also suppose that we are given a plain function which calls the
function ``go()`` on an arbitrary ``Animal`` instance.

.. code-block:: cpp

    std::string call_go(Animal *animal) {
        return animal->go(3);
    }

Normally, the binding code for these classes would look as follows:

.. code-block:: cpp

    PYBIND11_MODULE(example, m) {
        py::class_<Animal>(m, "Animal")
            .def("go", &Animal::go);

        py::class_<Dog, Animal>(m, "Dog")
            .def(py::init<>());

        m.def("call_go", &call_go);
    }

However, these bindings are impossible to extend: ``Animal`` is not
constructible, and we clearly require some kind of "trampoline" that
redirects virtual calls back to Python.

Defining a new type of ``Animal`` from within Python is possible but requires a
helper class that is defined as follows:

.. code-block:: cpp

    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) override {
            PYBIND11_OVERLOAD_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function in C++ (must match Python name) */
                n_times      /* Argument(s) */
            );
        }
    };

The macro :func:`PYBIND11_OVERLOAD_PURE` should be used for pure virtual
functions, and :func:`PYBIND11_OVERLOAD` should be used for functions which have
a default implementation.  There are also two alternate macros
:func:`PYBIND11_OVERLOAD_PURE_NAME` and :func:`PYBIND11_OVERLOAD_NAME` which
take a string-valued name argument between the *Parent class* and *Name of the
function* slots, which defines the name of function in Python. This is required
when the C++ and Python versions of the
function have different names, e.g.  ``operator()`` vs ``__call__``.

The binding code also needs a few minor adaptations (highlighted):

.. code-block:: cpp
    :emphasize-lines: 2,3

    PYBIND11_MODULE(example, m) {
        py::class_<Animal, PyAnimal /* <--- trampoline*/>(m, "Animal")
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog, Animal>(m, "Dog")
            .def(py::init<>());

        m.def("call_go", &call_go);
    }

Importantly, pybind11 is made aware of the trampoline helper class by
specifying it as an extra template argument to :class:`class_`. (This can also
be combined with other template arguments such as a custom holder type; the
order of template types does not matter).  Following this, we are able to
define a constructor as usual.

Bindings should be made against the actual class, not the trampoline helper class.

.. code-block:: cpp
    :emphasize-lines: 3

    py::class_<Animal, PyAnimal /* <--- trampoline*/>(m, "Animal");
        .def(py::init<>())
        .def("go", &PyAnimal::go); /* <--- THIS IS WRONG, use &Animal::go */

Note, however, that the above is sufficient for allowing python classes to
extend ``Animal``, but not ``Dog``: see :ref:`virtual_and_inheritance` for the
necessary steps required to providing proper overload support for inherited
classes.

The Python session below shows how to override ``Animal::go`` and invoke it via
a virtual method call.

.. code-block:: pycon

    >>> from example import *
    >>> d = Dog()
    >>> call_go(d)
    u'woof! woof! woof! '
    >>> class Cat(Animal):
    ...     def go(self, n_times):
    ...             return "meow! " * n_times
    ...
    >>> c = Cat()
    >>> call_go(c)
    u'meow! meow! meow! '

If you are defining a custom constructor in a derived Python class, you *must*
ensure that you explicitly call the bound C++ constructor using ``__init__``,
*regardless* of whether it is a default constructor or not. Otherwise, the
memory for the C++ portion of the instance will be left uninitialized, which
will generally leave the C++ instance in an invalid state and cause undefined
behavior if the C++ instance is subsequently used.

Here is an example:

.. code-block:: python

    class Dachschund(Dog):
        def __init__(self, name):
            Dog.__init__(self) # Without this, undefined behavior may occur if the C++ portions are referenced.
            self.name = name
        def bark(self):
            return "yap!"

Note that a direct ``__init__`` constructor *should be called*, and ``super()``
should not be used. For simple cases of linear inheritance, ``super()``
may work, but once you begin mixing Python and C++ multiple inheritance,
things will fall apart due to differences between Python's MRO and C++'s
mechanisms.

Please take a look at the :ref:`macro_notes` before using this feature.

.. note::

    When the overridden type returns a reference or pointer to a type that
    pybind11 converts from Python (for example, numeric values, std::string,
    and other built-in value-converting types), there are some limitations to
    be aware of:

    - because in these cases there is no C++ variable to reference (the value
      is stored in the referenced Python variable), pybind11 provides one in
      the PYBIND11_OVERLOAD macros (when needed) with static storage duration.
      Note that this means that invoking the overloaded method on *any*
      instance will change the referenced value stored in *all* instances of
      that type.

    - Attempts to modify a non-const reference will not have the desired
      effect: it will change only the static cache variable, but this change
      will not propagate to underlying Python instance, and the change will be
      replaced the next time the overload is invoked.

.. seealso::

    The file :file:`tests/test_virtual_functions.cpp` contains a complete
    example that demonstrates how to override virtual functions using pybind11
    in more detail.

.. _virtual_and_inheritance:

Combining virtual functions and inheritance
===========================================

When combining virtual methods with inheritance, you need to be sure to provide
an override for each method for which you want to allow overrides from derived
python classes.  For example, suppose we extend the above ``Animal``/``Dog``
example as follows:

.. code-block:: cpp

    class Animal {
    public:
        virtual std::string go(int n_times) = 0;
        virtual std::string name() { return "unknown"; }
    };
    class Dog : public Animal {
    public:
        std::string go(int n_times) override {
            std::string result;
            for (int i=0; i<n_times; ++i)
                result += bark() + " ";
            return result;
        }
        virtual std::string bark() { return "woof!"; }
    };

then the trampoline class for ``Animal`` must, as described in the previous
section, override ``go()`` and ``name()``, but in order to allow python code to
inherit properly from ``Dog``, we also need a trampoline class for ``Dog`` that
overrides both the added ``bark()`` method *and* the ``go()`` and ``name()``
methods inherited from ``Animal`` (even though ``Dog`` doesn't directly
override the ``name()`` method):

.. code-block:: cpp

    class PyAnimal : public Animal {
    public:
        using Animal::Animal; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, Animal, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, Animal, name, ); }
    };
    class PyDog : public Dog {
    public:
        using Dog::Dog; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, Dog, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, Dog, name, ); }
        std::string bark() override { PYBIND11_OVERLOAD(std::string, Dog, bark, ); }
    };

.. note::

    Note the trailing commas in the ``PYBIND11_OVERLOAD`` calls to ``name()``
    and ``bark()``. These are needed to portably implement a trampoline for a
    function that does not take any arguments. For functions that take
    a nonzero number of arguments, the trailing comma must be omitted.

A registered class derived from a pybind11-registered class with virtual
methods requires a similar trampoline class, *even if* it doesn't explicitly
declare or override any virtual methods itself:

.. code-block:: cpp

    class Husky : public Dog {};
    class PyHusky : public Husky {
    public:
        using Husky::Husky; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, Husky, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, Husky, name, ); }
        std::string bark() override { PYBIND11_OVERLOAD(std::string, Husky, bark, ); }
    };

There is, however, a technique that can be used to avoid this duplication
(which can be especially helpful for a base class with several virtual
methods).  The technique involves using template trampoline classes, as
follows:

.. code-block:: cpp

    template <class AnimalBase = Animal> class PyAnimal : public AnimalBase {
    public:
        using AnimalBase::AnimalBase; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, AnimalBase, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, AnimalBase, name, ); }
    };
    template <class DogBase = Dog> class PyDog : public PyAnimal<DogBase> {
    public:
        using PyAnimal<DogBase>::PyAnimal; // Inherit constructors
        // Override PyAnimal's pure virtual go() with a non-pure one:
        std::string go(int n_times) override { PYBIND11_OVERLOAD(std::string, DogBase, go, n_times); }
        std::string bark() override { PYBIND11_OVERLOAD(std::string, DogBase, bark, ); }
    };

This technique has the advantage of requiring just one trampoline method to be
declared per virtual method and pure virtual method override.  It does,
however, require the compiler to generate at least as many methods (and
possibly more, if both pure virtual and overridden pure virtual methods are
exposed, as above).

The classes are then registered with pybind11 using:

.. code-block:: cpp

    py::class_<Animal, PyAnimal<>> animal(m, "Animal");
    py::class_<Dog, PyDog<>> dog(m, "Dog");
    py::class_<Husky, PyDog<Husky>> husky(m, "Husky");
    // ... add animal, dog, husky definitions

Note that ``Husky`` did not require a dedicated trampoline template class at
all, since it neither declares any new virtual methods nor provides any pure
virtual method implementations.

With either the repeated-virtuals or templated trampoline methods in place, you
can now create a python class that inherits from ``Dog``:

.. code-block:: python

    class ShihTzu(Dog):
        def bark(self):
            return "yip!"

.. seealso::

    See the file :file:`tests/test_virtual_functions.cpp` for complete examples
    using both the duplication and templated trampoline approaches.

.. _extended_aliases:

Extended trampoline class functionality
=======================================

The trampoline classes described in the previous sections are, by default, only
initialized when needed.  More specifically, they are initialized when a python
class actually inherits from a registered type (instead of merely creating an
instance of the registered type), or when a registered constructor is only
valid for the trampoline class but not the registered class.  This is primarily
for performance reasons: when the trampoline class is not needed for anything
except virtual method dispatching, not initializing the trampoline class
improves performance by avoiding needing to do a run-time check to see if the
inheriting python instance has an overloaded method.

Sometimes, however, it is useful to always initialize a trampoline class as an
intermediate class that does more than just handle virtual method dispatching.
For example, such a class might perform extra class initialization, extra
destruction operations, and might define new members and methods to enable a
more python-like interface to a class.

In order to tell pybind11 that it should *always* initialize the trampoline
class when creating new instances of a type, the class constructors should be
declared using ``py::init_alias<Args, ...>()`` instead of the usual
``py::init<Args, ...>()``.  This forces construction via the trampoline class,
ensuring member initialization and (eventual) destruction.

.. seealso::

    See the file :file:`tests/test_virtual_functions.cpp` for complete examples
    showing both normal and forced trampoline instantiation.

.. _custom_constructors:

Custom constructors
===================

The syntax for binding constructors was previously introduced, but it only
works when a constructor of the appropriate arguments actually exists on the
C++ side.  To extend this to more general cases, pybind11 makes it possible
to bind factory functions as constructors. For example, suppose you have a
class like this:

.. code-block:: cpp

    class Example {
    private:
        Example(int); // private constructor
    public:
        // Factory function:
        static Example create(int a) { return Example(a); }
    };

    py::class_<Example>(m, "Example")
        .def(py::init(&Example::create));

While it is possible to create a straightforward binding of the static
``create`` method, it may sometimes be preferable to expose it as a constructor
on the Python side. This can be accomplished by calling ``.def(py::init(...))``
with the function reference returning the new instance passed as an argument.
It is also possible to use this approach to bind a function returning a new
instance by raw pointer or by the holder (e.g. ``std::unique_ptr``).

The following example shows the different approaches:

.. code-block:: cpp

    class Example {
    private:
        Example(int); // private constructor
    public:
        // Factory function - returned by value:
        static Example create(int a) { return Example(a); }

        // These constructors are publicly callable:
        Example(double);
        Example(int, int);
        Example(std::string);
    };

    py::class_<Example>(m, "Example")
        // Bind the factory function as a constructor:
        .def(py::init(&Example::create))
        // Bind a lambda function returning a pointer wrapped in a holder:
        .def(py::init([](std::string arg) {
            return std::unique_ptr<Example>(new Example(arg));
        }))
        // Return a raw pointer:
        .def(py::init([](int a, int b) { return new Example(a, b); }))
        // You can mix the above with regular C++ constructor bindings as well:
        .def(py::init<double>())
        ;

When the constructor is invoked from Python, pybind11 will call the factory
function and store the resulting C++ instance in the Python instance.

When combining factory functions constructors with :ref:`virtual function
trampolines <overriding_virtuals>` there are two approaches.  The first is to
add a constructor to the alias class that takes a base value by
rvalue-reference.  If such a constructor is available, it will be used to
construct an alias instance from the value returned by the factory function.
The second option is to provide two factory functions to ``py::init()``: the
first will be invoked when no alias class is required (i.e. when the class is
being used but not inherited from in Python), and the second will be invoked
when an alias is required.

You can also specify a single factory function that always returns an alias
instance: this will result in behaviour similar to ``py::init_alias<...>()``,
as described in the :ref:`extended trampoline class documentation
<extended_aliases>`.

The following example shows the different factory approaches for a class with
an alias:

.. code-block:: cpp

    #include <pybind11/factory.h>
    class Example {
    public:
        // ...
        virtual ~Example() = default;
    };
    class PyExample : public Example {
    public:
        using Example::Example;
        PyExample(Example &&base) : Example(std::move(base)) {}
    };
    py::class_<Example, PyExample>(m, "Example")
        // Returns an Example pointer.  If a PyExample is needed, the Example
        // instance will be moved via the extra constructor in PyExample, above.
        .def(py::init([]() { return new Example(); }))
        // Two callbacks:
        .def(py::init([]() { return new Example(); } /* no alias needed */,
                      []() { return new PyExample(); } /* alias needed */))
        // *Always* returns an alias instance (like py::init_alias<>())
        .def(py::init([]() { return new PyExample(); }))
        ;

Brace initialization
--------------------

``pybind11::init<>`` internally uses C++11 brace initialization to call the
constructor of the target class. This means that it can be used to bind
*implicit* constructors as well:

.. code-block:: cpp

    struct Aggregate {
        int a;
        std::string b;
    };

    py::class_<Aggregate>(m, "Aggregate")
        .def(py::init<int, const std::string &>());

.. note::

    Note that brace initialization preferentially invokes constructor overloads
    taking a ``std::initializer_list``. In the rare event that this causes an
    issue, you can work around it by using ``py::init(...)`` with a lambda
    function that constructs the new object as desired.

.. _classes_with_non_public_destructors:

Non-public destructors
======================

If a class has a private or protected destructor (as might e.g. be the case in
a singleton pattern), a compile error will occur when creating bindings via
pybind11. The underlying issue is that the ``std::unique_ptr`` holder type that
is responsible for managing the lifetime of instances will reference the
destructor even if no deallocations ever take place. In order to expose classes
with private or protected destructors, it is possible to override the holder
type via a holder type argument to ``class_``. Pybind11 provides a helper class
``py::nodelete`` that disables any destructor invocations. In this case, it is
crucial that instances are deallocated on the C++ side to avoid memory leaks.

.. code-block:: cpp

    /* ... definition ... */

    class MyClass {
    private:
        ~MyClass() { }
    };

    /* ... binding code ... */

    py::class_<MyClass, std::unique_ptr<MyClass, py::nodelete>>(m, "MyClass")
        .def(py::init<>())

.. _implicit_conversions:

Implicit conversions
====================

Suppose that instances of two types ``A`` and ``B`` are used in a project, and
that an ``A`` can easily be converted into an instance of type ``B`` (examples of this
could be a fixed and an arbitrary precision number type).

.. code-block:: cpp

    py::class_<A>(m, "A")
        /// ... members ...

    py::class_<B>(m, "B")
        .def(py::init<A>())
        /// ... members ...

    m.def("func",
        [](const B &) { /* .... */ }
    );

To invoke the function ``func`` using a variable ``a`` containing an ``A``
instance, we'd have to write ``func(B(a))`` in Python. On the other hand, C++
will automatically apply an implicit type conversion, which makes it possible
to directly write ``func(a)``.

In this situation (i.e. where ``B`` has a constructor that converts from
``A``), the following statement enables similar implicit conversions on the
Python side:

.. code-block:: cpp

    py::implicitly_convertible<A, B>();

.. note::

    Implicit conversions from ``A`` to ``B`` only work when ``B`` is a custom
    data type that is exposed to Python via pybind11.

    To prevent runaway recursion, implicit conversions are non-reentrant: an
    implicit conversion invoked as part of another implicit conversion of the
    same type (i.e. from ``A`` to ``B``) will fail.

.. _static_properties:

Static properties
=================

The section on :ref:`properties` discussed the creation of instance properties
that are implemented in terms of C++ getters and setters.

Static properties can also be created in a similar way to expose getters and
setters of static class attributes. Note that the implicit ``self`` argument
also exists in this case and is used to pass the Python ``type`` subclass
instance. This parameter will often not be needed by the C++ side, and the
following example illustrates how to instantiate a lambda getter function
that ignores it:

.. code-block:: cpp

    py::class_<Foo>(m, "Foo")
        .def_property_readonly_static("foo", [](py::object /* self */) { return Foo(); });

Operator overloading
====================

Suppose that we're given the following ``Vector2`` class with a vector addition
and scalar multiplication operation, all implemented using overloaded operators
in C++.

.. code-block:: cpp

    class Vector2 {
    public:
        Vector2(float x, float y) : x(x), y(y) { }

        Vector2 operator+(const Vector2 &v) const { return Vector2(x + v.x, y + v.y); }
        Vector2 operator*(float value) const { return Vector2(x * value, y * value); }
        Vector2& operator+=(const Vector2 &v) { x += v.x; y += v.y; return *this; }
        Vector2& operator*=(float v) { x *= v; y *= v; return *this; }

        friend Vector2 operator*(float f, const Vector2 &v) {
            return Vector2(f * v.x, f * v.y);
        }

        std::string toString() const {
            return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
        }
    private:
        float x, y;
    };

The following snippet shows how the above operators can be conveniently exposed
to Python.

.. code-block:: cpp

    #include <pybind11/operators.h>

    PYBIND11_MODULE(example, m) {
        py::class_<Vector2>(m, "Vector2")
            .def(py::init<float, float>())
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(py::self *= float())
            .def(float() * py::self)
            .def(py::self * float())
            .def("__repr__", &Vector2::toString);
    }

Note that a line like

.. code-block:: cpp

            .def(py::self * float())

is really just short hand notation for

.. code-block:: cpp

    .def("__mul__", [](const Vector2 &a, float b) {
        return a * b;
    }, py::is_operator())

This can be useful for exposing additional operators that don't exist on the
C++ side, or to perform other types of customization. The ``py::is_operator``
flag marker is needed to inform pybind11 that this is an operator, which
returns ``NotImplemented`` when invoked with incompatible arguments rather than
throwing a type error.

.. note::

    To use the more convenient ``py::self`` notation, the additional
    header file :file:`pybind11/operators.h` must be included.

.. seealso::

    The file :file:`tests/test_operator_overloading.cpp` contains a
    complete example that demonstrates how to work with overloaded operators in
    more detail.

.. _pickling:

Pickling support
================

Python's ``pickle`` module provides a powerful facility to serialize and
de-serialize a Python object graph into a binary data stream. To pickle and
unpickle C++ classes using pybind11, a ``py::pickle()`` definition must be
provided. Suppose the class in question has the following signature:

.. code-block:: cpp

    class Pickleable {
    public:
        Pickleable(const std::string &value) : m_value(value) { }
        const std::string &value() const { return m_value; }

        void setExtra(int extra) { m_extra = extra; }
        int extra() const { return m_extra; }
    private:
        std::string m_value;
        int m_extra = 0;
    };

Pickling support in Python is enabled by defining the ``__setstate__`` and
``__getstate__`` methods [#f3]_. For pybind11 classes, use ``py::pickle()``
to bind these two functions:

.. code-block:: cpp

    py::class_<Pickleable>(m, "Pickleable")
        .def(py::init<std::string>())
        .def("value", &Pickleable::value)
        .def("extra", &Pickleable::extra)
        .def("setExtra", &Pickleable::setExtra)
        .def(py::pickle(
            [](const Pickleable &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.value(), p.extra());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Pickleable p(t[0].cast<std::string>());

                /* Assign any additional state */
                p.setExtra(t[1].cast<int>());

                return p;
            }
        ));

The ``__setstate__`` part of the ``py::picke()`` definition follows the same
rules as the single-argument version of ``py::init()``. The return type can be
a value, pointer or holder type. See :ref:`custom_constructors` for details.

An instance can now be pickled as follows:

.. code-block:: python

    try:
        import cPickle as pickle  # Use cPickle on Python 2.7
    except ImportError:
        import pickle

    p = Pickleable("test_value")
    p.setExtra(15)
    data = pickle.dumps(p, 2)

Note that only the cPickle module is supported on Python 2.7. The second
argument to ``dumps`` is also crucial: it selects the pickle protocol version
2, since the older version 1 is not supported. Newer versions are also fine—for
instance, specify ``-1`` to always use the latest available version. Beware:
failure to follow these instructions will cause important pybind11 memory
allocation routines to be skipped during unpickling, which will likely lead to
memory corruption and/or segmentation faults.

.. seealso::

    The file :file:`tests/test_pickling.cpp` contains a complete example
    that demonstrates how to pickle and unpickle types using pybind11 in more
    detail.

.. [#f3] http://docs.python.org/3/library/pickle.html#pickling-class-instances

Multiple Inheritance
====================

pybind11 can create bindings for types that derive from multiple base types
(aka. *multiple inheritance*). To do so, specify all bases in the template
arguments of the ``class_`` declaration:

.. code-block:: cpp

    py::class_<MyType, BaseType1, BaseType2, BaseType3>(m, "MyType")
       ...

The base types can be specified in arbitrary order, and they can even be
interspersed with alias types and holder types (discussed earlier in this
document)---pybind11 will automatically find out which is which. The only
requirement is that the first template argument is the type to be declared.

It is also permitted to inherit multiply from exported C++ classes in Python,
as well as inheriting from multiple Python and/or pybind11-exported classes.

There is one caveat regarding the implementation of this feature:

When only one base type is specified for a C++ type that actually has multiple
bases, pybind11 will assume that it does not participate in multiple
inheritance, which can lead to undefined behavior. In such cases, add the tag
``multiple_inheritance`` to the class constructor:

.. code-block:: cpp

    py::class_<MyType, BaseType2>(m, "MyType", py::multiple_inheritance());

The tag is redundant and does not need to be specified when multiple base types
are listed.

.. _module_local:

Module-local class bindings
===========================

When creating a binding for a class, pybind11 by default makes that binding
"global" across modules.  What this means is that a type defined in one module
can be returned from any module resulting in the same Python type.  For
example, this allows the following:

.. code-block:: cpp

    // In the module1.cpp binding code for module1:
    py::class_<Pet>(m, "Pet")
        .def(py::init<std::string>())
        .def_readonly("name", &Pet::name);

.. code-block:: cpp

    // In the module2.cpp binding code for module2:
    m.def("create_pet", [](std::string name) { return new Pet(name); });

.. code-block:: pycon

    >>> from module1 import Pet
    >>> from module2 import create_pet
    >>> pet1 = Pet("Kitty")
    >>> pet2 = create_pet("Doggy")
    >>> pet2.name()
    'Doggy'

When writing binding code for a library, this is usually desirable: this
allows, for example, splitting up a complex library into multiple Python
modules.

In some cases, however, this can cause conflicts.  For example, suppose two
unrelated modules make use of an external C++ library and each provide custom
bindings for one of that library's classes.  This will result in an error when
a Python program attempts to import both modules (directly or indirectly)
because of conflicting definitions on the external type:

.. code-block:: cpp

    // dogs.cpp

    // Binding for external library class:
    py::class<pets::Pet>(m, "Pet")
        .def("name", &pets::Pet::name);

    // Binding for local extension class:
    py::class<Dog, pets::Pet>(m, "Dog")
        .def(py::init<std::string>());

.. code-block:: cpp

    // cats.cpp, in a completely separate project from the above dogs.cpp.

    // Binding for external library class:
    py::class<pets::Pet>(m, "Pet")
        .def("get_name", &pets::Pet::name);

    // Binding for local extending class:
    py::class<Cat, pets::Pet>(m, "Cat")
        .def(py::init<std::string>());

.. code-block:: pycon

    >>> import cats
    >>> import dogs
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: generic_type: type "Pet" is already registered!

To get around this, you can tell pybind11 to keep the external class binding
localized to the module by passing the ``py::module_local()`` attribute into
the ``py::class_`` constructor:

.. code-block:: cpp

    // Pet binding in dogs.cpp:
    py::class<pets::Pet>(m, "Pet", py::module_local())
        .def("name", &pets::Pet::name);

.. code-block:: cpp

    // Pet binding in cats.cpp:
    py::class<pets::Pet>(m, "Pet", py::module_local())
        .def("get_name", &pets::Pet::name);

This makes the Python-side ``dogs.Pet`` and ``cats.Pet`` into distinct classes,
avoiding the conflict and allowing both modules to be loaded.  C++ code in the
``dogs`` module that casts or returns a ``Pet`` instance will result in a
``dogs.Pet`` Python instance, while C++ code in the ``cats`` module will result
in a ``cats.Pet`` Python instance.

This does come with two caveats, however: First, external modules cannot return
or cast a ``Pet`` instance to Python (unless they also provide their own local
bindings).  Second, from the Python point of view they are two distinct classes.

Note that the locality only applies in the C++ -> Python direction.  When
passing such a ``py::module_local`` type into a C++ function, the module-local
classes are still considered.  This means that if the following function is
added to any module (including but not limited to the ``cats`` and ``dogs``
modules above) it will be callable with either a ``dogs.Pet`` or ``cats.Pet``
argument:

.. code-block:: cpp

    m.def("pet_name", [](const pets::Pet &pet) { return pet.name(); });

For example, suppose the above function is added to each of ``cats.cpp``,
``dogs.cpp`` and ``frogs.cpp`` (where ``frogs.cpp`` is some other module that
does *not* bind ``Pets`` at all).

.. code-block:: pycon

    >>> import cats, dogs, frogs  # No error because of the added py::module_local()
    >>> mycat, mydog = cats.Cat("Fluffy"), dogs.Dog("Rover")
    >>> (cats.pet_name(mycat), dogs.pet_name(mydog))
    ('Fluffy', 'Rover')
    >>> (cats.pet_name(mydog), dogs.pet_name(mycat), frogs.pet_name(mycat))
    ('Rover', 'Fluffy', 'Fluffy')

It is possible to use ``py::module_local()`` registrations in one module even
if another module registers the same type globally: within the module with the
module-local definition, all C++ instances will be cast to the associated bound
Python type.  In other modules any such values are converted to the global
Python type created elsewhere.

.. note::

    STL bindings (as provided via the optional :file:`pybind11/stl_bind.h`
    header) apply ``py::module_local`` by default when the bound type might
    conflict with other modules; see :ref:`stl_bind` for details.

.. note::

    The localization of the bound types is actually tied to the shared object
    or binary generated by the compiler/linker.  For typical modules created
    with ``PYBIND11_MODULE()``, this distinction is not significant.  It is
    possible, however, when :ref:`embedding` to embed multiple modules in the
    same binary (see :ref:`embedding_modules`).  In such a case, the
    localization will apply across all embedded modules within the same binary.

.. seealso::

    The file :file:`tests/test_local_bindings.cpp` contains additional examples
    that demonstrate how ``py::module_local()`` works.

Binding protected member functions
==================================

It's normally not possible to expose ``protected`` member functions to Python:

.. code-block:: cpp

    class A {
    protected:
        int foo() const { return 42; }
    };

    py::class_<A>(m, "A")
        .def("foo", &A::foo); // error: 'foo' is a protected member of 'A'

On one hand, this is good because non-``public`` members aren't meant to be
accessed from the outside. But we may want to make use of ``protected``
functions in derived Python classes.

The following pattern makes this possible:

.. code-block:: cpp

    class A {
    protected:
        int foo() const { return 42; }
    };

    class Publicist : public A { // helper type for exposing protected functions
    public:
        using A::foo; // inherited with different access modifier
    };

    py::class_<A>(m, "A") // bind the primary class
        .def("foo", &Publicist::foo); // expose protected methods via the publicist

This works because ``&Publicist::foo`` is exactly the same function as
``&A::foo`` (same signature and address), just with a different access
modifier. The only purpose of the ``Publicist`` helper class is to make
the function name ``public``.

If the intent is to expose ``protected`` ``virtual`` functions which can be
overridden in Python, the publicist pattern can be combined with the previously
described trampoline:

.. code-block:: cpp

    class A {
    public:
        virtual ~A() = default;

    protected:
        virtual int foo() const { return 42; }
    };

    class Trampoline : public A {
    public:
        int foo() const override { PYBIND11_OVERLOAD(int, A, foo, ); }
    };

    class Publicist : public A {
    public:
        using A::foo;
    };

    py::class_<A, Trampoline>(m, "A") // <-- `Trampoline` here
        .def("foo", &Publicist::foo); // <-- `Publicist` here, not `Trampoline`!

.. note::

    MSVC 2015 has a compiler bug (fixed in version 2017) which
    requires a more explicit function binding in the form of
    ``.def("foo", static_cast<int (A::*)() const>(&Publicist::foo));``
    where ``int (A::*)() const`` is the type of ``A::foo``.

Custom automatic downcasters
============================

As explained in :ref:`inheritance`, pybind11 comes with built-in
understanding of the dynamic type of polymorphic objects in C++; that
is, returning a Pet to Python produces a Python object that knows it's
wrapping a Dog, if Pet has virtual methods and pybind11 knows about
Dog and this Pet is in fact a Dog. Sometimes, you might want to
provide this automatic downcasting behavior when creating bindings for
a class hierarchy that does not use standard C++ polymorphism, such as
LLVM [#f4]_. As long as there's some way to determine at runtime
whether a downcast is safe, you can proceed by specializing the
``pybind11::polymorphic_type_hook`` template:

.. code-block:: cpp

    enum class PetKind { Cat, Dog, Zebra };
    struct Pet {   // Not polymorphic: has no virtual methods
        const PetKind kind;
        int age = 0;
      protected:
        Pet(PetKind _kind) : kind(_kind) {}
    };
    struct Dog : Pet {
        Dog() : Pet(PetKind::Dog) {}
        std::string sound = "woof!";
        std::string bark() const { return sound; }
    };

    namespace pybind11 {
        template<> struct polymorphic_type_hook<Pet> {
            static const void *get(const Pet *src, const std::type_info*& type) {
                // note that src may be nullptr
                if (src && src->kind == PetKind::Dog) {
                    type = &typeid(Dog);
                    return static_cast<const Dog*>(src);
                }
                return src;
            }
        };
    } // namespace pybind11

When pybind11 wants to convert a C++ pointer of type ``Base*`` to a
Python object, it calls ``polymorphic_type_hook<Base>::get()`` to
determine if a downcast is possible. The ``get()`` function should use
whatever runtime information is available to determine if its ``src``
parameter is in fact an instance of some class ``Derived`` that
inherits from ``Base``. If it finds such a ``Derived``, it sets ``type
= &typeid(Derived)`` and returns a pointer to the ``Derived`` object
that contains ``src``. Otherwise, it just returns ``src``, leaving
``type`` at its default value of nullptr. If you set ``type`` to a
type that pybind11 doesn't know about, no downcasting will occur, and
the original ``src`` pointer will be used with its static type
``Base*``.

It is critical that the returned pointer and ``type`` argument of
``get()`` agree with each other: if ``type`` is set to something
non-null, the returned pointer must point to the start of an object
whose type is ``type``. If the hierarchy being exposed uses only
single inheritance, a simple ``return src;`` will achieve this just
fine, but in the general case, you must cast ``src`` to the
appropriate derived-class pointer (e.g. using
``static_cast<Derived>(src)``) before allowing it to be returned as a
``void*``.

.. [#f4] https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html

.. note::

    pybind11's standard support for downcasting objects whose types
    have virtual methods is implemented using
    ``polymorphic_type_hook`` too, using the standard C++ ability to
    determine the most-derived type of a polymorphic object using
    ``typeid()`` and to cast a base pointer to that most-derived type
    (even if you don't know what it is) using ``dynamic_cast<void*>``.

.. seealso::

    The file :file:`tests/test_tagbased_polymorphic.cpp` contains a
    more complete example, including a demonstration of how to provide
    automatic downcasting for an entire class hierarchy without
    writing one get() function for each class.
