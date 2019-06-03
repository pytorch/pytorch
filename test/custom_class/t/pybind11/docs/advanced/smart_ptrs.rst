Smart pointers
##############

std::unique_ptr
===============

Given a class ``Example`` with Python bindings, it's possible to return
instances wrapped in C++11 unique pointers, like so

.. code-block:: cpp

    std::unique_ptr<Example> create_example() { return std::unique_ptr<Example>(new Example()); }

.. code-block:: cpp

    m.def("create_example", &create_example);

In other words, there is nothing special that needs to be done. While returning
unique pointers in this way is allowed, it is *illegal* to use them as function
arguments. For instance, the following function signature cannot be processed
by pybind11.

.. code-block:: cpp

    void do_something_with_example(std::unique_ptr<Example> ex) { ... }

The above signature would imply that Python needs to give up ownership of an
object that is passed to this function, which is generally not possible (for
instance, the object might be referenced elsewhere).

std::shared_ptr
===============

The binding generator for classes, :class:`class_`, can be passed a template
type that denotes a special *holder* type that is used to manage references to
the object.  If no such holder type template argument is given, the default for
a type named ``Type`` is ``std::unique_ptr<Type>``, which means that the object
is deallocated when Python's reference count goes to zero.

It is possible to switch to other types of reference counting wrappers or smart
pointers, which is useful in codebases that rely on them. For instance, the
following snippet causes ``std::shared_ptr`` to be used instead.

.. code-block:: cpp

    py::class_<Example, std::shared_ptr<Example> /* <- holder type */> obj(m, "Example");

Note that any particular class can only be associated with a single holder type.

One potential stumbling block when using holder types is that they need to be
applied consistently. Can you guess what's broken about the following binding
code?

.. code-block:: cpp

    class Child { };

    class Parent {
    public:
       Parent() : child(std::make_shared<Child>()) { }
       Child *get_child() { return child.get(); }  /* Hint: ** DON'T DO THIS ** */
    private:
        std::shared_ptr<Child> child;
    };

    PYBIND11_MODULE(example, m) {
        py::class_<Child, std::shared_ptr<Child>>(m, "Child");

        py::class_<Parent, std::shared_ptr<Parent>>(m, "Parent")
           .def(py::init<>())
           .def("get_child", &Parent::get_child);
    }

The following Python code will cause undefined behavior (and likely a
segmentation fault).

.. code-block:: python

   from example import Parent
   print(Parent().get_child())

The problem is that ``Parent::get_child()`` returns a pointer to an instance of
``Child``, but the fact that this instance is already managed by
``std::shared_ptr<...>`` is lost when passing raw pointers. In this case,
pybind11 will create a second independent ``std::shared_ptr<...>`` that also
claims ownership of the pointer. In the end, the object will be freed **twice**
since these shared pointers have no way of knowing about each other.

There are two ways to resolve this issue:

1. For types that are managed by a smart pointer class, never use raw pointers
   in function arguments or return values. In other words: always consistently
   wrap pointers into their designated holder types (such as
   ``std::shared_ptr<...>``). In this case, the signature of ``get_child()``
   should be modified as follows:

.. code-block:: cpp

    std::shared_ptr<Child> get_child() { return child; }

2. Adjust the definition of ``Child`` by specifying
   ``std::enable_shared_from_this<T>`` (see cppreference_ for details) as a
   base class. This adds a small bit of information to ``Child`` that allows
   pybind11 to realize that there is already an existing
   ``std::shared_ptr<...>`` and communicate with it. In this case, the
   declaration of ``Child`` should look as follows:

.. _cppreference: http://en.cppreference.com/w/cpp/memory/enable_shared_from_this

.. code-block:: cpp

    class Child : public std::enable_shared_from_this<Child> { };

.. _smart_pointers:

Custom smart pointers
=====================

pybind11 supports ``std::unique_ptr`` and ``std::shared_ptr`` right out of the
box. For any other custom smart pointer, transparent conversions can be enabled
using a macro invocation similar to the following. It must be declared at the
top namespace level before any binding code:

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>);

The first argument of :func:`PYBIND11_DECLARE_HOLDER_TYPE` should be a
placeholder name that is used as a template parameter of the second argument.
Thus, feel free to use any identifier, but use it consistently on both sides;
also, don't use the name of a type that already exists in your codebase.

The macro also accepts a third optional boolean parameter that is set to false
by default. Specify

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>, true);

if ``SmartPtr<T>`` can always be initialized from a ``T*`` pointer without the
risk of inconsistencies (such as multiple independent ``SmartPtr`` instances
believing that they are the sole owner of the ``T*`` pointer). A common
situation where ``true`` should be passed is when the ``T`` instances use
*intrusive* reference counting.

Please take a look at the :ref:`macro_notes` before using this feature.

By default, pybind11 assumes that your custom smart pointer has a standard
interface, i.e. provides a ``.get()`` member function to access the underlying
raw pointer. If this is not the case, pybind11's ``holder_helper`` must be
specialized:

.. code-block:: cpp

    // Always needed for custom holder types
    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>);

    // Only needed if the type's `.get()` goes by another name
    namespace pybind11 { namespace detail {
        template <typename T>
        struct holder_helper<SmartPtr<T>> { // <-- specialization
            static const T *get(const SmartPtr<T> &p) { return p.getPointer(); }
        };
    }}

The above specialization informs pybind11 that the custom ``SmartPtr`` class
provides ``.get()`` functionality via ``.getPointer()``.

.. seealso::

    The file :file:`tests/test_smart_ptr.cpp` contains a complete example
    that demonstrates how to work with custom reference-counting holder types
    in more detail.
