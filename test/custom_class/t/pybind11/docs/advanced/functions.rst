Functions
#########

Before proceeding with this section, make sure that you are already familiar
with the basics of binding functions and classes, as explained in :doc:`/basics`
and :doc:`/classes`. The following guide is applicable to both free and member
functions, i.e. *methods* in Python.

.. _return_value_policies:

Return value policies
=====================

Python and C++ use fundamentally different ways of managing the memory and
lifetime of objects managed by them. This can lead to issues when creating
bindings for functions that return a non-trivial type. Just by looking at the
type information, it is not clear whether Python should take charge of the
returned value and eventually free its resources, or if this is handled on the
C++ side. For this reason, pybind11 provides a several *return value policy*
annotations that can be passed to the :func:`module::def` and
:func:`class_::def` functions. The default policy is
:enum:`return_value_policy::automatic`.

Return value policies are tricky, and it's very important to get them right.
Just to illustrate what can go wrong, consider the following simple example:

.. code-block:: cpp

    /* Function declaration */
    Data *get_data() { return _data; /* (pointer to a static data structure) */ }
    ...

    /* Binding code */
    m.def("get_data", &get_data); // <-- KABOOM, will cause crash when called from Python

What's going on here? When ``get_data()`` is called from Python, the return
value (a native C++ type) must be wrapped to turn it into a usable Python type.
In this case, the default return value policy (:enum:`return_value_policy::automatic`)
causes pybind11 to assume ownership of the static ``_data`` instance.

When Python's garbage collector eventually deletes the Python
wrapper, pybind11 will also attempt to delete the C++ instance (via ``operator
delete()``) due to the implied ownership. At this point, the entire application
will come crashing down, though errors could also be more subtle and involve
silent data corruption.

In the above example, the policy :enum:`return_value_policy::reference` should have
been specified so that the global data instance is only *referenced* without any
implied transfer of ownership, i.e.:

.. code-block:: cpp

    m.def("get_data", &get_data, return_value_policy::reference);

On the other hand, this is not the right policy for many other situations,
where ignoring ownership could lead to resource leaks.
As a developer using pybind11, it's important to be familiar with the different
return value policies, including which situation calls for which one of them.
The following table provides an overview of available policies:

.. tabularcolumns:: |p{0.5\textwidth}|p{0.45\textwidth}|

+--------------------------------------------------+----------------------------------------------------------------------------+
| Return value policy                              | Description                                                                |
+==================================================+============================================================================+
| :enum:`return_value_policy::take_ownership`      | Reference an existing object (i.e. do not create a new copy) and take      |
|                                                  | ownership. Python will call the destructor and delete operator when the    |
|                                                  | object's reference count reaches zero. Undefined behavior ensues when the  |
|                                                  | C++ side does the same, or when the data was not dynamically allocated.    |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::copy`                | Create a new copy of the returned object, which will be owned by Python.   |
|                                                  | This policy is comparably safe because the lifetimes of the two instances  |
|                                                  | are decoupled.                                                             |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::move`                | Use ``std::move`` to move the return value contents into a new instance    |
|                                                  | that will be owned by Python. This policy is comparably safe because the   |
|                                                  | lifetimes of the two instances (move source and destination) are decoupled.|
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::reference`           | Reference an existing object, but do not take ownership. The C++ side is   |
|                                                  | responsible for managing the object's lifetime and deallocating it when    |
|                                                  | it is no longer used. Warning: undefined behavior will ensue when the C++  |
|                                                  | side deletes an object that is still referenced and used by Python.        |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::reference_internal`  | Indicates that the lifetime of the return value is tied to the lifetime    |
|                                                  | of a parent object, namely the implicit ``this``, or ``self`` argument of  |
|                                                  | the called method or property. Internally, this policy works just like     |
|                                                  | :enum:`return_value_policy::reference` but additionally applies a          |
|                                                  | ``keep_alive<0, 1>`` *call policy* (described in the next section) that    |
|                                                  | prevents the parent object from being garbage collected as long as the     |
|                                                  | return value is referenced by Python. This is the default policy for       |
|                                                  | property getters created via ``def_property``, ``def_readwrite``, etc.     |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::automatic`           | **Default policy.** This policy falls back to the policy                   |
|                                                  | :enum:`return_value_policy::take_ownership` when the return value is a     |
|                                                  | pointer. Otherwise, it uses :enum:`return_value_policy::move` or           |
|                                                  | :enum:`return_value_policy::copy` for rvalue and lvalue references,        |
|                                                  | respectively. See above for a description of what all of these different   |
|                                                  | policies do.                                                               |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::automatic_reference` | As above, but use policy :enum:`return_value_policy::reference` when the   |
|                                                  | return value is a pointer. This is the default conversion policy for       |
|                                                  | function arguments when calling Python functions manually from C++ code    |
|                                                  | (i.e. via handle::operator()). You probably won't need to use this.        |
+--------------------------------------------------+----------------------------------------------------------------------------+

Return value policies can also be applied to properties:

.. code-block:: cpp

    class_<MyClass>(m, "MyClass")
        .def_property("data", &MyClass::getData, &MyClass::setData,
                      py::return_value_policy::copy);

Technically, the code above applies the policy to both the getter and the
setter function, however, the setter doesn't really care about *return*
value policies which makes this a convenient terse syntax. Alternatively,
targeted arguments can be passed through the :class:`cpp_function` constructor:

.. code-block:: cpp

    class_<MyClass>(m, "MyClass")
        .def_property("data"
            py::cpp_function(&MyClass::getData, py::return_value_policy::copy),
            py::cpp_function(&MyClass::setData)
        );

.. warning::

    Code with invalid return value policies might access uninitialized memory or
    free data structures multiple times, which can lead to hard-to-debug
    non-determinism and segmentation faults, hence it is worth spending the
    time to understand all the different options in the table above.

.. note::

    One important aspect of the above policies is that they only apply to
    instances which pybind11 has *not* seen before, in which case the policy
    clarifies essential questions about the return value's lifetime and
    ownership.  When pybind11 knows the instance already (as identified by its
    type and address in memory), it will return the existing Python object
    wrapper rather than creating a new copy.

.. note::

    The next section on :ref:`call_policies` discusses *call policies* that can be
    specified *in addition* to a return value policy from the list above. Call
    policies indicate reference relationships that can involve both return values
    and parameters of functions.

.. note::

   As an alternative to elaborate call policies and lifetime management logic,
   consider using smart pointers (see the section on :ref:`smart_pointers` for
   details). Smart pointers can tell whether an object is still referenced from
   C++ or Python, which generally eliminates the kinds of inconsistencies that
   can lead to crashes or undefined behavior. For functions returning smart
   pointers, it is not necessary to specify a return value policy.

.. _call_policies:

Additional call policies
========================

In addition to the above return value policies, further *call policies* can be
specified to indicate dependencies between parameters or ensure a certain state
for the function call.

Keep alive
----------

In general, this policy is required when the C++ object is any kind of container
and another object is being added to the container. ``keep_alive<Nurse, Patient>``
indicates that the argument with index ``Patient`` should be kept alive at least
until the argument with index ``Nurse`` is freed by the garbage collector. Argument
indices start at one, while zero refers to the return value. For methods, index
``1`` refers to the implicit ``this`` pointer, while regular arguments begin at
index ``2``. Arbitrarily many call policies can be specified. When a ``Nurse``
with value ``None`` is detected at runtime, the call policy does nothing.

When the nurse is not a pybind11-registered type, the implementation internally
relies on the ability to create a *weak reference* to the nurse object. When
the nurse object is not a pybind11-registered type and does not support weak
references, an exception will be thrown.

Consider the following example: here, the binding code for a list append
operation ties the lifetime of the newly added element to the underlying
container:

.. code-block:: cpp

    py::class_<List>(m, "List")
        .def("append", &List::append, py::keep_alive<1, 2>());

For consistency, the argument indexing is identical for constructors. Index
``1`` still refers to the implicit ``this`` pointer, i.e. the object which is
being constructed. Index ``0`` refers to the return type which is presumed to
be ``void`` when a constructor is viewed like a function. The following example
ties the lifetime of the constructor element to the constructed object:

.. code-block:: cpp

    py::class_<Nurse>(m, "Nurse")
        .def(py::init<Patient &>(), py::keep_alive<1, 2>());

.. note::

    ``keep_alive`` is analogous to the ``with_custodian_and_ward`` (if Nurse,
    Patient != 0) and ``with_custodian_and_ward_postcall`` (if Nurse/Patient ==
    0) policies from Boost.Python.

Call guard
----------

The ``call_guard<T>`` policy allows any scope guard type ``T`` to be placed
around the function call. For example, this definition:

.. code-block:: cpp

    m.def("foo", foo, py::call_guard<T>());

is equivalent to the following pseudocode:

.. code-block:: cpp

    m.def("foo", [](args...) {
        T scope_guard;
        return foo(args...); // forwarded arguments
    });

The only requirement is that ``T`` is default-constructible, but otherwise any
scope guard will work. This is very useful in combination with `gil_scoped_release`.
See :ref:`gil`.

Multiple guards can also be specified as ``py::call_guard<T1, T2, T3...>``. The
constructor order is left to right and destruction happens in reverse.

.. seealso::

    The file :file:`tests/test_call_policies.cpp` contains a complete example
    that demonstrates using `keep_alive` and `call_guard` in more detail.

.. _python_objects_as_args:

Python objects as arguments
===========================

pybind11 exposes all major Python types using thin C++ wrapper classes. These
wrapper classes can also be used as parameters of functions in bindings, which
makes it possible to directly work with native Python types on the C++ side.
For instance, the following statement iterates over a Python ``dict``:

.. code-block:: cpp

    void print_dict(py::dict dict) {
        /* Easily interact with Python types */
        for (auto item : dict)
            std::cout << "key=" << std::string(py::str(item.first)) << ", "
                      << "value=" << std::string(py::str(item.second)) << std::endl;
    }

It can be exported:

.. code-block:: cpp

    m.def("print_dict", &print_dict);

And used in Python as usual:

.. code-block:: pycon

    >>> print_dict({'foo': 123, 'bar': 'hello'})
    key=foo, value=123
    key=bar, value=hello

For more information on using Python objects in C++, see :doc:`/advanced/pycpp/index`.

Accepting \*args and \*\*kwargs
===============================

Python provides a useful mechanism to define functions that accept arbitrary
numbers of arguments and keyword arguments:

.. code-block:: python

   def generic(*args, **kwargs):
       ...  # do something with args and kwargs

Such functions can also be created using pybind11:

.. code-block:: cpp

   void generic(py::args args, py::kwargs kwargs) {
       /// .. do something with args
       if (kwargs)
           /// .. do something with kwargs
   }

   /// Binding code
   m.def("generic", &generic);

The class ``py::args`` derives from ``py::tuple`` and ``py::kwargs`` derives
from ``py::dict``.

You may also use just one or the other, and may combine these with other
arguments as long as the ``py::args`` and ``py::kwargs`` arguments are the last
arguments accepted by the function.

Please refer to the other examples for details on how to iterate over these,
and on how to cast their entries into C++ objects. A demonstration is also
available in ``tests/test_kwargs_and_defaults.cpp``.

.. note::

    When combining \*args or \*\*kwargs with :ref:`keyword_args` you should
    *not* include ``py::arg`` tags for the ``py::args`` and ``py::kwargs``
    arguments.

Default arguments revisited
===========================

The section on :ref:`default_args` previously discussed basic usage of default
arguments using pybind11. One noteworthy aspect of their implementation is that
default arguments are converted to Python objects right at declaration time.
Consider the following example:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg("arg") = SomeType(123));

In this case, pybind11 must already be set up to deal with values of the type
``SomeType`` (via a prior instantiation of ``py::class_<SomeType>``), or an
exception will be thrown.

Another aspect worth highlighting is that the "preview" of the default argument
in the function signature is generated using the object's ``__repr__`` method.
If not available, the signature may not be very helpful, e.g.:

.. code-block:: pycon

    FUNCTIONS
    ...
    |  myFunction(...)
    |      Signature : (MyClass, arg : SomeType = <SomeType object at 0x101b7b080>) -> NoneType
    ...

The first way of addressing this is by defining ``SomeType.__repr__``.
Alternatively, it is possible to specify the human-readable preview of the
default argument manually using the ``arg_v`` notation:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg_v("arg", SomeType(123), "SomeType(123)"));

Sometimes it may be necessary to pass a null pointer value as a default
argument. In this case, remember to cast it to the underlying type in question,
like so:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg("arg") = (SomeType *) nullptr);

.. _nonconverting_arguments:

Non-converting arguments
========================

Certain argument types may support conversion from one type to another.  Some
examples of conversions are:

* :ref:`implicit_conversions` declared using ``py::implicitly_convertible<A,B>()``
* Calling a method accepting a double with an integer argument
* Calling a ``std::complex<float>`` argument with a non-complex python type
  (for example, with a float).  (Requires the optional ``pybind11/complex.h``
  header).
* Calling a function taking an Eigen matrix reference with a numpy array of the
  wrong type or of an incompatible data layout.  (Requires the optional
  ``pybind11/eigen.h`` header).

This behaviour is sometimes undesirable: the binding code may prefer to raise
an error rather than convert the argument.  This behaviour can be obtained
through ``py::arg`` by calling the ``.noconvert()`` method of the ``py::arg``
object, such as:

.. code-block:: cpp

    m.def("floats_only", [](double f) { return 0.5 * f; }, py::arg("f").noconvert());
    m.def("floats_preferred", [](double f) { return 0.5 * f; }, py::arg("f"));

Attempting the call the second function (the one without ``.noconvert()``) with
an integer will succeed, but attempting to call the ``.noconvert()`` version
will fail with a ``TypeError``:

.. code-block:: pycon

    >>> floats_preferred(4)
    2.0
    >>> floats_only(4)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: floats_only(): incompatible function arguments. The following argument types are supported:
        1. (f: float) -> float

    Invoked with: 4

You may, of course, combine this with the :var:`_a` shorthand notation (see
:ref:`keyword_args`) and/or :ref:`default_args`.  It is also permitted to omit
the argument name by using the ``py::arg()`` constructor without an argument
name, i.e. by specifying ``py::arg().noconvert()``.

.. note::

    When specifying ``py::arg`` options it is necessary to provide the same
    number of options as the bound function has arguments.  Thus if you want to
    enable no-convert behaviour for just one of several arguments, you will
    need to specify a ``py::arg()`` annotation for each argument with the
    no-convert argument modified to ``py::arg().noconvert()``.

.. _none_arguments:

Allow/Prohibiting None arguments
================================

When a C++ type registered with :class:`py::class_` is passed as an argument to
a function taking the instance as pointer or shared holder (e.g. ``shared_ptr``
or a custom, copyable holder as described in :ref:`smart_pointers`), pybind
allows ``None`` to be passed from Python which results in calling the C++
function with ``nullptr`` (or an empty holder) for the argument.

To explicitly enable or disable this behaviour, using the
``.none`` method of the :class:`py::arg` object:

.. code-block:: cpp

    py::class_<Dog>(m, "Dog").def(py::init<>());
    py::class_<Cat>(m, "Cat").def(py::init<>());
    m.def("bark", [](Dog *dog) -> std::string {
        if (dog) return "woof!"; /* Called with a Dog instance */
        else return "(no dog)"; /* Called with None, dog == nullptr */
    }, py::arg("dog").none(true));
    m.def("meow", [](Cat *cat) -> std::string {
        // Can't be called with None argument
        return "meow";
    }, py::arg("cat").none(false));

With the above, the Python call ``bark(None)`` will return the string ``"(no
dog)"``, while attempting to call ``meow(None)`` will raise a ``TypeError``:

.. code-block:: pycon

    >>> from animals import Dog, Cat, bark, meow
    >>> bark(Dog())
    'woof!'
    >>> meow(Cat())
    'meow'
    >>> bark(None)
    '(no dog)'
    >>> meow(None)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: meow(): incompatible function arguments. The following argument types are supported:
        1. (cat: animals.Cat) -> str

    Invoked with: None

The default behaviour when the tag is unspecified is to allow ``None``.

Overload resolution order
=========================

When a function or method with multiple overloads is called from Python,
pybind11 determines which overload to call in two passes.  The first pass
attempts to call each overload without allowing argument conversion (as if
every argument had been specified as ``py::arg().noconvert()`` as described
above).

If no overload succeeds in the no-conversion first pass, a second pass is
attempted in which argument conversion is allowed (except where prohibited via
an explicit ``py::arg().noconvert()`` attribute in the function definition).

If the second pass also fails a ``TypeError`` is raised.

Within each pass, overloads are tried in the order they were registered with
pybind11.

What this means in practice is that pybind11 will prefer any overload that does
not require conversion of arguments to an overload that does, but otherwise prefers
earlier-defined overloads to later-defined ones.

.. note::

    pybind11 does *not* further prioritize based on the number/pattern of
    overloaded arguments.  That is, pybind11 does not prioritize a function
    requiring one conversion over one requiring three, but only prioritizes
    overloads requiring no conversion at all to overloads that require
    conversion of at least one argument.
