Eigen
#####

`Eigen <http://eigen.tuxfamily.org>`_ is C++ header-based library for dense and
sparse linear algebra. Due to its popularity and widespread adoption, pybind11
provides transparent conversion and limited mapping support between Eigen and
Scientific Python linear algebra data types.

To enable the built-in Eigen support you must include the optional header file
:file:`pybind11/eigen.h`.

Pass-by-value
=============

When binding a function with ordinary Eigen dense object arguments (for
example, ``Eigen::MatrixXd``), pybind11 will accept any input value that is
already (or convertible to) a ``numpy.ndarray`` with dimensions compatible with
the Eigen type, copy its values into a temporary Eigen variable of the
appropriate type, then call the function with this temporary variable.

Sparse matrices are similarly copied to or from
``scipy.sparse.csr_matrix``/``scipy.sparse.csc_matrix`` objects.

Pass-by-reference
=================

One major limitation of the above is that every data conversion implicitly
involves a copy, which can be both expensive (for large matrices) and disallows
binding functions that change their (Matrix) arguments.  Pybind11 allows you to
work around this by using Eigen's ``Eigen::Ref<MatrixType>`` class much as you
would when writing a function taking a generic type in Eigen itself (subject to
some limitations discussed below).

When calling a bound function accepting a ``Eigen::Ref<const MatrixType>``
type, pybind11 will attempt to avoid copying by using an ``Eigen::Map`` object
that maps into the source ``numpy.ndarray`` data: this requires both that the
data types are the same (e.g. ``dtype='float64'`` and ``MatrixType::Scalar`` is
``double``); and that the storage is layout compatible.  The latter limitation
is discussed in detail in the section below, and requires careful
consideration: by default, numpy matrices and eigen matrices are *not* storage
compatible.

If the numpy matrix cannot be used as is (either because its types differ, e.g.
passing an array of integers to an Eigen parameter requiring doubles, or
because the storage is incompatible), pybind11 makes a temporary copy and
passes the copy instead.

When a bound function parameter is instead ``Eigen::Ref<MatrixType>`` (note the
lack of ``const``), pybind11 will only allow the function to be called if it
can be mapped *and* if the numpy array is writeable (that is
``a.flags.writeable`` is true).  Any access (including modification) made to
the passed variable will be transparently carried out directly on the
``numpy.ndarray``.

This means you can can write code such as the following and have it work as
expected:

.. code-block:: cpp

    void scale_by_2(Eigen::Ref<Eigen::VectorXd> v) {
        v *= 2;
    }

Note, however, that you will likely run into limitations due to numpy and
Eigen's difference default storage order for data; see the below section on
:ref:`storage_orders` for details on how to bind code that won't run into such
limitations.

.. note::

    Passing by reference is not supported for sparse types.

Returning values to Python
==========================

When returning an ordinary dense Eigen matrix type to numpy (e.g.
``Eigen::MatrixXd`` or ``Eigen::RowVectorXf``) pybind11 keeps the matrix and
returns a numpy array that directly references the Eigen matrix: no copy of the
data is performed.  The numpy array will have ``array.flags.owndata`` set to
``False`` to indicate that it does not own the data, and the lifetime of the
stored Eigen matrix will be tied to the returned ``array``.

If you bind a function with a non-reference, ``const`` return type (e.g.
``const Eigen::MatrixXd``), the same thing happens except that pybind11 also
sets the numpy array's ``writeable`` flag to false.

If you return an lvalue reference or pointer, the usual pybind11 rules apply,
as dictated by the binding function's return value policy (see the
documentation on :ref:`return_value_policies` for full details).  That means,
without an explicit return value policy, lvalue references will be copied and
pointers will be managed by pybind11.  In order to avoid copying, you should
explicitly specify an appropriate return value policy, as in the following
example:

.. code-block:: cpp

    class MyClass {
        Eigen::MatrixXd big_mat = Eigen::MatrixXd::Zero(10000, 10000);
    public:
        Eigen::MatrixXd &getMatrix() { return big_mat; }
        const Eigen::MatrixXd &viewMatrix() { return big_mat; }
    };

    // Later, in binding code:
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def("copy_matrix", &MyClass::getMatrix) // Makes a copy!
        .def("get_matrix", &MyClass::getMatrix, py::return_value_policy::reference_internal)
        .def("view_matrix", &MyClass::viewMatrix, py::return_value_policy::reference_internal)
        ;

.. code-block:: python

    a = MyClass()
    m = a.get_matrix()   # flags.writeable = True,  flags.owndata = False
    v = a.view_matrix()  # flags.writeable = False, flags.owndata = False
    c = a.copy_matrix()  # flags.writeable = True,  flags.owndata = True
    # m[5,6] and v[5,6] refer to the same element, c[5,6] does not.

Note in this example that ``py::return_value_policy::reference_internal`` is
used to tie the life of the MyClass object to the life of the returned arrays.

You may also return an ``Eigen::Ref``, ``Eigen::Map`` or other map-like Eigen
object (for example, the return value of ``matrix.block()`` and related
methods) that map into a dense Eigen type.  When doing so, the default
behaviour of pybind11 is to simply reference the returned data: you must take
care to ensure that this data remains valid!  You may ask pybind11 to
explicitly *copy* such a return value by using the
``py::return_value_policy::copy`` policy when binding the function.  You may
also use ``py::return_value_policy::reference_internal`` or a
``py::keep_alive`` to ensure the data stays valid as long as the returned numpy
array does.

When returning such a reference of map, pybind11 additionally respects the
readonly-status of the returned value, marking the numpy array as non-writeable
if the reference or map was itself read-only.

.. note::

    Sparse types are always copied when returned.

.. _storage_orders:

Storage orders
==============

Passing arguments via ``Eigen::Ref`` has some limitations that you must be
aware of in order to effectively pass matrices by reference.  First and
foremost is that the default ``Eigen::Ref<MatrixType>`` class requires
contiguous storage along columns (for column-major types, the default in Eigen)
or rows if ``MatrixType`` is specifically an ``Eigen::RowMajor`` storage type.
The former, Eigen's default, is incompatible with ``numpy``'s default row-major
storage, and so you will not be able to pass numpy arrays to Eigen by reference
without making one of two changes.

(Note that this does not apply to vectors (or column or row matrices): for such
types the "row-major" and "column-major" distinction is meaningless).

The first approach is to change the use of ``Eigen::Ref<MatrixType>`` to the
more general ``Eigen::Ref<MatrixType, 0, Eigen::Stride<Eigen::Dynamic,
Eigen::Dynamic>>`` (or similar type with a fully dynamic stride type in the
third template argument).  Since this is a rather cumbersome type, pybind11
provides a ``py::EigenDRef<MatrixType>`` type alias for your convenience (along
with EigenDMap for the equivalent Map, and EigenDStride for just the stride
type).

This type allows Eigen to map into any arbitrary storage order.  This is not
the default in Eigen for performance reasons: contiguous storage allows
vectorization that cannot be done when storage is not known to be contiguous at
compile time.  The default ``Eigen::Ref`` stride type allows non-contiguous
storage along the outer dimension (that is, the rows of a column-major matrix
or columns of a row-major matrix), but not along the inner dimension.

This type, however, has the added benefit of also being able to map numpy array
slices.  For example, the following (contrived) example uses Eigen with a numpy
slice to multiply by 2 all coefficients that are both on even rows (0, 2, 4,
...) and in columns 2, 5, or 8:

.. code-block:: cpp

    m.def("scale", [](py::EigenDRef<Eigen::MatrixXd> m, double c) { m *= c; });

.. code-block:: python

    # a = np.array(...)
    scale_by_2(myarray[0::2, 2:9:3])

The second approach to avoid copying is more intrusive: rearranging the
underlying data types to not run into the non-contiguous storage problem in the
first place.  In particular, that means using matrices with ``Eigen::RowMajor``
storage, where appropriate, such as:

.. code-block:: cpp

    using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // Use RowMatrixXd instead of MatrixXd

Now bound functions accepting ``Eigen::Ref<RowMatrixXd>`` arguments will be
callable with numpy's (default) arrays without involving a copying.

You can, alternatively, change the storage order that numpy arrays use by
adding the ``order='F'`` option when creating an array:

.. code-block:: python

    myarray = np.array(source, order='F')

Such an object will be passable to a bound function accepting an
``Eigen::Ref<MatrixXd>`` (or similar column-major Eigen type).

One major caveat with this approach, however, is that it is not entirely as
easy as simply flipping all Eigen or numpy usage from one to the other: some
operations may alter the storage order of a numpy array.  For example, ``a2 =
array.transpose()`` results in ``a2`` being a view of ``array`` that references
the same data, but in the opposite storage order!

While this approach allows fully optimized vectorized calculations in Eigen, it
cannot be used with array slices, unlike the first approach.

When *returning* a matrix to Python (either a regular matrix, a reference via
``Eigen::Ref<>``, or a map/block into a matrix), no special storage
consideration is required: the created numpy array will have the required
stride that allows numpy to properly interpret the array, whatever its storage
order.

Failing rather than copying
===========================

The default behaviour when binding ``Eigen::Ref<const MatrixType>`` eigen
references is to copy matrix values when passed a numpy array that does not
conform to the element type of ``MatrixType`` or does not have a compatible
stride layout.  If you want to explicitly avoid copying in such a case, you
should bind arguments using the ``py::arg().noconvert()`` annotation (as
described in the :ref:`nonconverting_arguments` documentation).

The following example shows an example of arguments that don't allow data
copying to take place:

.. code-block:: cpp

    // The method and function to be bound:
    class MyClass {
        // ...
        double some_method(const Eigen::Ref<const MatrixXd> &matrix) { /* ... */ }
    };
    float some_function(const Eigen::Ref<const MatrixXf> &big,
                        const Eigen::Ref<const MatrixXf> &small) {
        // ...
    }

    // The associated binding code:
    using namespace pybind11::literals; // for "arg"_a
    py::class_<MyClass>(m, "MyClass")
        // ... other class definitions
        .def("some_method", &MyClass::some_method, py::arg().noconvert());

    m.def("some_function", &some_function,
        "big"_a.noconvert(), // <- Don't allow copying for this arg
        "small"_a            // <- This one can be copied if needed
    );

With the above binding code, attempting to call the the ``some_method(m)``
method on a ``MyClass`` object, or attempting to call ``some_function(m, m2)``
will raise a ``RuntimeError`` rather than making a temporary copy of the array.
It will, however, allow the ``m2`` argument to be copied into a temporary if
necessary.

Note that explicitly specifying ``.noconvert()`` is not required for *mutable*
Eigen references (e.g. ``Eigen::Ref<MatrixXd>`` without ``const`` on the
``MatrixXd``): mutable references will never be called with a temporary copy.

Vectors versus column/row matrices
==================================

Eigen and numpy have fundamentally different notions of a vector.  In Eigen, a
vector is simply a matrix with the number of columns or rows set to 1 at
compile time (for a column vector or row vector, respectively).  Numpy, in
contrast, has comparable 2-dimensional 1xN and Nx1 arrays, but *also* has
1-dimensional arrays of size N.

When passing a 2-dimensional 1xN or Nx1 array to Eigen, the Eigen type must
have matching dimensions: That is, you cannot pass a 2-dimensional Nx1 numpy
array to an Eigen value expecting a row vector, or a 1xN numpy array as a
column vector argument.

On the other hand, pybind11 allows you to pass 1-dimensional arrays of length N
as Eigen parameters.  If the Eigen type can hold a column vector of length N it
will be passed as such a column vector.  If not, but the Eigen type constraints
will accept a row vector, it will be passed as a row vector.  (The column
vector takes precedence when both are supported, for example, when passing a
1D numpy array to a MatrixXd argument).  Note that the type need not be
expicitly a vector: it is permitted to pass a 1D numpy array of size 5 to an
Eigen ``Matrix<double, Dynamic, 5>``: you would end up with a 1x5 Eigen matrix.
Passing the same to an ``Eigen::MatrixXd`` would result in a 5x1 Eigen matrix.

When returning an eigen vector to numpy, the conversion is ambiguous: a row
vector of length 4 could be returned as either a 1D array of length 4, or as a
2D array of size 1x4.  When encoutering such a situation, pybind11 compromises
by considering the returned Eigen type: if it is a compile-time vector--that
is, the type has either the number of rows or columns set to 1 at compile
time--pybind11 converts to a 1D numpy array when returning the value.  For
instances that are a vector only at run-time (e.g. ``MatrixXd``,
``Matrix<float, Dynamic, 4>``), pybind11 returns the vector as a 2D array to
numpy.  If this isn't want you want, you can use ``array.reshape(...)`` to get
a view of the same data in the desired dimensions.

.. seealso::

    The file :file:`tests/test_eigen.cpp` contains a complete example that
    shows how to pass Eigen sparse and dense data types in more detail.
