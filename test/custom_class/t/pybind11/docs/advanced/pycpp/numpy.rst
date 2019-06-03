.. _numpy:

NumPy
#####

Buffer protocol
===============

Python supports an extremely general and convenient approach for exchanging
data between plugin libraries. Types can expose a buffer view [#f2]_, which
provides fast direct access to the raw internal data representation. Suppose we
want to bind the following simplistic Matrix class:

.. code-block:: cpp

    class Matrix {
    public:
        Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
            m_data = new float[rows*cols];
        }
        float *data() { return m_data; }
        size_t rows() const { return m_rows; }
        size_t cols() const { return m_cols; }
    private:
        size_t m_rows, m_cols;
        float *m_data;
    };

The following binding code exposes the ``Matrix`` contents as a buffer object,
making it possible to cast Matrices into NumPy arrays. It is even possible to
completely avoid copy operations with Python expressions like
``np.array(matrix_instance, copy = False)``.

.. code-block:: cpp

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { m.rows(), m.cols() },                 /* Buffer dimensions */
                { sizeof(float) * m.cols(),             /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

Supporting the buffer protocol in a new type involves specifying the special
``py::buffer_protocol()`` tag in the ``py::class_`` constructor and calling the
``def_buffer()`` method with a lambda function that creates a
``py::buffer_info`` description record on demand describing a given matrix
instance. The contents of ``py::buffer_info`` mirror the Python buffer protocol
specification.

.. code-block:: cpp

    struct buffer_info {
        void *ptr;
        ssize_t itemsize;
        std::string format;
        ssize_t ndim;
        std::vector<ssize_t> shape;
        std::vector<ssize_t> strides;
    };

To create a C++ function that can take a Python buffer object as an argument,
simply use the type ``py::buffer`` as one of its arguments. Buffers can exist
in a great variety of configurations, hence some safety checks are usually
necessary in the function body. Below, you can see an basic example on how to
define a custom constructor for the Eigen double precision matrix
(``Eigen::MatrixXd``) type, which supports initialization from compatible
buffer objects (e.g. a NumPy matrix).

.. code-block:: cpp

    /* Bind MatrixXd (or some other Eigen type) to Python */
    typedef Eigen::MatrixXd Matrix;

    typedef Matrix::Scalar Scalar;
    constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def("__init__", [](Matrix &m, py::buffer b) {
            typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();

            /* Some sanity checks ... */
            if (info.format != py::format_descriptor<Scalar>::format())
                throw std::runtime_error("Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            auto strides = Strides(
                info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar),
                info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar));

            auto map = Eigen::Map<Matrix, 0, Strides>(
                static_cast<Scalar *>(info.ptr), info.shape[0], info.shape[1], strides);

            new (&m) Matrix(map);
        });

For reference, the ``def_buffer()`` call for this Eigen data type should look
as follows:

.. code-block:: cpp

    .def_buffer([](Matrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                                /* Pointer to buffer */
            sizeof(Scalar),                          /* Size of one scalar */
            py::format_descriptor<Scalar>::format(), /* Python struct-style format descriptor */
            2,                                       /* Number of dimensions */
            { m.rows(), m.cols() },                  /* Buffer dimensions */
            { sizeof(Scalar) * (rowMajor ? m.cols() : 1),
              sizeof(Scalar) * (rowMajor ? 1 : m.rows()) }
                                                     /* Strides (in bytes) for each index */
        );
     })

For a much easier approach of binding Eigen types (although with some
limitations), refer to the section on :doc:`/advanced/cast/eigen`.

.. seealso::

    The file :file:`tests/test_buffers.cpp` contains a complete example
    that demonstrates using the buffer protocol with pybind11 in more detail.

.. [#f2] http://docs.python.org/3/c-api/buffer.html

Arrays
======

By exchanging ``py::buffer`` with ``py::array`` in the above snippet, we can
restrict the function so that it only accepts NumPy arrays (rather than any
type of Python object satisfying the buffer protocol).

In many situations, we want to define a function which only accepts a NumPy
array of a certain data type. This is possible via the ``py::array_t<T>``
template. For instance, the following function requires the argument to be a
NumPy array containing double precision values.

.. code-block:: cpp

    void f(py::array_t<double> array);

When it is invoked with a different type (e.g. an integer or a list of
integers), the binding code will attempt to cast the input into a NumPy array
of the requested type. Note that this feature requires the
:file:`pybind11/numpy.h` header to be included.

Data in NumPy arrays is not guaranteed to packed in a dense manner;
furthermore, entries can be separated by arbitrary column and row strides.
Sometimes, it can be useful to require a function to only accept dense arrays
using either the C (row-major) or Fortran (column-major) ordering. This can be
accomplished via a second template argument with values ``py::array::c_style``
or ``py::array::f_style``.

.. code-block:: cpp

    void f(py::array_t<double, py::array::c_style | py::array::forcecast> array);

The ``py::array::forcecast`` argument is the default value of the second
template parameter, and it ensures that non-conforming arguments are converted
into an array satisfying the specified requirements instead of trying the next
function overload.

Structured types
================

In order for ``py::array_t`` to work with structured (record) types, we first
need to register the memory layout of the type. This can be done via
``PYBIND11_NUMPY_DTYPE`` macro, called in the plugin definition code, which
expects the type followed by field names:

.. code-block:: cpp

    struct A {
        int x;
        double y;
    };

    struct B {
        int z;
        A a;
    };

    // ...
    PYBIND11_MODULE(test, m) {
        // ...

        PYBIND11_NUMPY_DTYPE(A, x, y);
        PYBIND11_NUMPY_DTYPE(B, z, a);
        /* now both A and B can be used as template arguments to py::array_t */
    }

The structure should consist of fundamental arithmetic types, ``std::complex``,
previously registered substructures, and arrays of any of the above. Both C++
arrays and ``std::array`` are supported. While there is a static assertion to
prevent many types of unsupported structures, it is still the user's
responsibility to use only "plain" structures that can be safely manipulated as
raw memory without violating invariants.

Vectorizing functions
=====================

Suppose we want to bind a function with the following signature to Python so
that it can process arbitrary NumPy array arguments (vectors, matrices, general
N-D arrays) in addition to its normal arguments:

.. code-block:: cpp

    double my_func(int x, float y, double z);

After including the ``pybind11/numpy.h`` header, this is extremely simple:

.. code-block:: cpp

    m.def("vectorized_func", py::vectorize(my_func));

Invoking the function like below causes 4 calls to be made to ``my_func`` with
each of the array elements. The significant advantage of this compared to
solutions like ``numpy.vectorize()`` is that the loop over the elements runs
entirely on the C++ side and can be crunched down into a tight, optimized loop
by the compiler. The result is returned as a NumPy array of type
``numpy.dtype.float64``.

.. code-block:: pycon

    >>> x = np.array([[1, 3],[5, 7]])
    >>> y = np.array([[2, 4],[6, 8]])
    >>> z = 3
    >>> result = vectorized_func(x, y, z)

The scalar argument ``z`` is transparently replicated 4 times.  The input
arrays ``x`` and ``y`` are automatically converted into the right types (they
are of type  ``numpy.dtype.int64`` but need to be ``numpy.dtype.int32`` and
``numpy.dtype.float32``, respectively).

.. note::

    Only arithmetic, complex, and POD types passed by value or by ``const &``
    reference are vectorized; all other arguments are passed through as-is.
    Functions taking rvalue reference arguments cannot be vectorized.

In cases where the computation is too complicated to be reduced to
``vectorize``, it will be necessary to create and access the buffer contents
manually. The following snippet contains a complete example that shows how this
works (the code is somewhat contrived, since it could have been done more
simply using ``vectorize``).

.. code-block:: cpp

    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>

    namespace py = pybind11;

    py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
        py::buffer_info buf1 = input1.request(), buf2 = input2.request();

        if (buf1.ndim != 1 || buf2.ndim != 1)
            throw std::runtime_error("Number of dimensions must be one");

        if (buf1.size != buf2.size)
            throw std::runtime_error("Input shapes must match");

        /* No pointer is passed, so NumPy will allocate the buffer */
        auto result = py::array_t<double>(buf1.size);

        py::buffer_info buf3 = result.request();

        double *ptr1 = (double *) buf1.ptr,
               *ptr2 = (double *) buf2.ptr,
               *ptr3 = (double *) buf3.ptr;

        for (size_t idx = 0; idx < buf1.shape[0]; idx++)
            ptr3[idx] = ptr1[idx] + ptr2[idx];

        return result;
    }

    PYBIND11_MODULE(test, m) {
        m.def("add_arrays", &add_arrays, "Add two NumPy arrays");
    }

.. seealso::

    The file :file:`tests/test_numpy_vectorize.cpp` contains a complete
    example that demonstrates using :func:`vectorize` in more detail.

Direct access
=============

For performance reasons, particularly when dealing with very large arrays, it
is often desirable to directly access array elements without internal checking
of dimensions and bounds on every access when indices are known to be already
valid.  To avoid such checks, the ``array`` class and ``array_t<T>`` template
class offer an unchecked proxy object that can be used for this unchecked
access through the ``unchecked<N>`` and ``mutable_unchecked<N>`` methods,
where ``N`` gives the required dimensionality of the array:

.. code-block:: cpp

    m.def("sum_3d", [](py::array_t<double> x) {
        auto r = x.unchecked<3>(); // x must have ndim = 3; can be non-writeable
        double sum = 0;
        for (ssize_t i = 0; i < r.shape(0); i++)
            for (ssize_t j = 0; j < r.shape(1); j++)
                for (ssize_t k = 0; k < r.shape(2); k++)
                    sum += r(i, j, k);
        return sum;
    });
    m.def("increment_3d", [](py::array_t<double> x) {
        auto r = x.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
        for (ssize_t i = 0; i < r.shape(0); i++)
            for (ssize_t j = 0; j < r.shape(1); j++)
                for (ssize_t k = 0; k < r.shape(2); k++)
                    r(i, j, k) += 1.0;
    }, py::arg().noconvert());

To obtain the proxy from an ``array`` object, you must specify both the data
type and number of dimensions as template arguments, such as ``auto r =
myarray.mutable_unchecked<float, 2>()``.

If the number of dimensions is not known at compile time, you can omit the
dimensions template parameter (i.e. calling ``arr_t.unchecked()`` or
``arr.unchecked<T>()``.  This will give you a proxy object that works in the
same way, but results in less optimizable code and thus a small efficiency
loss in tight loops.

Note that the returned proxy object directly references the array's data, and
only reads its shape, strides, and writeable flag when constructed.  You must
take care to ensure that the referenced array is not destroyed or reshaped for
the duration of the returned object, typically by limiting the scope of the
returned instance.

The returned proxy object supports some of the same methods as ``py::array`` so
that it can be used as a drop-in replacement for some existing, index-checked
uses of ``py::array``:

- ``r.ndim()`` returns the number of dimensions

- ``r.data(1, 2, ...)`` and ``r.mutable_data(1, 2, ...)``` returns a pointer to
  the ``const T`` or ``T`` data, respectively, at the given indices.  The
  latter is only available to proxies obtained via ``a.mutable_unchecked()``.

- ``itemsize()`` returns the size of an item in bytes, i.e. ``sizeof(T)``.

- ``ndim()`` returns the number of dimensions.

- ``shape(n)`` returns the size of dimension ``n``

- ``size()`` returns the total number of elements (i.e. the product of the shapes).

- ``nbytes()`` returns the number of bytes used by the referenced elements
  (i.e. ``itemsize()`` times ``size()``).

.. seealso::

    The file :file:`tests/test_numpy_array.cpp` contains additional examples
    demonstrating the use of this feature.

Ellipsis
========

Python 3 provides a convenient ``...`` ellipsis notation that is often used to
slice multidimensional arrays. For instance, the following snippet extracts the
middle dimensions of a tensor with the first and last index set to zero.

.. code-block:: python

   a = # a NumPy array
   b = a[0, ..., 0]

The function ``py::ellipsis()`` function can be used to perform the same
operation on the C++ side:

.. code-block:: cpp

   py::array a = /* A NumPy array */;
   py::array b = a[py::make_tuple(0, py::ellipsis(), 0)];
