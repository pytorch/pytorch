CMP0176
-------

.. versionadded:: 3.31

:command:`execute_process` ``ENCODING`` is ``UTF-8`` by default.

The ``ENCODING`` option is meaningful only on Windows.  It specifies the
character encoding expected in the process's output on stdout and stderr.
In CMake 3.14 and below the default encoding was ``NONE``, which corresponds
to CMake's internal UTF-8 encoding.  In CMake 3.15 through CMake 3.30 the
default encoding was accidentally changed to ``AUTO``, but the change went
unnoticed and was not documented.

CMake 3.31 and above prefer the ``ENCODING`` default to be ``UTF-8``.
This policy provides compatibility with projects that may have been
relying on the default being ``AUTO``.

The ``OLD`` behavior of this policy is for :command:`execute_process`
to use ``AUTO`` by default if no ``ENCODING`` is specified.  The ``NEW``
behavior for this policy is to use ``UTF-8`` as the default ``ENCODING``.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.31
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
