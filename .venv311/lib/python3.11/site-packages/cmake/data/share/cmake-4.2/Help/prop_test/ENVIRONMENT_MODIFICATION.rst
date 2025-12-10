ENVIRONMENT_MODIFICATION
------------------------

.. versionadded:: 3.22

Specify environment variables that should be modified for running a test. Note
that the operations performed by this property are performed after the
:prop_test:`ENVIRONMENT` property is already applied.

Set to a :ref:`semicolon-separated list <CMake Language Lists>` of
environment variables and values of the form ``MYVAR=OP:VALUE``,
where ``MYVAR`` is the case-sensitive name of an environment variable
to be modified.  Entries are considered in the order specified in the
property's value.  The ``OP`` may be one of:

 .. include:: ../include/ENVIRONMENT_MODIFICATION_OPS.rst

Unrecognized ``OP`` values will result in the test failing before it is
executed. This is so that future operations may be added without changing
valid behavior of existing tests.

The environment changes from this property do not affect other tests.
