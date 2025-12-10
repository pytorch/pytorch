CMAKE_<LANG>_LINKER_WRAPPER_FLAG
--------------------------------

.. versionadded:: 3.13

Defines the syntax of compiler driver option to pass options to the linker
tool. It will be used to translate the ``LINKER:`` prefix in the link options
(see :command:`add_link_options` and :command:`target_link_options`).

This variable holds a :ref:`semicolon-separated list <CMake Language Lists>` of tokens.
If a space (i.e. " ") is specified as last token, flag and ``LINKER:``
arguments will be specified as separate arguments to the compiler driver.
The :variable:`CMAKE_<LANG>_LINKER_WRAPPER_FLAG_SEP` variable can be specified
to manage concatenation of arguments.

For example, for ``Clang`` we have:

.. code-block:: cmake

  set (CMAKE_C_LINKER_WRAPPER_FLAG "-Xlinker" " ")

Specifying ``"LINKER:-z,defs"`` will be transformed in
``-Xlinker -z -Xlinker defs``.

For ``GNU GCC``:

.. code-block:: cmake

  set (CMAKE_C_LINKER_WRAPPER_FLAG "-Wl,")
  set (CMAKE_C_LINKER_WRAPPER_FLAG_SEP ",")

Specifying ``"LINKER:-z,defs"`` will be transformed in ``-Wl,-z,defs``.

And for ``SunPro``:

.. code-block:: cmake

  set (CMAKE_C_LINKER_WRAPPER_FLAG "-Qoption" "ld" " ")
  set (CMAKE_C_LINKER_WRAPPER_FLAG_SEP ",")

Specifying ``"LINKER:-z,defs"`` will be transformed in ``-Qoption ld -z,defs``.
