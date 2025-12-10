macro
-----

Start recording a macro for later invocation as a command

.. code-block:: cmake

  macro(<name> [<arg1> ...])
    <commands>
  endmacro()

Defines a macro named ``<name>`` that takes arguments named
``<arg1>``, ... Commands listed after macro, but before the
matching :command:`endmacro()`, are not executed until the macro
is invoked.

Per legacy, the :command:`endmacro` command admits an optional
``<name>`` argument. If used, it must be a verbatim repeat of the
argument of the opening ``macro`` command.

See the :command:`cmake_policy()` command documentation for the behavior
of policies inside macros.

See the :ref:`Macro vs Function` section below for differences
between CMake macros and :command:`functions <function>`.

Invocation
^^^^^^^^^^

The macro invocation is case-insensitive. A macro defined as

.. code-block:: cmake

  macro(foo)
    <commands>
  endmacro()

can be invoked through any of

.. code-block:: cmake

  foo()
  Foo()
  FOO()
  cmake_language(CALL foo)

and so on. However, it is strongly recommended to stay with the
case chosen in the macro definition.  Typically macros use
all-lowercase names.

.. versionadded:: 3.18
  The :command:`cmake_language(CALL ...)` command can also be used to
  invoke the macro.

Arguments
^^^^^^^^^

When a macro is invoked, the commands recorded in the macro are
first modified by replacing formal parameters (``${arg1}``, ...)
with the arguments passed, and then invoked as normal commands.

In addition to referencing the formal parameters you can reference the
values ``${ARGC}`` which will be set to the number of arguments passed
into the macro as well as ``${ARGV0}``, ``${ARGV1}``, ``${ARGV2}``,
...  which will have the actual values of the arguments passed in.
This facilitates creating macros with optional arguments.

Furthermore, ``${ARGV}`` holds the list of all arguments given to the
macro and ``${ARGN}`` holds the list of arguments past the last expected
argument.
Referencing to ``${ARGV#}`` arguments beyond ``${ARGC}`` have undefined
behavior. Checking that ``${ARGC}`` is greater than ``#`` is the only
way to ensure that ``${ARGV#}`` was passed to the function as an extra
argument.

.. _`Macro vs Function`:

Macro vs Function
^^^^^^^^^^^^^^^^^

The ``macro`` command is very similar to the :command:`function` command.
Nonetheless, there are a few important differences.

In a function, ``ARGN``, ``ARGC``, ``ARGV`` and ``ARGV0``, ``ARGV1``, ...
are true variables in the usual CMake sense.  In a macro, they are not,
they are string replacements much like the C preprocessor would do
with a macro.  This has a number of consequences, as explained in
the :ref:`Argument Caveats` section below.

Another difference between macros and functions is the control flow.
A function is executed by transferring control from the calling
statement to the function body.  A macro is executed as if the macro
body were pasted in place of the calling statement.  This has the
consequence that a :command:`return()` in a macro body does not
just terminate execution of the macro; rather, control is returned
from the scope of the macro call.  To avoid confusion, it is recommended
to avoid :command:`return()` in macros altogether.

Unlike a function, the :variable:`CMAKE_CURRENT_FUNCTION`,
:variable:`CMAKE_CURRENT_FUNCTION_LIST_DIR`,
:variable:`CMAKE_CURRENT_FUNCTION_LIST_FILE`,
:variable:`CMAKE_CURRENT_FUNCTION_LIST_LINE` variables are not
set for a macro.

.. _`Argument Caveats`:

Argument Caveats
^^^^^^^^^^^^^^^^

Since ``ARGN``, ``ARGC``, ``ARGV``, ``ARGV0`` etc. are not variables,
you will NOT be able to use commands like

.. code-block:: cmake

 if(ARGV1) # ARGV1 is not a variable
 if(DEFINED ARGV2) # ARGV2 is not a variable
 if(ARGC GREATER 2) # ARGC is not a variable
 foreach(loop_var IN LISTS ARGN) # ARGN is not a variable

In the first case, you can use ``if(${ARGV1})``.  In the second and
third case, the proper way to check if an optional variable was
passed to the macro is to use ``if(${ARGC} GREATER 2)``.  In the
last case, you can use ``foreach(loop_var ${ARGN})`` but this will
skip empty arguments.  If you need to include them, you can use

.. code-block:: cmake

 set(list_var "${ARGN}")
 foreach(loop_var IN LISTS list_var)

Note that if you have a variable with the same name in the scope from
which the macro is called, using unreferenced names will use the
existing variable instead of the arguments. For example:

.. code-block:: cmake

 macro(bar)
   foreach(arg IN LISTS ARGN)
     <commands>
   endforeach()
 endmacro()

 function(foo)
   bar(x y z)
 endfunction()

 foo(a b c)

Will loop over ``a;b;c`` and not over ``x;y;z`` as one might have expected.
If you want true CMake variables and/or better CMake scope control you
should look at the function command.

See Also
^^^^^^^^

* :command:`cmake_parse_arguments`
* :command:`endmacro`
