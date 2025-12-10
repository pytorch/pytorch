.. |file| replace:: The output is printed to a named ``<file>`` if given.

.. option:: -version [<file>], --version [<file>], /V [<file>]

 Show program name/version banner and exit.
 |file|

.. option:: -h, -H, --help, -help, -usage, /?

 Print usage information and exit.

 Usage describes the basic command line interface and its options.

.. option:: --help <keyword> [<file>]

 Print help for one CMake keyword.

 ``<keyword>`` can be a property, variable, command, policy, generator
 or module.

 The relevant manual entry for ``<keyword>`` is
 printed in a human-readable text format.
 |file|

 .. versionchanged:: 3.28

   Prior to CMake 3.28, this option supported command names only.

.. option:: --help-full [<file>]

 Print all help manuals and exit.

 All manuals are printed in a human-readable text format.
 |file|

.. option:: --help-manual <man> [<file>]

 Print one help manual and exit.

 The specified manual is printed in a human-readable text format.
 |file|

.. option:: --help-manual-list [<file>]

 List help manuals available and exit.

 The list contains all manuals for which help may be obtained by
 using the ``--help-manual`` option followed by a manual name.
 |file|

.. option:: --help-command <cmd> [<file>]

 Print help for one command and exit.

 The :manual:`cmake-commands(7)` manual entry for ``<cmd>`` is
 printed in a human-readable text format.
 |file|

.. option:: --help-command-list [<file>]

 List commands with help available and exit.

 The list contains all commands for which help may be obtained by
 using the ``--help-command`` option followed by a command name.
 |file|

.. option:: --help-commands [<file>]

 Print cmake-commands manual and exit.

 The :manual:`cmake-commands(7)` manual is printed in a
 human-readable text format.
 |file|

.. option:: --help-module <mod> [<file>]

 Print help for one module and exit.

 The :manual:`cmake-modules(7)` manual entry for ``<mod>`` is printed
 in a human-readable text format.
 |file|

.. option:: --help-module-list [<file>]

 List modules with help available and exit.

 The list contains all modules for which help may be obtained by
 using the ``--help-module`` option followed by a module name.
 |file|

.. option:: --help-modules [<file>]

 Print cmake-modules manual and exit.

 The :manual:`cmake-modules(7)` manual is printed in a human-readable
 text format.
 |file|

.. option:: --help-policy <cmp> [<file>]

 Print help for one policy and exit.

 The :manual:`cmake-policies(7)` manual entry for ``<cmp>`` is
 printed in a human-readable text format.
 |file|

.. option:: --help-policy-list [<file>]

 List policies with help available and exit.

 The list contains all policies for which help may be obtained by
 using the ``--help-policy`` option followed by a policy name.
 |file|

.. option:: --help-policies [<file>]

 Print cmake-policies manual and exit.

 The :manual:`cmake-policies(7)` manual is printed in a
 human-readable text format.
 |file|

.. option:: --help-property <prop> [<file>]

 Print help for one property and exit.

 The :manual:`cmake-properties(7)` manual entries for ``<prop>`` are
 printed in a human-readable text format.
 |file|

.. option:: --help-property-list [<file>]

 List properties with help available and exit.

 The list contains all properties for which help may be obtained by
 using the ``--help-property`` option followed by a property name.
 |file|

.. option:: --help-properties [<file>]

 Print cmake-properties manual and exit.

 The :manual:`cmake-properties(7)` manual is printed in a
 human-readable text format.
 |file|

.. option:: --help-variable <var> [<file>]

 Print help for one variable and exit.

 The :manual:`cmake-variables(7)` manual entry for ``<var>`` is
 printed in a human-readable text format.
 |file|

.. option:: --help-variable-list [<file>]

 List variables with help available and exit.

 The list contains all variables for which help may be obtained by
 using the ``--help-variable`` option followed by a variable name.
 |file|

.. option:: --help-variables [<file>]

 Print cmake-variables manual and exit.

 The :manual:`cmake-variables(7)` manual is printed in a
 human-readable text format.
 |file|
