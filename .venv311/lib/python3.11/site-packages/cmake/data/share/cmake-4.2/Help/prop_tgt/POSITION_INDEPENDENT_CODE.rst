POSITION_INDEPENDENT_CODE
-------------------------

A target property that specifies whether to create a target that has
position-independent code enabled.

The ``POSITION_INDEPENDENT_CODE`` target property determines whether
position-independent executables or libraries will be created.  This
property is ``True`` by default for ``SHARED`` and ``MODULE`` library
targets.  For other targets, this property is initialized by the value
of the :variable:`CMAKE_POSITION_INDEPENDENT_CODE` variable if it is set
when the target is created, or ``False`` otherwise.

.. note::

  For executable targets, the link step is controlled by the :policy:`CMP0083`
  policy and the :module:`CheckPIESupported` module.

Position-independent code (PIC) refers to machine code that executes
properly regardless of its absolute memory address.  This is particularly
important for shared libraries, which are often loaded at different memory
addresses by different programs.  Generating position-independent code
ensures that these libraries can be safely and efficiently shared among
multiple processes without causing address conflicts.  On some platforms
(notably UNIX-like systems), generating PIC is also a requirement for
creating shared libraries.

Use of position-independent code is recommended or required in the following
cases:

* When building shared or module libraries (e.g., with
  ``add_library(... SHARED)``, or ``add_library(... MODULE)``), where PIC
  allows dynamic relocation at runtime.

* When building executables as position-independent executables (PIE), which
  can enhance security by enabling Address Space Layout Randomization (ASLR).

* On platforms or toolchains that require PIC for certain types of linking
  or sandboxed environments.

Enabling PIC can result in slightly larger or slower code on some
architectures, but this is often outweighed by the benefits of flexibility
and security.

Examples
^^^^^^^^

Enabling PIC for a static library target:

.. code-block:: cmake

  add_library(foo STATIC foo.c)
  set_target_properties(foo PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

Enabling PIC for an executable target:

.. code-block:: cmake

  add_executable(app app.c)

  set_target_properties(app PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

  # Additionally, pass PIE-related link-time options to executable(s).
  include(CheckPIESupported)
  check_pie_supported()

See Also
^^^^^^^^

* The :module:`CheckPIESupported` module to pass PIE-related options to the
  linker for executables.
