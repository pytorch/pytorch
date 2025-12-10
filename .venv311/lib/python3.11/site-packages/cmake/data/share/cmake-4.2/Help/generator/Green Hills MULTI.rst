Green Hills MULTI
-----------------

.. versionadded:: 3.3

.. versionadded:: 3.15
  Linux support.

Generates Green Hills MULTI project files (experimental, work-in-progress).

  The buildsystem has predetermined build-configuration settings that can be controlled
  via the :variable:`CMAKE_BUILD_TYPE` variable.

Platform Selection
^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.13

The variable ``GHS_PRIMARY_TARGET`` can be used to select the target platform.

  | Sets ``primaryTarget`` entry in project file.

For example:

* ``cmake -G "Green Hills MULTI" -D GHS_PRIMARY_TARGET=ppc_integrity.tgt``

Otherwise the ``primaryTarget`` will be composed from the values of :variable:`CMAKE_GENERATOR_PLATFORM`
and ``GHS_TARGET_PLATFORM``. Defaulting to the value of ``arm_integrity.tgt``

* The :variable:`CMAKE_GENERATOR_PLATFORM` variable may be set, perhaps
  via the :option:`cmake -A` option.

  | Typical values of ``arm``, ``ppc``, ``86``, etcetera, are used.

* The variable ``GHS_TARGET_PLATFORM`` may be set, perhaps via the :option:`cmake -D`
  option.

  | Defaults to ``integrity``.
  | Usual values are ``integrity``, ``threadx``, ``uvelosity``, ``velosity``,
    ``vxworks``, ``standalone``.

For example:

* ``cmake -G "Green Hills MULTI"`` for ``arm_integrity.tgt``.
* ``cmake -G "Green Hills MULTI" -A 86`` for ``86_integrity.tgt``.
* ``cmake -G "Green Hills MULTI" -D GHS_TARGET_PLATFORM=standalone`` for ``arm_standalone.tgt``.
* ``cmake -G "Green Hills MULTI" -A ppc -D GHS_TARGET_PLATFORM=standalone`` for ``ppc_standalone.tgt``.

Toolset Selection
^^^^^^^^^^^^^^^^^

.. versionadded:: 3.13

The generator searches for the latest compiler or can be given a location to use.
``GHS_TOOLSET_ROOT`` is the directory that is checked for the latest compiler.

* The :variable:`CMAKE_GENERATOR_TOOLSET` option may be set, perhaps
  via the :option:`cmake -T` option, to specify the location of the toolset.
  Both absolute and relative paths are valid. Paths are relative to ``GHS_TOOLSET_ROOT``.

* The variable ``GHS_TOOLSET_ROOT`` may be set, perhaps via the :option:`cmake -D`
  option.

  | Root path for toolset searches and relative paths.
  | Defaults to ``C:/ghs`` in Windows or ``/usr/ghs`` in Linux.

For example, setting a specific compiler:

* ``cmake -G "Green Hills MULTI" -T comp_201754`` for ``/usr/ghs/comp_201754``.
* ``cmake -G "Green Hills MULTI" -T comp_201754 -D GHS_TOOLSET_ROOT=/opt/ghs`` for ``/opt/ghs/comp_201754``.
* ``cmake -G "Green Hills MULTI" -T /usr/ghs/comp_201554``
* ``cmake -G "Green Hills MULTI" -T C:/ghs/comp_201754``

For example, searching for latest compiler:

* ``cmake -G "Green Hills MULTI"`` for searching ``/usr/ghs``.
* ``cmake -G "Green Hills MULTI -D GHS_TOOLSET_ROOT=/opt/ghs"`` for searching ``/opt/ghs``.

.. note::
  The :variable:`CMAKE_GENERATOR_TOOLSET` should use CMake style paths.

OS and BSP Selection
^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.3

Certain target platforms, like Integrity, require an OS.  The RTOS directory path
can be explicitly set using ``GHS_OS_DIR``.  Otherwise ``GHS_OS_ROOT`` will be
searched for the latest Integrity RTOS.

If the target platform, like Integrity, requires a BSP name then it can be set via
the ``GHS_BSP_NAME`` variable.

* ``GHS_OS_DIR`` and ``GHS_OS_DIR_OPTION``

  | Sets ``-os_dir`` entry in project file.

  | ``GHS_OS_DIR_OPTION`` default value is ``-os_dir``.

  .. versionadded:: 3.15
    The ``GHS_OS_DIR_OPTION`` variable.

  For example:

  * ``cmake -G "Green Hills MULTI" -D GHS_OS_DIR=/usr/ghs/int1144``

* ``GHS_OS_ROOT``

  | Root path for RTOS searches.
  | Defaults to ``C:/ghs`` in Windows or ``/usr/ghs`` in Linux.

  For example:

  * ``cmake -G "Green Hills MULTI" -D GHS_OS_ROOT=/opt/ghs``

* ``GHS_BSP_NAME``

  | Sets ``-bsp`` entry in project file.
  | Defaults to ``sim<arch>`` for ``integrity`` platforms.

  For example:

  * ``cmake -G "Green Hills MULTI"`` for ``simarm`` on ``arm_integrity.tgt``.
  * ``cmake -G "Green Hills MULTI" -A 86`` for ``sim86`` on ``86_integrity.tgt``.
  * ``cmake -G "Green Hills MULTI" -A ppc -D GHS_BSP_NAME=sim800`` for ``sim800``
    on ``ppc_integrity.tgt``.
  * ``cmake -G "Green Hills MULTI" -D GHS_PRIMARY_TARGET=ppc_integrity.tgt -D GHS_BSP_NAME=fsl-t1040``
    for ``fsl-t1040`` on ``ppc_integrity.tgt``.

Target Properties
^^^^^^^^^^^^^^^^^

.. versionadded:: 3.14

The following properties are available:

* :prop_tgt:`GHS_INTEGRITY_APP`
* :prop_tgt:`GHS_NO_SOURCE_GROUP_FILE`

MULTI Project Variables
^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.3

Adding a Customization file and macros are available through the use of the following
variables:

* ``GHS_CUSTOMIZATION`` - CMake path name to Customization File.
* ``GHS_GPJ_MACROS`` - CMake list of Macros.

.. note::
  This generator is deemed experimental as of CMake |release|
  and is still a work in progress.  Future versions of CMake
  may make breaking changes as the generator matures.
