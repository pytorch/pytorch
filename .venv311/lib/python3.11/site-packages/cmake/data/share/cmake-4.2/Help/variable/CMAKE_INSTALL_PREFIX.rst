CMAKE_INSTALL_PREFIX
--------------------

Install directory used by :command:`install`.

If ``make install`` is invoked or ``INSTALL`` is built, this directory is
prepended onto all install directories.

This variable defaults as follows:

* .. versionadded:: 3.29

    If the :envvar:`CMAKE_INSTALL_PREFIX` environment variable is set,
    its value is used as default for this variable.

* ``c:/Program Files/${PROJECT_NAME}`` on Windows.

* ``/usr/local`` on UNIX platforms.

See :variable:`CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT` for how a
project might choose its own default.

On UNIX one can use the ``DESTDIR`` mechanism in order to relocate the
whole installation to a staging area.  See the :envvar:`DESTDIR` environment
variable for more information.

The installation prefix is also added to :variable:`CMAKE_SYSTEM_PREFIX_PATH`
so that :command:`find_package`, :command:`find_program`,
:command:`find_library`, :command:`find_path`, and :command:`find_file`
will search the prefix for other software. This behavior can be disabled by
setting the :variable:`CMAKE_FIND_NO_INSTALL_PREFIX` to ``TRUE`` before the
first :command:`project` invocation.

.. note::

  Use the :module:`GNUInstallDirs` module to provide GNU-style
  options for the layout of directories within the installation.

The ``CMAKE_INSTALL_PREFIX`` may be defined when configuring a build tree
to set its installation prefix.  Or, when using the :manual:`cmake(1)`
command-line tool's :option:`--install <cmake --install>` mode, one may specify
a different prefix using the :option:`--prefix <cmake--install --prefix>`
option:

.. code-block:: shell

  cmake --install . --prefix /my/install/prefix
