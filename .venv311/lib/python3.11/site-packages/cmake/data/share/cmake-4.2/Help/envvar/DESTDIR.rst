DESTDIR
-------

.. include:: include/ENV_VAR.rst

On UNIX one can use the ``DESTDIR`` mechanism in order to relocate the
whole installation.  ``DESTDIR`` means DESTination DIRectory.  It is
commonly used by packagers to install software in a staging directory.

For example, running

.. code-block:: shell

  make DESTDIR=/package/stage install

will install the software using the installation prefix, e.g. ``/usr/local``,
prepended with the ``DESTDIR`` value which gives ``/package/stage/usr/local``.
The packaging tool may then construct the package from the content of the
``/package/stage`` directory.

See the :variable:`CMAKE_INSTALL_PREFIX` variable to control the
installation prefix when configuring a build tree.  Or, when using
the :manual:`cmake(1)` command-line tool's :option:`--install <cmake --install>`
mode, one may specify a different prefix using the
:option:`--prefix <cmake--install --prefix>` option.

.. note::

  ``DESTDIR`` may not be used on Windows because installation
  prefix usually contains a drive letter like in ``C:/Program Files``
  which cannot be prepended with some other prefix.
