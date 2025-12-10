CPACK_SET_DESTDIR
-----------------

Boolean toggle to make CPack use ``DESTDIR`` mechanism when packaging.

``DESTDIR`` means DESTination DIRectory.  It is commonly used by makefile
users in order to install software at non-default location.  It is a
basic relocation mechanism that should not be used on Windows (see
:variable:`CMAKE_INSTALL_PREFIX` documentation).  It is usually invoked like
this:

.. code-block:: sh

 make DESTDIR=/home/john install

which will install the concerned software using the installation
prefix, e.g. ``/usr/local`` prepended with the ``DESTDIR`` value which
finally gives ``/home/john/usr/local``.  When preparing a package, CPack
first installs the items to be packaged in a local (to the build tree)
directory by using the same ``DESTDIR`` mechanism.  Nevertheless, if
``CPACK_SET_DESTDIR`` is set then CPack will set ``DESTDIR`` before doing the
local install.  The most noticeable difference is that without
``CPACK_SET_DESTDIR``, CPack uses :variable:`CPACK_PACKAGING_INSTALL_PREFIX`
as a prefix whereas with ``CPACK_SET_DESTDIR`` set, CPack will use
:variable:`CMAKE_INSTALL_PREFIX` as a prefix.

Manually setting ``CPACK_SET_DESTDIR`` may help (or simply be necessary)
if some install rules uses absolute ``DESTINATION`` (see CMake
:command:`install` command).  However, starting with CPack/CMake 2.8.3 RPM
and DEB installers tries to handle ``DESTDIR`` automatically so that it is
seldom necessary for the user to set it.
