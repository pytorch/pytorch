qt_wrap_ui
----------

.. deprecated:: 3.14

  This command was originally added to support Qt 3 before the
  :command:`add_custom_command()` command was sufficiently mature.  The
  :module:`FindQt4` module provides the ``qt4_wrap_ui()`` macro, which
  should be used instead for Qt 4 projects.  For projects using Qt 5 or
  later, use the equivalent macro provided by Qt itself (e.g. Qt 5 provides
  ``qt5_wrap_ui()``).

Manually create Qt user interfaces Wrappers.

.. code-block:: cmake

  qt_wrap_ui(resultingLibraryName HeadersDestName
             SourcesDestName SourceLists ...)

Produces .h and .cxx files for all the .ui files listed in the
``SourceLists``.  The .h files will be added to the library using the
``HeadersDestNamesource`` list.  The .cxx files will be added to the
library using the ``SourcesDestNamesource`` list.

Consider updating the project to use the :prop_tgt:`AUTOUIC` target property
instead for a more automated way of invoking the ``uic`` tool.
