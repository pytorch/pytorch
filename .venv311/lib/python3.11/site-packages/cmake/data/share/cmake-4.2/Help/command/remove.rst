remove
------

.. deprecated:: 3.0

  Use the :command:`list(REMOVE_ITEM)` command instead.

.. code-block:: cmake

  remove(VAR VALUE VALUE ...)

Removes ``VALUE`` from the variable ``VAR``.  This is typically used to
remove entries from a vector (e.g.  semicolon separated list).  ``VALUE``
is expanded.
