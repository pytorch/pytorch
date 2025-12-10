.. note::

  When evaluating :ref:`Variable References` of the form ``${VAR}``, CMake
  first searches for a normal variable with that name.  If no such normal
  variable exists, CMake will then search for a cache entry with that name.
  Because of this, **unsetting a normal variable can expose a cache variable
  that was previously hidden**.  To force a variable reference of the form
  ``${VAR}`` to return an empty string, use ``set(<variable> "")``, which
  clears the normal variable but leaves it defined.
