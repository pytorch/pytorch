The default search order is designed to be most-specific to
least-specific for common use cases.
Projects may override the order by simply calling the command
multiple times and using the ``NO_*`` options:

.. parsed-literal::

   |FIND_XXX| (|FIND_ARGS_XXX| PATHS paths... NO_DEFAULT_PATH)
   |FIND_XXX| (|FIND_ARGS_XXX|)

Once one of the calls succeeds the result variable will be set
and stored in the cache so that no call will search again.
