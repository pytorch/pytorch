A common convention is to specify both ``VERSION`` and ``SOVERSION``
such that ``SOVERSION`` matches the first component of ``VERSION``:

.. code-block:: cmake

  set_target_properties(mylib PROPERTIES VERSION 1.2.3 SOVERSION 1)

The idea is that breaking changes to the ABI increment both the
``SOVERSION`` and the major ``VERSION`` number.
