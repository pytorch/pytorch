LINK_FLAGS
----------

Additional flags to use when linking this target if it is a shared library,
module library, or an executable. Static libraries need to use
:prop_tgt:`STATIC_LIBRARY_OPTIONS` or :prop_tgt:`STATIC_LIBRARY_FLAGS`
properties.

The ``LINK_FLAGS`` property, managed as a string, can be used to add extra
flags to the link step of a target.  :prop_tgt:`LINK_FLAGS_<CONFIG>` will add
to the configuration ``<CONFIG>``, for example, ``DEBUG``, ``RELEASE``,
``MINSIZEREL``, ``RELWITHDEBINFO``, ...

.. note::

  This property has been superseded by :prop_tgt:`LINK_OPTIONS` property.
