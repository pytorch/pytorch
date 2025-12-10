The value of |INTERFACE_PROPERTY_LINK_DIRECT| may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

.. note::

  The |INTERFACE_PROPERTY_LINK_DIRECT| target property is intended for
  advanced use cases such as injection of static plugins into a consuming
  executable.  It should not be used as a substitute for organizing
  normal calls to :command:`target_link_libraries`.
