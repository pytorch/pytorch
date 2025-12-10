COMPATIBLE_INTERFACE_NUMBER_MAX
-------------------------------

Properties whose maximum value from the link interface will be used.

The ``COMPATIBLE_INTERFACE_NUMBER_MAX`` property may contain a list of
properties for this target whose maximum value may be read at generate
time when evaluated in the ``INTERFACE`` variant of the property in all
linked dependees.  For example, if a property ``FOO`` appears in the list,
then for each dependee, the ``INTERFACE_FOO`` property content in all of
its dependencies will be compared with each other and with the ``FOO``
property in the depender.  When reading the ``FOO`` property at generate
time, the maximum value will be returned. If the property is not set,
then it is ignored.

Note that for each dependee, the set of properties specified in this
property must not intersect with the set specified in any of the other
:ref:`Compatible Interface Properties`.
