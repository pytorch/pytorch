COMPATIBLE_INTERFACE_STRING
---------------------------

Properties which must be string-compatible with their link interface

The ``COMPATIBLE_INTERFACE_STRING`` property may contain a list of
properties for this target which must be the same when evaluated as a
string in the ``INTERFACE`` variant of the property all linked dependees.
For example, if a property ``FOO`` appears in the list, then for each
dependee, the ``INTERFACE_FOO`` property content in all of its
dependencies must be equal with each other, and with the ``FOO`` property
in the depender.  If the property is not set, then it is ignored.

Note that for each dependee, the set of properties specified in this
property must not intersect with the set specified in any of the other
:ref:`Compatible Interface Properties`.
