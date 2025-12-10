RESOURCE_GROUPS
---------------

.. versionadded:: 3.16

Specify resources required by a test, grouped in a way that is meaningful to
the test.  See :ref:`resource allocation <ctest-resource-allocation>`
for more information on how this property integrates into the CTest resource
allocation feature.

The ``RESOURCE_GROUPS`` property is a :ref:`semicolon-separated list <CMake
Language Lists>` of group descriptions. Each entry consists of an optional
number of groups using the description followed by a series of resource
requirements for those groups. These requirements (and the number of groups)
are separated by commas. The resource requirements consist of the name of a
resource type, followed by a colon, followed by an unsigned integer
specifying the number of slots required on one resource of the given type.

The ``RESOURCE_GROUPS`` property tells CTest what resources a test expects
to use grouped in a way meaningful to the test.  The test itself must read
the :ref:`environment variables <ctest-resource-environment-variables>` to
determine which resources have been allocated to each group.  For example,
each group may correspond to a process the test will spawn when executed.

Consider the following example:

.. code-block:: cmake

  add_test(NAME MyTest COMMAND MyExe)
  set_property(TEST MyTest PROPERTY RESOURCE_GROUPS
    "2,gpus:2"
    "gpus:4,crypto_chips:2")

In this example, there are two group descriptions (implicitly separated by a
semicolon.) The content of the first description is ``2,gpus:2``. This
description specifies 2 groups, each of which requires 2 slots from a single
GPU. The content of the second description is ``gpus:4,crypto_chips:2``. This
description does not specify a group count, so a default of 1 is assumed.
This single group requires 4 slots from a single GPU and 2 slots from a
single cryptography chip. In total, 3 resource groups are specified for this
test, each with its own unique requirements.

Note that the number of slots following the resource type specifies slots from
a *single* instance of the resource. If the resource group can tolerate
receiving slots from different instances of the same resource, it can indicate
this by splitting the specification into multiple requirements of one slot. For
example:

.. code-block:: cmake

  add_test(NAME MyTest COMMAND MyExe)
  set_property(TEST MyTest PROPERTY RESOURCE_GROUPS
    "gpus:1,gpus:1,gpus:1,gpus:1")

In this case, the single resource group indicates that it needs four GPU slots,
all of which may come from separate GPUs (though they don't have to; CTest may
still assign slots from the same GPU.)

When CTest sets the :ref:`environment variables
<ctest-resource-environment-variables>` for a test, it assigns a group number
based on the group description, starting at 0 on the left and the number of
groups minus 1 on the right. For example, in the example above, the two
groups in the first description would have IDs of 0 and 1, and the single
group in the second description would have an ID of 2.

Both the ``RESOURCE_GROUPS`` and :prop_test:`RESOURCE_LOCK` properties serve
similar purposes, but they are distinct and orthogonal. Resources specified by
``RESOURCE_GROUPS`` do not affect :prop_test:`RESOURCE_LOCK`, and vice versa.
Whereas :prop_test:`RESOURCE_LOCK` is a simpler property that is used for
locking one global resource, ``RESOURCE_GROUPS`` is a more advanced property
that allows multiple tests to simultaneously use multiple resources of the
same type, specifying their requirements in a fine-grained manner.
