LABELS
------

.. versionadded:: 3.10

Specify a list of text labels associated with a directory and all of its
subdirectories. This is equivalent to setting the :prop_tgt:`LABELS` target
property and the :prop_test:`LABELS` test property on all targets and tests in
the current directory and subdirectories. Note: Launchers must enabled to
propagate labels to targets.

The :variable:`CMAKE_DIRECTORY_LABELS` variable can be used to initialize this
property.

The list is reported in dashboard submissions.
