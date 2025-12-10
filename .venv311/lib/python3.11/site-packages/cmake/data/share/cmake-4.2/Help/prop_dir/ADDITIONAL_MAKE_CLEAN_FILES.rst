ADDITIONAL_MAKE_CLEAN_FILES
---------------------------

.. deprecated:: 3.15

  Use :prop_dir:`ADDITIONAL_CLEAN_FILES` instead.

Additional files to remove during the clean stage.

A :ref:`;-list <CMake Language Lists>` of files that will be removed as a
part of the ``make clean`` target.

Arguments to ``ADDITIONAL_MAKE_CLEAN_FILES`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

This property only works for the Makefile generators.
It is ignored on other generators.
