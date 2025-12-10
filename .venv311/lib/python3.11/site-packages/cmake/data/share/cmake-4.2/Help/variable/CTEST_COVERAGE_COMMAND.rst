CTEST_COVERAGE_COMMAND
----------------------

.. versionadded:: 3.1

Specify the CTest ``CoverageCommand`` setting
in a :manual:`ctest(1)` dashboard client script.

Cobertura
'''''''''

Using `Cobertura`_ as the coverage generation within your multi-module
Java project can generate a series of XML files.

The Cobertura Coverage parser expects to read the coverage data from a
single XML file which contains the coverage data for all modules.
Cobertura has a program with the ability to merge given ``cobertura.ser`` files
and then another program to generate a combined XML file from the previous
merged file.  For command line testing, this can be done by hand prior to
CTest looking for the coverage files. For script builds,
set the ``CTEST_COVERAGE_COMMAND`` variable to point to a file which will
perform these same steps, such as a ``.sh`` or ``.bat`` file.

.. code-block:: cmake

  set(CTEST_COVERAGE_COMMAND .../run-coverage-and-consolidate.sh)

where the ``run-coverage-and-consolidate.sh`` script is perhaps created by
the :command:`configure_file` command and might contain the following code:

.. code-block:: bash

  #!/usr/bin/env bash
  CoberturaFiles="$(find "/path/to/source" -name "cobertura.ser")"
  SourceDirs="$(find "/path/to/source" -name "java" -type d)"
  cobertura-merge --datafile coberturamerge.ser $CoberturaFiles
  cobertura-report --datafile coberturamerge.ser --destination . \
                   --format xml $SourceDirs

The script uses ``find`` to capture the paths to all of the ``cobertura.ser``
files found below the project's source directory.  It keeps the list of files
and supplies it as an argument to the ``cobertura-merge`` program. The
``--datafile`` argument signifies where the result of the merge will be kept.

The combined ``coberturamerge.ser`` file is then used to generate the XML report
using the ``cobertura-report`` program.  The call to the cobertura-report
program requires some named arguments.

``--datafila``
  path to the merged ``.ser`` file

``--destination``
  path to put the output files(s)

``--format``
  file format to write output in: xml or html

The rest of the supplied arguments consist of the full paths to the
``/src/main/java`` directories of each module within the source tree. These
directories are needed and should not be forgotten.

.. _`Cobertura`: https://cobertura.github.io/cobertura/
