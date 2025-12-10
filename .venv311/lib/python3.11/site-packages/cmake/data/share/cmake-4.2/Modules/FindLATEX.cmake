# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLATEX
---------

Finds LaTeX compiler and Latex-related software like BibTeX:

.. code-block:: cmake

  find_package(LATEX [...])

LaTeX is a typesetting system for the production of technical and scientific
documentation.

Components
^^^^^^^^^^

.. versionadded:: 3.2

Components can be optionally specified using a standard CMake syntax:

.. code-block:: cmake

  find_package(LATEX [COMPONENTS <component>...])

Supported components are:

``PDFLATEX``
  Finds the PdfLaTeX compiler.

``XELATEX``
  Finds the XeLaTeX compiler.

``LUALATEX``
  Finds the LuaLaTeX compiler.

``BIBTEX``
  Finds the BibTeX compiler.

``BIBER``
  Finds the Biber compiler.

``MAKEINDEX``
  Finds the MakeIndex compiler.

``XINDY``
  Finds the xindy compiler.

``DVIPS``
  Finds the DVI-to-PostScript (DVIPS) converter.

``DVIPDF``
  Finds the DVIPDF converter.

``PS2PDF``
  Finds the the PS2PDF converter.

``PDFTOPS``
  Finds the PDF-to-PostScript converter.

``LATEX2HTML``
  Finds the converter for converting LaTeX documents to HTML.

``HTLATEX``
  Finds htlatex compiler.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LATEX_FOUND``
  Boolean indicating whether the LaTex compiler and all its required components
  were found.

``LATEX_<component>_FOUND``
  Boolean indicating whether the LaTeX ``<component>`` was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LATEX_COMPILER``
  The path to the LaTeX compiler.

``PDFLATEX_COMPILER``
  The path to the PdfLaTeX compiler.

``XELATEX_COMPILER``
  .. versionadded:: 3.2

  The path to the XeLaTeX compiler.

``LUALATEX_COMPILER``
  .. versionadded:: 3.2

  The path to the LuaLaTeX compiler.

``BIBTEX_COMPILER``
  The path to the BibTeX compiler.

``BIBER_COMPILER``
  .. versionadded:: 3.2

  The path to the Biber compiler.

``MAKEINDEX_COMPILER``
  The path to the MakeIndex compiler.

``XINDY_COMPILER``
  .. versionadded:: 3.2

  The path to the xindy compiler.

``DVIPS_CONVERTER``
  The path to the DVIPS converter.

``DVIPDF_CONVERTER``
  The path to the DVIPDF converter.

``PS2PDF_CONVERTER``
  The path to the PS2PDF converter.

``PDFTOPS_CONVERTER``
  .. versionadded:: 3.2

  The path to the pdftops converter.

``LATEX2HTML_CONVERTER``
  The path to the LaTeX2Html converter.

``HTLATEX_COMPILER``
  .. versionadded:: 3.2

  The path to the htlatex compiler.

Examples
^^^^^^^^

Finding LaTeX in a project:

.. code-block:: cmake

  find_package(LATEX)

Finding LaTeX compiler and specifying which additional LaTeX components are
required for LaTeX to be considered found:

.. code-block:: cmake

  find_package(LATEX COMPONENTS PDFLATEX)

  if(LATEX_FOUND)
    execute_process(COMMAND ${LATEX_COMPILER} ...)
    execute_process(COMMAND ${PDFLATEX_COMPILER} ...)
  endif()

Or finding LaTeX compiler and specifying multiple components:

.. code-block:: cmake

  find_package(LATEX COMPONENTS BIBTEX PS2PDF)

  if(LATEXT_FOUND)
    # ...
  endif()
#]=======================================================================]

if (WIN32)
  # Try to find the MikTex binary path (look for its package manager).
  find_path(MIKTEX_BINARY_PATH mpm.exe
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MiK\\MiKTeX\\CurrentVersion\\MiKTeX;Install Root]/miktex/bin"
     "$ENV{LOCALAPPDATA}/Programs/MiKTeX/miktex/bin"
     "$ENV{LOCALAPPDATA}/Programs/MiKTeX/miktex/bin/x64"
     "$ENV{APPDATA}/Programs/MiKTeX/miktex/bin"
     "$ENV{APPDATA}/Programs/MiKTeX/miktex/bin/x64"
    DOC
    "Path to the MikTex binary directory."
  )
  mark_as_advanced(MIKTEX_BINARY_PATH)

  # Try to find the GhostScript binary path (look for gswin32).
  get_filename_component(GHOSTSCRIPT_BINARY_PATH_FROM_REGISTRY_8_00
     "[HKEY_LOCAL_MACHINE\\SOFTWARE\\AFPL Ghostscript\\8.00;GS_DLL]" PATH
  )

  get_filename_component(GHOSTSCRIPT_BINARY_PATH_FROM_REGISTRY_7_04
     "[HKEY_LOCAL_MACHINE\\SOFTWARE\\AFPL Ghostscript\\7.04;GS_DLL]" PATH
  )

  find_path(GHOSTSCRIPT_BINARY_PATH gswin32.exe
    ${GHOSTSCRIPT_BINARY_PATH_FROM_REGISTRY_8_00}
    ${GHOSTSCRIPT_BINARY_PATH_FROM_REGISTRY_7_04}
    DOC "Path to the GhostScript binary directory."
  )
  mark_as_advanced(GHOSTSCRIPT_BINARY_PATH)

  find_path(GHOSTSCRIPT_LIBRARY_PATH ps2pdf13.bat
    "${GHOSTSCRIPT_BINARY_PATH}/../lib"
    DOC "Path to the GhostScript library directory."
  )
  mark_as_advanced(GHOSTSCRIPT_LIBRARY_PATH)
endif ()

# try to find Latex and the related programs
find_program(LATEX_COMPILER
  NAMES latex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)

# find pdflatex
find_program(PDFLATEX_COMPILER
  NAMES pdflatex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (PDFLATEX_COMPILER)
  set(LATEX_PDFLATEX_FOUND TRUE)
else()
  set(LATEX_PDFLATEX_FOUND FALSE)
endif()

# find xelatex
find_program(XELATEX_COMPILER
  NAMES xelatex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (XELATEX_COMPILER)
  set(LATEX_XELATEX_FOUND TRUE)
else()
  set(LATEX_XELATEX_FOUND FALSE)
endif()

# find lualatex
find_program(LUALATEX_COMPILER
  NAMES lualatex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (LUALATEX_COMPILER)
  set(LATEX_LUALATEX_FOUND TRUE)
else()
  set(LATEX_LUALATEX_FOUND FALSE)
endif()

# find bibtex
find_program(BIBTEX_COMPILER
  NAMES bibtex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (BIBTEX_COMPILER)
  set(LATEX_BIBTEX_FOUND TRUE)
else()
  set(LATEX_BIBTEX_FOUND FALSE)
endif()

# find biber
find_program(BIBER_COMPILER
  NAMES biber
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (BIBER_COMPILER)
  set(LATEX_BIBER_FOUND TRUE)
else()
  set(LATEX_BIBER_FOUND FALSE)
endif()

# find makeindex
find_program(MAKEINDEX_COMPILER
  NAMES makeindex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (MAKEINDEX_COMPILER)
  set(LATEX_MAKEINDEX_FOUND TRUE)
else()
  set(LATEX_MAKEINDEX_FOUND FALSE)
endif()

# find xindy
find_program(XINDY_COMPILER
  NAMES xindy
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (XINDY_COMPILER)
  set(LATEX_XINDY_FOUND TRUE)
else()
  set(LATEX_XINDY_FOUND FALSE)
endif()

# find dvips
find_program(DVIPS_CONVERTER
  NAMES dvips
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (DVIPS_CONVERTER)
  set(LATEX_DVIPS_FOUND TRUE)
else()
  set(LATEX_DVIPS_FOUND FALSE)
endif()

# find dvipdf
find_program(DVIPDF_CONVERTER
  NAMES dvipdfm dvipdft dvipdf
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (DVIPDF_CONVERTER)
  set(LATEX_DVIPDF_FOUND TRUE)
else()
  set(LATEX_DVIPDF_FOUND FALSE)
endif()

# find ps2pdf
if (WIN32)
  find_program(PS2PDF_CONVERTER
    NAMES ps2pdf14.bat ps2pdf14 ps2pdf
    PATHS ${GHOSTSCRIPT_LIBRARY_PATH}
          ${MIKTEX_BINARY_PATH}
  )
else ()
  find_program(PS2PDF_CONVERTER
    NAMES ps2pdf14 ps2pdf
  )
endif ()
if (PS2PDF_CONVERTER)
  set(LATEX_PS2PDF_FOUND TRUE)
else()
  set(LATEX_PS2PDF_FOUND FALSE)
endif()

# find pdftops
find_program(PDFTOPS_CONVERTER
  NAMES pdftops
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (PDFTOPS_CONVERTER)
  set(LATEX_PDFTOPS_FOUND TRUE)
else()
  set(LATEX_PDFTOPS_FOUND FALSE)
endif()

# find latex2html
find_program(LATEX2HTML_CONVERTER
  NAMES latex2html
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (LATEX2HTML_CONVERTER)
  set(LATEX_LATEX2HTML_FOUND TRUE)
else()
  set(LATEX_LATEX2HTML_FOUND FALSE)
endif()

# find htlatex
find_program(HTLATEX_COMPILER
  NAMES htlatex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (HTLATEX_COMPILER)
  set(LATEX_HTLATEX_FOUND TRUE)
else()
  set(LATEX_HTLATEX_FOUND FALSE)
endif()


mark_as_advanced(
  LATEX_COMPILER
  PDFLATEX_COMPILER
  XELATEX_COMPILER
  LUALATEX_COMPILER
  BIBTEX_COMPILER
  BIBER_COMPILER
  MAKEINDEX_COMPILER
  XINDY_COMPILER
  DVIPS_CONVERTER
  DVIPDF_CONVERTER
  PS2PDF_CONVERTER
  PDFTOPS_CONVERTER
  LATEX2HTML_CONVERTER
  HTLATEX_COMPILER
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LATEX
  REQUIRED_VARS LATEX_COMPILER
  HANDLE_COMPONENTS
)
