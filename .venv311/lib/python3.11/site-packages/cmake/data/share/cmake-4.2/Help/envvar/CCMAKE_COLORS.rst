CCMAKE_COLORS
-------------

.. versionadded:: 3.18

Determines what colors are used by the CMake curses interface,
when run on a terminal that supports colors.
The syntax follows the same conventions as ``LS_COLORS``;
that is, a list of key/value pairs separated by ``:``.

Keys are a single letter corresponding to a CMake cache variable type:

- ``s``: A ``STRING``.
- ``p``: A ``FILEPATH``.
- ``c``: A value which has an associated list of choices.
- ``y``: A ``BOOL`` which has a true-like value (e.g. ``ON``, ``YES``).
- ``n``: A ``BOOL`` which has a false-like value (e.g. ``OFF``, ``NO``).

Values are an integer number that specifies what color to use.
``0`` is black (you probably don't want to use that).
Others are determined by your terminal's color support.
Most (color) terminals will support at least 8 or 16 colors.
Some will support up to 256 colors. The colors will likely match
`this chart <https://upload.wikimedia.org/wikipedia/commons/1/15/Xterm_256color_chart.svg>`_,
although the first 16 colors may match the original
`CGA color palette <https://en.wikipedia.org/wiki/Color_Graphics_Adapter#Color_palette>`_.
(Many modern terminal emulators also allow their color palette,
at least for the first 16 colors, to be configured by the user.)

Note that fairly minimal checking is done for bad colors
(although a value higher than what curses believes your terminal supports
will be silently ignored) or bad syntax.

For example::

  CCMAKE_COLORS='s=39:p=220:c=207:n=196:y=46'
