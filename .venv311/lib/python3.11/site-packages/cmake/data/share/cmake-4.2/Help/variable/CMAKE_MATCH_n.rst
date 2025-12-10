CMAKE_MATCH_<n>
---------------

Capture group ``<n>`` matched by the last regular expression, for groups
0 through 9.  Group 0 is the entire match.  Groups 1 through 9 are the
subexpressions captured by ``()`` syntax.

When a regular expression match is used, CMake fills in ``CMAKE_MATCH_<n>``
variables with the match contents.  The :variable:`CMAKE_MATCH_COUNT`
variable holds the number of match expressions when these are filled.
