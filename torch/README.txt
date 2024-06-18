Note [TH abstraction violation]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TH/THC provide some hpp headers, which are proper C++ headers rather than
C headers.  These headers serve double duty as *internal implementation
detail* headers, whose contents should largely not be used by external
clients.

Ideally, we would not install these headers at all; instead, you should
use public functions (in headers like `THTensor.h`, NOT `THTensor.hpp`)
to manipulate these structs.  However, there are a few places
in torch/csrc where we violate this abstraction.  They are marked with
a pointer to this note.  Each of those sites will have to be refactored
when we refactor the guts of THTensor and related structures.
