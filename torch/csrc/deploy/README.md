# Torch Deploy
This is an experimental feature to embed multiple python interpreters inside the torch library,
providing a solution to the 'GIL problem' for multithreading with the convenience of python
and eager or torchscripted pytorch programs.

# libinterpreter
This is an internal library used behind the scenes to enable multiple python interpreters in
a single deploy runtime.  libinterpreter.so is DLOPENed multiple times by the deploy library.
Each copy of libinterpreter exposes a simple interpreter interface but hides its python and other
internal symbols, preventing the different python instances from seeing each other.
