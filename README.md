<a name="torch.reference.dok"/>
# Torch Package Reference Manual #

__Torch__ is the main package in [Torch7](http://torch.ch) where data
structures for multi-dimensional tensors and mathematical operations
over these are defined. Additionally, it provides many utilities for
accessing files, serializing objects of arbitrary types and other
useful utilities.

<a name="torch.reference.dok"/>
## Torch Packages ##

  * Tensor Library
    * [Tensor](doc/tensor.md) defines the _all powerful_ tensor object that provides multi-dimensional numerical arrays with type templating.
    * [Mathematical operations](doc/maths.md) that are defined for the tensor object types.
    * [Storage](doc/storage.md) defines a simple storage interface that controls the underlying storage for any tensor object.
  * File I/O Interface Library
    * [File](doc/file.md) is an abstract interface for common file operations.
    * [Disk File](doc/diskfile.md) defines operations on files stored on disk.
    * [Memory File](doc/memoryfile.md) defines operations on stored in RAM.
    * [Pipe File](doc/pipefile.md) defines operations for using piped commands.
    * [High-Level File operations](doc/serialization.md) defines higher-level serialization functions.
  * Useful Utilities
    * [Timer](doc/timer.md) provides functionality for _measuring time_.
    * [Tester](doc/tester.md) is a generic tester framework.
    * [CmdLine](doc/cmdline.md) is a command line argument parsing utility.
    * [Random](doc/random.md) defines a random number generator package with various distributions.
    * Finally useful [utility](doc/utility.md) functions are provided for easy handling of torch tensor types and class inheritance.

