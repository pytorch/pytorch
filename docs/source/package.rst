.. currentmodule:: torch.package

torch.package
=============

.. warning::

    This module is experimental and has not yet been publicly released.

``torch.package`` adds support for creating hermetic packages containing arbitrary
PyTorch code. These packages can be saved, shared, used to load and execute models
at a later date or on a different machine, and can even be deployed to production using
``torch::deploy``.

This document contains tutorials, how-to guides, explanations, and an API reference that
will help you learn more about ``torch.package`` and how to use it.

Tutorials
---------
Packaging your first model
^^^^^^^^^^^^^^^^^^^^^^^^^^
A tutorial that guides you through packaging and unpackaging a simple model is available
`on Colab <https://colab.research.google.com/drive/1dWATcDir22kgRQqBg2X_Lsh5UPfC7UTK?usp=sharing>`_.
After completing this exercise, you will be familiar with the basic API for creating and using
Torch packages.

How do I...
-----------
See what is inside a package?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Treat the package like a ZIP archive
""""""""""""""""""""""""""""""""""""
The container format for a ``torch.package`` is ZIP, so any tools that work with standard ZIP files should
work for exploring the contents. Some common ways to interact with ZIP files:

* ``unzip my_package.pt`` will unzip the ``torch.package`` archive to disk, where you can freely inspect its contents.


::

    $ unzip my_package.pt && tree my_package
    my_package
    ├── .data
    │   ├── 94304870911616.storage
    │   ├── 94304900784016.storage
    │   ├── extern_modules
    │   └── version
    ├── models
    │   └── model_1.pkl
    └── torchvision
        └── models
            ├── resnet.py
            └── utils.py
    ~ cd my_package && cat torchvision/models/resnet.py
    ...


* The Python ``zipfile`` module provides a standard way to read and write ZIP archive contents.


::

    from zipfile import ZipFile
    with ZipFile("my_package.pt") as myzip:
        file_bytes = myzip.read("torchvision/models/resnet.py")
        # edit file_bytes in some way
        myzip.writestr("torchvision/models/resnet.py", new_file_bytes)


* vim has the ability to natively read ZIP archives. You can even edit files and :``write`` them back into the archive!


::

    # add this to your .vimrc to treat `*.pt` files as zip files
    au BufReadCmd *.pt call zip#Browse(expand("<amatch>"))

    ~ vi my_package.pt


Use the ``file_structure()`` API
""""""""""""""""""""""""""""""""
:class:`PackageImporter` and :class:`PackageExporter` provide a ``file_structure()`` method, which will return a printable
and queryable ``Folder`` object. The ``Folder`` object is a simple directory structure that you can use to explore the
current contents of a ``torch.package``.

The ``Folder`` object itself is directly printable and will print out a file tree representation. To filter what is returned,
use the glob-style ``include`` and ``exclude`` filtering arguments.


::

    with PackageExporter('my_package.pt', verbose=False) as pe:
        pe.save_pickle('models', 'model_1.pkl', mod)
        # can limit printed items with include/exclude args
        print(pe.file_structure(include=["**/utils.py", "**/*.pkl"], exclude="**/*.storages"))

    importer = PackageImporter('my_package.pt')
    print(importer.file_structure()) # will print out all files


Output:


::

    # filtered with glob pattern:
    #    include=["**/utils.py", "**/*.pkl"], exclude="**/*.storages"
    ─── my_package.pt
        ├── models
        │   └── model_1.pkl
        └── torchvision
            └── models
                └── utils.py

    # all files
    ─── my_package.pt
        ├── .data
        │   ├── 94304870911616.storage
        │   ├── 94304900784016.storage
        │   ├── extern_modules
        │   └── version
        ├── models
        │   └── model_1.pkl
        └── torchvision
            └── models
                ├── resnet.py
                └── utils.py


You can also query ``Folder`` objects with the ``has_file()`` method.


::

    exporter_file_structure = exporter.file_structure()
    found: bool = exporter_file_structure.has_file("package_a/subpackage.py")


Include arbitrary resources with my package and access them later?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`PackageExporter` exposes three methods, ``save_pickle``, ``save_text`` and ``save_binary`` that allow you to save
Python objects, text, and binary data to a package.


::

    with torch.PackageExporter("package.pt") as exporter:
        # Pickles the object and saves to `my_resources/tens.pkl` in the archive.
        exporter.save_pickle("my_resources", "tensor.pkl", torch.randn(4))
        exporter.save_text("config_stuff", "words.txt", "a sample string")
        exporter.save_binary("raw_data", "binary", my_bytes)


:class:`PackageImporter` exposes complementary methods named ``load_pickle``, ``load_text`` and ``load_binary`` that allow you to load
Python objects, text and binary data from a package.


::

    importer = torch.PackageImporter("package.pt")
    my_tensor = importer.load_pickle("my_resources", "tensor.pkl")
    text = importer.load_text("config_stuff", "words.txt")
    binary = importer.load_binary("raw_data", "binary")


Customize how a class is packaged?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``torch.package`` allows for the customization of how classes are packaged. This behavior is accessed through defining the method
``__reduce_package__`` on a class and by defining a corresponding de-packaging function. This is similar to defining ``__reduce__`` for
Python’s normal pickling process.

In the below example, class Foo defines  ``__reduce_package__`` to customize how Foo instances are saved into a ``torch.package``.
``__reduce_package__`` is called by a :class:`PackageExporter` during the pickling process and is passed the :class:`PackageExporter` itself
which the save_pickle() call was invoked from. Foo’s ``__reduce_package__`` implementation does the work needed to save the Foo instance
into the package. ``__reduce_package__`` returns a tuple containing the function ``unpackage_foo`` along with the arguments call to ``unpackage_foo``
with. ``unpackage_foo`` is a de-packaging function that does the necessary work to reconstruct and return Foo objects from what was saved into the
``torch.package``. The de-packaging function, in this case ``unpackage_foo``, is called by the loading :class:`PackageImporter` during ``load_pickle()``
invocations. The arguments passed to the de-packaging function include the calling :class:`PackageImporter` followed by the arguments in the
``__reduce_package__``’s return tuple. Note: the arguments to the de-packaging function must themselves not need ``persistent_id`` to be pickled.


::

    # foo.py [Example of customizing how class Foo is packaged]
    from torch.package import PackageExporter, PackageImporter
    import time


    class Foo:
        def __init__(self, my_string: str):
            super().__init__()
            self.my_string = my_string
            self.time_imported = 0
            self.time_exported = 0

        def __reduce_package__(self, exporter: PackageExporter):
            """
            Called by ``torch.package.PackageExporter``'s Pickler's ``persistent_id`` when
            saving an instance of this object. This method should do the work to save this
            object inside of the ``torch.package`` archive.

            Returns function w/ arguments to load the object from a
            ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function.
            """

            # use this pattern to ensure no naming conflicts with normal dependencies,
            # anything saved under this module name shouldn't conflict with other
            # items in the package
            generated_module_name = f"foo-generated._{exporter.get_unique_id()}"
            exporter.save_text(
                generated_module_name,
                "foo.txt",
                self.my_string + ", with exporter modification!",
            )
            time_exported = time.clock_gettime(1)

            # returns de-packaging function w/ arguments to invoke with
            return (unpackage_foo, (generated_module_name, time_exported,))


    def unpackage_foo(
        importer: PackageImporter, generated_module_name: str, time_exported: float
    ) -> Foo:
        """
        Called by ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function
        when depickling a Foo object.
        Performs work of loading and returning a Foo instance from a ``torch.package`` archive.
        """
        time_imported = time.clock_gettime(1)
        foo = Foo(importer.load_text(generated_module_name, "foo.txt"))
        foo.time_imported = time_imported
        foo.time_exported = time_exported
        return foo


::

    # example of saving instances of class Foo

    import torch
    from torch.package import PackageImporter, PackageExporter
    import foo

    foo_1 = foo.Foo("foo_1 initial string")
    foo_2 = foo.Foo("foo_2 initial string")
    with PackageExporter('foo_package.pt', verbose=False) as pe:
        # save as normal, no extra work necessary
        pe.save_pickle('foo_collection', 'foo1.pkl', foo_1)
        pe.save_pickle('foo_collection', 'foo2.pkl', foo_2)
        print(pe.file_structure())

    pi = PackageImporter('foo_package.pt')
    imported_foo = pi.load_pickle('foo_collection', 'foo1.pkl')
    print(f"foo_1 string: '{imported_foo.my_string}'")
    print(f"foo_1 export time: {imported_foo.time_exported}")
    print(f"foo_1 import time: {imported_foo.time_imported}")


::

    # output of running above script
    ─── foo_package
        ├── foo-generated
        │   ├── _0
        │   │   └── foo.txt
        │   └── _1
        │       └── foo.txt
        ├── foo_collection
        │   ├── foo1.pkl
        │   └── foo2.pkl
        └── foo.py

    foo_1 string: 'foo_1 initial string, with reduction modification!'
    foo_1 export time: 9857706.650140837
    foo_1 import time: 9857706.652698385


Test in my source code whether or not it is executing inside a package?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A :class:`PackageImporter` will add the attribute ``__torch_package__`` to every module that it initializes. Your code can check for the
presence of this attribute to determine whether it is executing in a packaged context or not.


::

    # In foo/bar.py:

    if "__torch_package__" in dir():  # true if the code is being loaded from a package
        def is_in_package():
            return True

        UserException = Exception
    else:
        def is_in_package():
            return False

        UserException = UnpackageableException


Now, the code will behave differently depending on whether it’s imported normally through your Python environment or imported from a
``torch.package``.


::

    from foo.bar import is_in_package

    print(is_in_package())  # False

    loaded_module = PackageImporter(my_pacakge).import_module("foo.bar")
    loaded_module.is_in_package()  # True


*Warning*: in general, it’s bad practice to have code that behaves differently depending on whether it’s packaged or not. This can lead to
hard-to-debug issues that are sensitive to how you imported your code. If your package is intended to be heavily used, consider restructuring
your code so that it behaves the same way no matter how it was loaded.


Patch code into a package?
^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`PackageExporter` offers a ``save_source_string()`` method that allows one to save arbitrary Python source code to a module of your choosing.


::

    with PackageExporter(f) as exporter:
        # Save the my_module.foo available in your current Python environment.
        exporter.save_module("my_module.foo")

        # This saves the provided string to my_module/foo.py in the package archive.
        # It will override the my_module.foo that was previously saved.
        exporter.save_source_string("my_module.foo", textwrap.dedent(
            """\
            def my_function():
                print('hello world')
            """
        ))

        # If you want to treat my_module.bar as a package
        # (e.g. save to `my_module/bar/__init__.py` instead of `my_module/bar.py)
        # pass is_package=True,
        exporter.save_source_string("my_module.bar",
                                    "def foo(): print('hello')\n",
                                    is_package=True)

    importer = PackageImporter(f)
    importer.import_module("my_module.foo").my_function()  # prints 'hello world'


Access package contents from packaged code?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`PackageImporter` implements the
`importlib.resources <https://docs.python.org/3/library/importlib.html#module-importlib.resources>`_
API for accessing resources from inside a package.


::

    with PackageExporter(f) as exporter:
        # saves text to one/a.txt in the archive
        exporter.save_text("my_resource", "a.txt", "hello world!")
        # saves the tensor to my_pickle/obj.pkl
        exporter.save_pickle("my_pickle", "obj.pkl", torch.ones(2, 2))

        # see below for module contents
        exporter.save_module("foo")
        exporter.save_module("bar")


The ``importlib.resources`` API allows access to resources from within packaged code.


::

    # foo.py:
    import importlib.resources
    import my_resource

    # returns "hello world!"
    def get_my_resource():
        return importlib.resources.read_text(my_resource, "a.txt")


Using ``importlib.resources`` is the recommended way to access package contents from within packaged code, since it complies
with the Python standard. However, it is also possible to access the parent :class:`PackageImporter` instance itself from within
packaged code.


::

    # bar.py:
    import torch_package_importer # this is the PackageImporter that imported this module.

    # Prints "hello world!", equivalient to importlib.resources.read_text
    def get_my_resource():
        return torch_package_importer.load_text("my_resource", "a.txt")

    # You also do things that the importlib.resources API does not support, like loading
    # a pickled object from the package.
    def get_my_pickle():
        return torch_package_importer.load_pickle("my_pickle", "obj.pkl")


Distinguish between packaged code and non-packaged code?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To tell if an object’s code is from a ``torch.package``, use the ``torch.package.is_from_package()`` function.
Note: if an object is from a package but its definition is from a module marked ``extern`` or from ``stdlib``,
this check will return ``False``.


::

    importer = PackageImporter(f)
    mod = importer.import_module('foo')
    obj = importer.load_pickle('model', 'model.pkl')
    txt = importer.load_text('text', 'my_test.txt')

    assert is_from_package(mod)
    assert is_from_package(obj)
    assert not is_from_package(txt) # str is from stdlib, so this will return False


Fix mocked object being used in ``Pickle`` error?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Diagnose and resolve unpicklable objects?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Re-export an imported object?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To re-export an object that was previously imported by a :class:`PackageImporter`, you must make the new :class:`PackageExporter`
aware of the original :class:`PackageImporter` so that it can find source code for your object’s dependencies.


::

    importer = PackageImporter(f)
    obj = importer.load_pickle("model", "model.pkl")

    # re-export obj in a new package
    with PackageExporter(f2, importer=(importer, sys_importer)) as exporter:
        exporter.save_pickle("model", "model.pkl", obj)


Package a TorchScript module?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Customize how dependencies are packaged?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Understand what dependencies my code has?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


API Reference
-------------
.. autoclass:: torch.package.PackagingError

.. autoclass:: torch.package.EmptyMatchError

.. autoclass:: torch.package.PackageExporter
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.PackageImporter
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.Directory
  :members:
