# Torchscript serialization

This document explains the Torchscript serialization format, and the anatomy of a call to `torch::jit::save()` or `torch::jit::load()`.

## Overview

A serialized model (call it `model.pt`) is a ZIP archive containing many files. It can be directly inspected by calling `unzip` on it. The archive's file structure looks like:

```
model.pt
|-- model.json
|-- code/
    |-- __torch__.py
    |-- __torch__.py.debug_pkl
    |-- foo/
        |-- bar.py
        |-- bar.py.debug_pkl
|-- data.pkl
|-- tensors/
    |-- 0
    |-- 1
```

You'll notice that there are `.py` and `.pkl` files in this archive. That's because our serialization format tries to mimic Python's. All "code-like" information (methods, modules, classes, functions) are stored as human-readable `.py` containing valid Python syntax, and all "data-like" information (attributes, objects, etc.) are pickled using a subset of Python's pickle protocol.

A model is really a top-level module with some submodules, parameters, and so on depending on what the author needs. So, `data.pkl` contains the pickled top-level module. Deserializing the model is as simple as calling `unpickle()` on `data.pkl`, which will restore the module object with its associated code and data.

### Design Notes

Some things to keep in mind while working on the serialization code. These may help make technical decisions on which approach to take when making a change.

**Do what Python does**. When it comes to the serialized format, it's much simpler in the long-run to be consistent with whatever Python does. A good rule of thumb is: if I tried to interact with serialized artifacts using Python, would it work? i.e., all serialized code should be valid Python, and all pickled objects should be depickle-able by Python.

Being consistent with Python means our format is more debuggable (you can always crack it open and poke at it from Python) and leads to fewer surprises for developers familiar with Python but not familiar with Torchscript.

**Human readable**. In addition to being valid Python, serialized code should attempt to be readable Python. We should try to preserve the variable names that authors wrote, appropriately inline short expressions, and so on. This helps with debugging the serialized code.

**No jitter**. If we do:

```
m = MyModule()
m.save("foo.pt")
m_loaded = torch.load("foo.pt")
m_loaded.save("foo2.pt")
m_loaded2 = torch.load("foo2.pt")
```

We want the property that `m_loaded` and `m_loaded2` are identical. This "no-jitter" property is useful in catching bugs in the serialization process, and generally is desirable for debugging (models won't drift depending on how many times you saved/loaded them).

**Initial load should be fast**. Calling `load()` should be effectively instantaneous to a human. Anything that takes a long time (reading in tensor data, for example) should be done lazily.

## `model.json`

The `model.json` file holds metadata about the model and how it was produced. It also contains a table of tensor metadata, which stores metadata about each tensor along with a reference to a file in the `tensors/` folder that actually contains the tensor data. The full description of `model.json`'s schema can be found in `caffe2/proto/torch.proto`.

Here is small example `model.json`. **NOTE: we want to kill `model.json` and pickle tensors directly, this will happen soon.**

```
{
  "protoVersion": "6",
  "producerName": "pytorch",
  "producerVersion": "1.0",
  "tensors": [
    {
      "dims": [
        "2",
        "2"
      ],
      "offset": "0",
      "strides": [
        "2",
        "1"
      ],
      "requiresGrad": false,
      "dataType": "FLOAT",
      "data": {
        "key": "tensors/0"
      },
      "device": "cpu"
    },
    {
      "dims": [
        "2",
        "2"
      ],
      "offset": "0",
      "strides": [
        "2",
        "1"
      ],
      "requiresGrad": false,
      "dataType": "FLOAT",
      "data": {
        "key": "tensors/1"
      },
      "device": "cpu"
    },
    ...
  ]
}
```

## `code/`: How code is serialized

At a high level, code serialization means:

1. Transforming `ClassType`s and `Function`s (called "code objects") into Python source code.
2. Placing the source code in the model ZIP archive.

### Printing code objects as Python source
`PythonPrint` is the function that takes as input a `ClassType` or `Function` ("code object") and outputs Python source code.

`PythonPrint` works by walking a `Graph` (the IR representation of either a `ClassType`'s method or raw `Function`) and emitting Python code that corresponds to it. The rules for emitting Python code are mostly straightforward uninteresting. There are some extra pieces of information that `PythonPrint` tracks, however:

**Class dependencies**. While walking the graph, `PythonPrint` keeps track of what classes are used in the graph and adds them to a list of classes that the current code object depends on. For example, if we are printing a `Module`, it will depend on its submodules, as well as any classes used in its methods or attributes.

This information is used to write an `import` statement at the top of the printed source, which tells the importer that they need to go compile those dependencies (covered more in the "Code layout and qualified naming" section below).

**Uses of tensor constants**. Most constants are inlined as literals, like strings or ints. But since tensors are potentially very large, when `PythonPrint` encouters a constant tensor it will emit a reference to a global `CONSTANTS` table (like `foo = CONSTANTS.c0`). This table is the same as the general tensor table (described in the `tensors/` section below).

When importing, the importer will know how to resolve this reference into an actual tensor by looking it up in the tensor table. So `CONSTANTS.c0` means "this is the `0th` tensor in the tensor list in `model.json`."

**Original source range records**. To aid debugging, `PythonPrint` remembers the "original" (user-written) location of the source code it's emitting. That way, when the user is debugging a model they loaded, they will see diagnostics that point to the code that they actually wrote, rather than the code that `PythonPrint` emitted.

The original source range records are pickled and saved in a corresponding `.debug_pkl` file with the same name as the code. You can think of this `.debug_pkl` file as a map between source ranges in the serialized code and the original user-written code.

**Module information**. Modules are special in a few ways. First are `Parameter`s: some module attributes are actually `Parameter`s, which have special properties (see [the `torch.nn` documentation](https://pytorch.org/docs/stable/nn.html#parameters) for exact details). We track which attributes are parameters by emitting a special assignment in the class body, like:

```
class MyModule(Module):
    __parameters__ = ["foo", "bar", ]
    foo : Tensor
    bar : Tensor
    attribute_but_not_param : Tensor
```

Another special thing with modules is that they are typically constructed in Python, and we do not compile the `__init__()` method. So in order to ensure they are statically typed, `PythonPrint` must enumerate a module's attributes (as you can see above), because it can't rely on compiling `__init__()` to infer the attributes.

A final special thing is that some modules (like `nn.Sequential`) have attributes that are not valid Python identifiers. We can't write

```
# wrong!
class MyModule(Module):
    0 : ASubmodule
    1 : BSubmodule
```
because this is not valid Python syntax (even though it is legal in Python to have attributes with those names!). So we use a trick where we write directly to the `__annotations__` dict:

```
class MyModule(Module):
    __annotations__ = []
    __annotations__["0"] = ASubmodule
    __annotations__["1"] = ASubmodule
```

### Placing the source code in the archive

Once all code objects have been `PythonPrint`ed into source strings, we have to figure out where to actually put this source. Explaining this necessitates an introduction to `CompilationUnit` and `QualifiedName`.

**`CompilationUnit`**: this is the owning container for all code objects associated with a given model. When we load, we load all the code objects to a single `CompilationUnit`.

**`QualifiedName`**: this is the fully qualified name for a code object. It is similar to qualified names in Python, and looks like `"foo.bar.baz"`. Each code object has a *unique* `QualifiedName` within a `CompilationUnit`.

The exporter uses the `QualifiedName` of a code object to determine its location in the `code/` folder. The way it does so is similar to how Python does it; for example, the class `Baz` with a `QualifiedName` `"foo.bar.Baz"` will be placed in `code/foo/bar.py` under the name `Baz`.

Classes at the root of the hierarchy are given the qualified name `__torch__` as a prefix, just so that they can go in `__torch__.py`. (Why not `__main__`? Because pickle has weird special rules about things that live in `__main__`).

That's about it; there's some additional logic to make sure that within a file, we place the classes in reverse-dependency order so that we compile the "leaf" dependencies before things that depend on them.

## `data.pkl`: How data is serialized

A model is really a top-level `ScriptModule` with any number of submodules, parameters, attributes, and so on. We implement a subset of the Pickle format necessary for pickling a module object.

`pickle`'s format was chosen due to:

* **user friendliness** - the attributes file can be loaded in Python with `pickle`
* **size limits** - formats such as Protobuf empose size limits on total message size, whereas pickle limits are on individual values (e.g. strings cannot be longer than 4 GB)
* **standard format** - `pickle` is a standard Python module with a reasonably simple format. The format is a program to be consumed by a stack machine that is detailed in Python's [`pickletools.py`](https://svn.python.org/projects/python/trunk/Lib/pickletools.py)
* **built-in memoization** - for shared reference types (e.g. Tensor, string, lists, dicts)
* **self describing** - a separate definition file is not needed to understand the pickled data
* **eager mode save** - `torch.save()` already produces a `pickle` archive, so doing the same with attributes avoids introducing yet another format

All data is written into the `data.pkl` file with the exception of tensors (see "tensors" below). PyTorch functions defined in torch/jit/_pickle.py are used to mark special data types, such as this tensor table index or specialized lists.

## `tensors/`: How tensors are serialized
UNDER CONSTRUCTION/WILL CHANGE IF WE KILL TENSORS

During export a list of all the tensors in a model is created. Tensors can come from either module parameters or Tensor type attributes. Metadata about each tensor is stored in `model.json` with an index into this list. The data field refers to the file which contains the tensor storage data. Tensors are saved by directly writing the Tensor storage to a file.



