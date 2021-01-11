# Import mangling in `torch.package`

## Mangling rules
These are the core invariants; if you are changing mangling code please preserve them.

1. For every module imported by `PackageImporter`, two attributes are mangled:
    - `__module__`
    - `__file__`
2. No mangled names should be serialized by `PackageExporter`.
3. Every entry point to `PackageImporter` and `PackageExporter` must
   *demangle* module names before attempting to use them.
4. Internally to `PackageImporter` and `PackageExporter`, you can assume that
   all names are demangled, unless you got one by directly inspecting a module.

## Why do we mangle imported names?
To avoid accidental name collisions with modules in `sys.modules`. Consider the following:

    from torchvision.models import resnet18
    local_resnet18 = resnet18()

    # a loaded resnet18, potentially with a different implementation than the local one!
    i = torch.PackageImporter('my_resnet_18.pt')
    loaded_resnet18 = i.load_pickle('model', 'model.pkl')

    print(type(local_resnet18).__module__)  # 'torchvision.models.resnet18'
    print(type(loaded_resnet18).__module__)  # ALSO 'torchvision.models.resnet18'

These two model types have the same originating `__module__` name set.
While this isn't facially incorrect, there are a number of places in
`cpython` and elsewhere that assume you can take any module name, look it
up `sys.modules`, and get the right module back, including:
- [`import_from`](https://github.com/python/cpython/blob/5977a7989d49c3e095c7659a58267d87a17b12b1/Python/ceval.c)
- `inspect`: used in TorchScript to retrieve source code to compile
- …probably more that we don't know about.

In these cases, we may silently pick up the wrong module for `loaded_resnet18`
and e.g. TorchScript the wrong source code for our model.

## How names are mangled
On import, all modules produced by a given `PackageImporter` are given a
new top-level module as their parent. This is called the `mangle parent`. For example:

    torchvision.models.resnet18

becomes

    <torch_package_0>.torchvision.models.resnet18

The mangle parent is made unique to a given `PackageImporter` instance by
bumping a process-global `mangle_index`, i.e. `<torch__package{mangle_index}>`.

The mangle parent intentionally uses angle brackets (`<` and `>`) to make it
very unlikely that mangled names will collide with any "real" user module.

An imported module's `__file__` attribute is mangled in the same way, so:

    torchvision/modules/resnet18.py

becomes

    <torch_package_0>.torchvision/modules/resnet18.py

Similarly, the use of angle brackets makes it very unlikely that such a name
will exist in the user's file system.

## Don't serialize mangled names

Mangling happens `on import`, and the results are never saved into a package.
Assigning mangle parents on import means that we can enforce that mangle
parents are unique within the environment doing the importing. It also allows
us to avoid serializing (and maintaining backward compatibility for) this
detail.

## Demangling in PackageExporter
Occasionally `PackageExporter` may encounter mangled names during export. For
example, the user may be re-packaging an object that was imported (and thus
had its module name mangled).

This means that all entry points to `PackageExporter` must properly
demangle any module names passed to them before doing anything else. That
way, internally `PackageExporter` only ever deals with unmangled names.

There are two additional complications with demangling in PackageExporter.

First, name collisions. Consider the following user code:

    pe = PackageExporter('package.pt')
    pe.save_module('foo.bar')
    pe.save_module('__torch_package0__.foo.bar')
    pe.save_module('__torch_package1__.foo.bar')

All three packages demangle to the same `'foo.bar'` package, which leads to
confusing behavior. To guard against this, `PackageExporter` keeps track of
the demanglings it has performed and errors when a collision would have
occurred.

Second, pickled `GLOBAL` opcodes. When the pickler is writing out where to
find a class to reconstruct its state, it typically looks at
`obj.__module__` to record how to retrieve it. We need to ensure that any
global references in the pickle bytecode are always demangled, which is taken
care of in `CustomImportPickler`.
