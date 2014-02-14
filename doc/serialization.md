
<a name="torch.serialization.dok"/>
# Serialization #

Torch provides 4 high-level methods to serialize/deserialize arbitrary Lua/Torch objects.
These functions are just abstractions over the [File](#torch.File) object, and were created
for convenience (these are very common routines).

The first two functions are useful to serialize/deserialize data to/from files:

  - `torch.save(filename, object [, format])`
  - `[object] torch.load(filename [, format])`

The next two functions are useful to serialize/deserialize data to/from strings:

  - `[str] torch.serialize(object)`
  - `[object] torch.deserialize(str)`

Serializing to files is useful to save arbitrary data structures, or share them with other people.
Serializing to strings is useful to store arbitrary data structures in databases, or 3rd party
software.

<a name="torch.save"/>
### torch.save(filename, object [, format]) ###

Writes `object` into a file named `filename`. The `format` can be set
to `ascii` or `binary` (default is binary). Binary format is platform
dependent, but typically more compact and faster to read/write. The ASCII
format is platform-independent, and should be used to share data structures
across platforms.

```
-- arbitrary object:
obj = {
   mat = torch.randn(10,10),
   name = '10',
   test = {
      entry = 1
   }
}

-- save to disk:
torch.save('test.dat', obj)
```

<a name="torch.load"/>
### [object] torch.load(filename [, format]) ###

Reads `object` from a file named `filename`. The `format` can be set
to `ascii` or `binary` (default is binary). Binary format is platform
dependent, but typically more compact and faster to read/write. The ASCII
format is platform-independent, and should be used to share data structures
across platforms.

```
-- given serialized object from section above, reload:
obj = torch.load('test.dat')

print(obj)
-- will print:
-- {[mat]  = DoubleTensor - size: 10x10
--  [name] = string : "10"
--  [test] = table - size: 0}
```

<a name="torch.serialize"/>
### [str] torch.serialize(object) ###

Serializes `object` into a string.

```
-- arbitrary object:
obj = {
   mat = torch.randn(10,10),
   name = '10',
   test = {
      entry = 1
   }
}

-- serialize:
str = torch.serialize(obj)
```

<a name="torch.deserialize"/>
### [object] torch.deserialize(str) ###

Deserializes `object` from a string.

```
-- given serialized object from section above, deserialize:
obj = torch.deserialize(str)

print(obj)
-- will print:
-- {[mat]  = DoubleTensor - size: 10x10
--  [name] = string : "10"
--  [test] = table - size: 0}
```

