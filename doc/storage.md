<a name="torch.Storage.dok"/>
# Storage #
<a name="torch.CharStorage.dok"/>
<a name="torch.ByteStorage.dok"/>
<a name="torch.IntStorage.dok"/>
<a name="torch.ShortStorage.dok"/>
<a name="torch.FloatStorage.dok"/>
<a name="torch.LongStorage.dok"/>
<a name="torch.DoubleStorage.dok"/>

_Storages_ are basically a way for `Lua` to access memory of a `C` pointer
or array. _Storages_ can also [map the contents of a file to memory](#__torch.StorageMap).
A `Storage` is an array of _basic_ `C` types. For arrays of `Torch` objects,
use the `Lua` tables.

Several `Storage` classes for all the basic `C` types exist and have the
following self-explanatory names: `ByteStorage`, `CharStorage`, `ShortStorage`,
`IntStorage`, `LongStorage`, `FloatStorage`, `DoubleStorage`.

Note that `ByteStorage` and `CharStorage` represent both arrays of bytes. `ByteStorage` represents an array of
_unsigned_ chars, while `CharStorage` represents an array of _signed_ chars.

Conversions between two `Storage` type might be done using `copy`:
```lua
x = torch.IntStorage(10):fill(1)
y = torch.DoubleStorage(10):copy(x)
```

[Classical storages](#torch.Storage) are [serializable](file.md#torch.File.serialization).
[Storages mapping a file](#__torch.StorageMap) are also [serializable](#FileSerialization),
but _will be saved as a normal storage_.

An alias `torch.Storage()` is made over your preferred Storage type,
controlled by the
[torch.setdefaulttensortype](utility.md#torch.setdefaulttensortype)
function. By default, this "points" on `torch.DoubleStorage`.

## Constructors and Access Methods ##

<a name="torch.Storage"/>
### torch.TYPEStorage([size]) ###

Returns a new `Storage` of type `TYPE`. Valid `TYPE` are `Byte`, `Char`, `Short`,
`Int`, `Long`, `Float`, and `Double`. If `size` is given, resize the
`Storage` accordingly, else create an empty `Storage`.

Example:
```lua
-- Creates a Storage of 10 double:
x = torch.DoubleStorage(10)
```

The data in the `Storage` is _uninitialized_.

<a name="torch.Storage"/>
### torch.TYPEStorage(table) ###

The argument is assumed to be a Lua array of numbers. The constructor returns a new storage of the specified 'TYPE', 
of the size of the table, containing all the table elements converted

Example:
```lua
> = torch.IntStorage({1,2,3,4})

 1
 2
 3
 4
[torch.IntStorage of size 4]
```

<a name="torch.Storage"/>
### torch.TYPEStorage(filename [, shared]) ###
<a name="__torch.StorageMap"/>

Returns a new kind of `Storage` which maps the contents of the given
`filename` to memory. Valid `TYPE` are `Byte`, `Char`, `Short`, `Int`, `Long`,
`Float`, and `Double`. If the optional boolean argument `shared` is `true`,
the mapped memory is shared amongst all processes on the computer.

When `shared` is `true`, the file must be accessible in read-write mode. Any
changes on the storage will be written in the file. The changes might be written
only after destruction of the storage.

When `shared` is `false` (or not provided), the file must be at least
readable. Any changes on the storage will not affect the file. Note:
changes made on the file after creation of the storage have an unspecified
effect on the storage contents.

The [size](#torch.Storage.size) of the returned `Storage` will be
```lua
(size of file in byte)/(size of TYPE).
```

Example:
```lua
$ echo "Hello World" > hello.txt
$ lua
Lua 5.1.3  Copyright (C) 1994-2008 Lua.org, PUC-Rio
> require 'torch'
> x = torch.CharStorage('hello.txt')
> = x
  72
 101
 108
 108
 111
  32
  87
 111
 114
 108
 100
  10
[torch.CharStorage of size 12]

> = x:string()
Hello World

> = x:fill(42):string()
____________
> 
$ cat hello.txt 
Hello World
$ lua
Lua 5.1.3  Copyright (C) 1994-2008 Lua.org, PUC-Rio
> require 'torch'
> x = torch.CharStorage('hello.txt', true)
> = x:string()
Hello World

> x:fill(42)
>
$ cat hello.txt 
____________
```

<a name="__torch.StorageSharp"/>
### [number] #self ###

Returns the number of elements in the storage. Equivalent to [size()](#torch.Storage.size).

<a name="torch.Storage.__index__"/>
### [number] self[index] ###

Returns or set the element at position `index` in the storage. Valid range
of `index` is 1 to [size()](#torch.Storage.size).

Example:
```lua
x = torch.DoubleStorage(10)
print(x[5])
```

<a name="torch.Storage.copy"/>
### [self] copy(storage) ###

Copy another `storage`. The types of the two storages might be different: in that case
a conversion of types occur (which might result, of course, in loss of precision or rounding).
This method returns self, allowing things like:
```lua
x = torch.IntStorage(10):fill(1)
y = torch.DoubleStorage(10):copy(x) -- y won't be nil!
```

<a name="torch.Storage.fill"/>
### [self] fill(value) ###

Fill the `Storage` with the given value. This method returns self, allowing things like:
```lua
x = torch.IntStorage(10):fill(0) -- x won't be nil!
```

<a name="torch.Storage.resize"/>
### [self] resize(size) ###

Resize the storage to the provide `size`. _The new contents are undertermined_.

This function returns self, allowing things like:
```lua
x = torch.DoubleStorage(10):fill(1)
y = torch.DoubleStorage():resize(x:size()):copy(x) -- y won't be nil!
```

<a name="torch.Storage.size"/>
### [number] size() ###

Returns the number of elements in the storage. Equivalent to [#](#__torch.StorageSharp).

<a name="torch.Storage.string"/>
### [self] string(str) ###

This function is available only on `ByteStorage` and `CharStorage`.

This method resizes the storage to the length of the provided
string `str`, and copy the contents of `str` into the storage. The `NULL` terminating character is not copied,
but `str` might contain `NULL` characters. The method returns the `Storage`.
```lua
> x = torch.CharStorage():string("blah blah")
> print(x)
  98
 108
  97
 104
  32
  98
 108
  97
 104
[torch.CharStorage of size 9]
```

<a name="torch.Storage.string"/>
### [string] string() ###

This function is available only on `ByteStorage` and `CharStorage`.

The contents of the storage viewed as a string are returned. The string might contain
`NULL` characters.
```lua
> x = torch.CharStorage():string("blah blah")
> print(x:string())
blah blah
```

