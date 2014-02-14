<a name="torch.File.dok"/>
# File #

This is an _abstract_ class. It defines most methods implemented by its
child classes, like [DiskFile](diskfile.md),
[MemoryFile](memoryfile.md) and [PipeFile](pipefile.md).

Methods defined here are intended for basic read/write functionalities.
Read/write methods might write in [ASCII](#torch.File.ascii) mode or
[binary](#torch.File.binary) mode.
 
In [ASCII](#torch.File.ascii) mode, numbers are converted in human readable
format (characters). Booleans are converted into `0` (false) or `1` (true).
In [binary](#torch.File.binary) mode, numbers and boolean are directly encoded
as represented in a register of the computer. While not being human
readable and less portable, the binary mode is obviously faster.

In [ASCII](#torch.File.ascii) mode, if the default option
[autoSpacing()](#torch.File.autoSpacing) is chosen, a space will be generated
after each written number or boolean. A carriage return will also be added
after each call to a write method. With this option, the spaces are
supposed to exist while reading. This option can be deactivated with
[noAutoSpacing()](#torch.File.noAutoSpacing).

A `Lua` error might or might be not generated in case of read/write error
or problem in the file. This depends on the choice made between
[quiet()](#torch.File.quiet) and [pedantic()](#torch.File.pedantic) options. It
is possible to query if an error occured in the last operation by calling
[hasError()](#torch.File.hasError).

<a name="torch.File.read"/>
## Read methods ##
<a name="torch.File.readByte"/>
<a name="torch.File.readBool"/>
<a name="torch.File.readShort"/>
<a name="torch.File.readChar"/>
<a name="torch.File.readLong"/>
<a name="torch.File.readInt"/>
<a name="torch.File.readDouble"/>
<a name="torch.File.readFloat"/>

They are three types of reading methods:
  - `[number] readTYPE()`
  - `[TYPEStorage] readTYPE(n)`
  - `[number] readTYPE(TYPEStorage)`

where `TYPE` can be either `Byte`, `Char`, `Short`, `Int`, `Long`, `Float` or `Double`.

A convenience method also exist for boolean types: `[boolean] readBool()`. It reads
a value on the file with `readInt()` and returns `true` if and only if this value is `1`. It is not possible
to read storages of booleans.

All these methods depends on the encoding choice: [ASCII](#torch.File.ascii)
or [binary](#torch.File.binary) mode.  In [ASCII](#torch.File.ascii) mode, the
option [autoSpacing()](#torch.File.autoSpacing) and
[noAutoSpacing()](#torch.File.noAutoSpacing) have also an effect on these
methods.

If no parameter is given, one element is returned. This element is
converted to a `Lua` number when reading.

If `n` is given, `n` values of the specified type are read
and returned in a new [Storage](storage.md) of that particular type.
The storage size corresponds to the number of elements actually read.

If a `Storage` is given, the method will attempt to read a number of elements
equals to the size of the given storage, and fill up the storage with these elements.
The number of elements actually read is returned.

In case of read error, these methods will call the `Lua` error function using the default
[pedantic](#torch.File.pedantic) option, or stay quiet with the [quiet](#torch.File.quiet)
option. In the latter case, one can check if an error occurred with
[hasError()](#torch.File.hasError).

<a name="torch.File.write"/>
## Write methods ##
<a name="torch.File.writeByte"/>
<a name="torch.File.writeBool"/>
<a name="torch.File.writeShort"/>
<a name="torch.File.writeChar"/>
<a name="torch.File.writeLong"/>
<a name="torch.File.writeInt"/>
<a name="torch.File.writeDouble"/>
<a name="torch.File.writeFloat"/>

They are two types of reading methods:
  - `[number] writeTYPE(number)`
  - `[number] writeTYPE(TYPEStorage)`

where `TYPE` can be either `Byte`, `Char`, `Short`, `Int`, `Long`, `Float` or `Double`.

A convenience method also exist for boolean types: `writeBool(value)`. If `value` is `nil` or
not `true` a it is equivalent to a `writeInt(0)` call, else to `writeInt(1)`. It is not possible
to write storages of booleans.

All these methods depends on the encoding choice: [ASCII](#torch.File.ascii)
or [binary](#torch.File.ascii) mode.  In [ASCII](#torch.File.ascii) mode, the
option [autoSpacing()](#torch.File.autoSpacing) and
[noAutoSpacing()](#torch.File.noAutoSpacing) have also an effect on these
methods.

If one `Lua` number is given, this number is converted according to the
name of the method when writing (e.g. `writeInt(3.14)` will write `3`).

If a `Storage` is given, the method will attempt to write all the elements contained
in the storage.

These methods return the number of elements actually written.

In case of read error, these methods will call the `Lua` error function using the default
[pedantic](#torch.File.pedantic) option, or stay quiet with the [quiet](#torch.File.quiet)
option. In the latter case, one can check if an error occurred with
[hasError()](#torch.File.hasError).

<a name="torch.File.serialization"/>
## Serialization methods ##

These methods allow the user to save any serializable objects on disk and
reload it later in its original state. In other words, it can perform a
_deep_ copy of an object into a given `File`.

Serializable objects are `Torch` objects having a `read()` and
`write()` method. `Lua` objects such as `table`, `number` or
`string` or _pure Lua_ functions are also serializable.

If the object to save contains several other objects (let say it is a tree
of objects), then objects appearing several times in this tree will be
_saved only once_. This saves disk space, speedup loading/saving and
respect the dependencies between objects.

Interestingly, if the `File` is a [MemoryFile](memoryfile.md), it allows
the user to easily make a _clone_ of any serializable object:
```lua
file = torch.MemoryFile() -- creates a file in memory
file:writeObject(object) -- writes the object into file
file:seek(1) -- comes back at the beginning of the file
objectClone = file:readObject() -- gets a clone of object
```

<a name="torch.File.readObject"/>
### readObject() ###

Returns the next [serializable](#torch.File.serialization) object saved beforehand
in the file with [writeObject()](#torch.File.writeObject).

Note that objects which were [written](#torch.File.writeObject) with the same
reference have still the same reference after loading.

Example:
```lua
-- creates an array which contains twice the same tensor  
array = {}
x = torch.Tensor(1)
table.insert(array, x)
table.insert(array, x)

-- array[1] and array[2] refer to the same address
-- x[1] == array[1][1] == array[2][1] == 3.14
array[1][1] = 3.14

-- write the array on disk
file = torch.DiskFile('foo.asc', 'w')
file:writeObject(array)
file:close() -- make sure the data is written

-- reload the array
file = torch.DiskFile('foo.asc', 'r')
arrayNew = file:readObject()

-- arrayNew[1] and arrayNew[2] refer to the same address!
-- arrayNew[1][1] == arrayNew[2][1] == 3.14
-- so if we do now:
arrayNew[1][1] = 2.72
-- arrayNew[1][1] == arrayNew[2][1] == 2.72 !
```

<a name="torch.File.writeObject"/>
### writeObject(object) ###

Writes `object` into the file. This object can be read later using
[readObject()](#torch.File.readObject). Serializable objects are `Torch`
objects having a `read()` and `write()` method. `Lua` objects such as
`table`, `number` or `string` or pure Lua functions are also serializable.

If the object has been already written in the file, only a _reference_ to
this already saved object will be written: this saves space an speed-up
writing; it also allows to keep the dependencies between objects intact.

In returns, if one writes an object, modify its member, and write the
object again in the same file, the modifications will not be recorded
in the file, as only a reference to the original will be written. See
[readObject()](#torch.File.readObject) for an example.

<a name="torch.File.readString"/>
### [string] readString(format) ###

If `format` starts with ''"*l"` then returns the next line in the `File''. The end-of-line character is skipped.

If `format` starts with ''"*a"` then returns all the remaining contents of the `File''.

If no data is available, then an error is raised, except if `File` is in [quiet()](#torch.File.quiet) mode where
it then returns `nil`.

Because Torch is more precised on number typing, the `Lua` format ''"*n"'' is not supported:
instead use one of the [number read methods](#torch.File.read).

<a name="torch.File.writeString"/>
### [number] writeString(str) ###

Writes the string `str` in the `File`. If the string cannot be written completely an error is raised, except
if `File` is in [quiet()](#torch.File.quiet) mode where it returns the number of character actually written.

## General Access and Control Methods ##

<a name="torch.File.ascii"/>
### ascii() [default] ###

The data read or written will be in `ASCII` mode: all numbers are converted
to characters (human readable format) and boolean are converted to `0`
(false) or `1` (true). The input-output format in this mode depends on the
options [autoSpacing()](#torch.File.autoSpacing) and
[noAutoSpacing()](#torch.File.noAutoSpacing).

<a name="torch.File.autoSpacing"/>
### autoSpacing() [default] ###

In [ASCII](#torch.File.ascii) mode, write additional spaces around the elements
written on disk: if writing a [Storage](storage.md), a space will be
generated between each _element_ and a _return line_ after the last
element. If only writing one element, a _return line_ will be generated
after this element.

Those spaces are supposed to exist while reading in this mode.

This is the default behavior. You can de-activate this option with the
[noAutoSpacing()](#torch.File.noAutoSpacing) method.

<a name="torch.File.binary"/>
### binary() ###

The data read or written will be in binary mode: the representation in the
`File` is the same that the one in the computer memory/register (not human
readable).  This mode is faster than [ASCII](#torch.File.ascii) but less
portable.

<a name="torch.File.clearError"/>
### clearError() ###

Clear the error.flag returned by [hasError()](#torch.File.hasError).

<a name="torch.File.close"/>
### close() ###

Close the file. Any subsequent operation will generate a `Lua` error.

<a name="torch.File.noAutoSpacing"/>
### noAutoSpacing() ###

In [ASCII](#torch.File.ascii) mode, do not put extra spaces between element
written on disk. This is the contrary of the option
[autoSpacing()](#torch.File.autoSpacing).

<a name="torch.File.synchronize"/>
### synchronize() ###

If the child class bufferize the data while writing, ensure that the data
is actually written.


<a name="torch.File.pedantic"/>
### pedantic() [default] ###

If this mode is chosen (which is the default), a `Lua` error will be
generated in case of error (which will cause the program to stop).

It is possible to use [quiet()](#torch.File.quiet) to avoid `Lua` error generation
and set a flag instead.

<a name="torch.File.position"/>
### [number] position() ###

Returns the current position (in bytes) in the file.
The first position is `1` (following Lua standard indexing).

<a name="torch.File.quiet"/>
### quiet() ###

If this mode is chosen instead of [pedantic()](#torch.File.pedantic), no `Lua`
error will be generated in case of read/write error. Instead, a flag will
be raised, readable through [hasError()](#torch.File.hasError). This flag can
be cleared with [clearError()](#torch.File.clearError)

Checking if a file is quiet can be performed using [isQuiet()](#torch.File.isQuiet).

<a name="torch.File.seek"/>
### seek(position) ###

Jump into the file at the given `position` (in byte). Might generate/raise
an error in case of problem. The first position is `1` (following Lua standard indexing).

<a name="torch.File.seekEnd"/>
### seekEnd() ###

Jump at the end of the file. Might generate/raise an error in case of
problem.

## File state query ##

These methods allow the user to query the state of the given `File`.

<a name="torch.File.hasError"/>
### [boolean] hasError() ###

Returns if an error occurred since the last [clearError()](#torch.File.clearError) call, or since
the opening of the file if `clearError()` has never been called.

<a name="torch.File.isQuiet"/>
### [boolean] isQuiet() ###

Returns a boolean which tells if the file is in [quiet](#torch.File.quiet) mode or not.

<a name="torch.File.isReadable"/>
### [boolean] isReadable() ###

Tells if one can read the file or not.

<a name="torch.File.isWritable"/>
### [boolean] isWritable() ###

Tells if one can write in the file or not.

<a name="torch.File.isAutoSpacing"/>
### [boolean] isAutoSpacing() ###

Return `true` if [autoSpacing](#torch.File.autoSpacing) has been chosen.

<a name="torch.File.referenced"/>
### referenced(ref) ###

Sets the referenced property of the File to `ref`. `ref` has to be `true` or `false`. By default it is true, which means that a File object keeps track of objects written using [writeObject](#torch.File.writeObject) method. When one needs to push the same tensor repeatedly into a file but everytime changing its contents, calling `referenced(false)` ensures desired behaviour.

<a name="torch.File.isReferenced"/>
### isReferenced() ###

Return the state set by [referenced](#torch.File.referenced).



