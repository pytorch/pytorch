<a name="torch.DiskFile.dok"/>
# DiskFile #

Parent classes: [File](file.md)

A `DiskFile` is a particular `File` which is able to perform basic read/write operations
on a file stored on disk. It implements all methods described in [File](file.md), and
some additional methods relative to _endian_ encoding.

By default, a `DiskFile` is in [ASCII](file.md#torch.File.binary) mode. If changed to
the [binary](file.md#torch.File.binary) mode, the default endian encoding is the native
computer one.

The file might be open in read, write, or read-write mode, depending on the parameter
`mode` (which can take the value `"r"`, `"w"` or `"rw"` respectively) 
given to the [torch.DiskFile(fileName, mode)](#torch.DiskFile).

<a name="torch.DiskFile"/>
### torch.DiskFile(fileName, [mode], [quiet]) ###

_Constructor_ which opens `fileName` on disk, using the given `mode`. Valid `mode` are
`"r"` (read), `"w"` (write) or `"rw"` (read-write). Default is read mode.

If read-write mode, the file _will be created_ if it does not exists. If it
exists, it will be positionned at the beginning of the file after opening.

If (and only if) `quiet` is `true`, no error will be raised in case of
problem opening the file: instead `nil` will be returned.

The file is opened in [ASCII](file.md#torch.File.ascii) mode by default.

<a name="torch.DiskFile.bigEndianEncoding"/>
### bigEndianEncoding() ###

In [binary](file.md#torch.File.binary) mode, force encoding in _big endian_. 
(_big end first_: decreasing numeric significance with increasing memory
addresses)

<a name="torch.DiskFile.isBigEndianCPU"/>
### [boolean] isBigEndianCPU() ###

Returns `true` if, and only if, the computer CPU operates in _big endian_.
_Big end first_: decreasing numeric significance with increasing
memory addresses.

<a name="torch.DiskFile.isLittleEndianCPU"/>
### [boolean] isLittleEndianCPU() ###

Returns `true` if, and only if, the computer CPU operates in _little endian_.
_Little end first_: increasing numeric significance with increasing
memory addresses.

<a name="torch.DiskFile.littleEndianEncoding"/>
### littleEndianEncoding() ###

In [binary](file.md#torch.File.binary) mode, force encoding in _little endian_.
(_little end first_: increasing numeric significance with increasing memory
addresses)

<a name="torch.DiskFile.nativeEndianEncoding"/>
### nativeEndianEncoding() ###

In [binary](file.md#torch.File.binary) mode, force encoding in _native endian_.


