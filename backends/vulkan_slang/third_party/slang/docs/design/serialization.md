Serialization
=============

Slang has a collection of serialization components. This document will be used to discuss serialization around IR/AST and modules as it currently exists. A separate document will describe the future serialization plans.

All of the serialization aspects here focus on binary serialization. 

The major components are

* IR Serialization
* AST/Generalized Serialization
* SourceLoc Serialization
* Riff container
* C++ Extractor

Generalized Serialization
=========================

Generalized serialization is the mechanism used to save 'arbitrary' C++ structures. It is currently used for serializing the AST. Although not necessary, generalized serialization is typically helped out by the `C++ extractor`, which can rudimentary parse C++ source, and extract class-like types and their fields. The extraction then produces header files that contain macros that can then be used to drive serialization. 

It's worth discussing briefly what the philosophy is behind the generalized serialization system. To talk about this design it is worth talking a little about serialization in general and the issues involved. Lets say we have a collection of C++ class instances that contain fields. Some of those fields might be pointers. Others of the fields might be a templated container type like a Dictionary<K,V>. We want to take all of these instances, write them to a file, such that when we read the file back we will have the equivalent objects with equivalent relationships. 

We could imagine a mechanism that saved off each instance, by writing off the address of the object, and then the in memory representation for all the instances that can be reached. When reading back the objects would be at different locations in memory. If we knew where the pointers were, we could use a map of old pointers to the new instances and fix them up. Problems with this simple mechanism occur because...

* If we try to read back on a different machine, with a different pointer size, the object layout will be incompatible
* If we try to read back on the same machine where the source is compiled by a different compiler, the object layout might be incompatible (say bool or enum are different size)
* Endianness might be different
* Knowing where all the pointers are and what they point to and therefore what to serialize is far from simple. 
* The alignment of types might be different across different processors and different compilers 

The implementation makes a distinction between the 'native' types, the regular C++ in memory types and 'serial' types. Each serializable C++ type has an associated 'serial' type - with the distinction that it can be written out and (with perhaps some other data) read back in to recreate the C++ type. The serial type can be a C++ type, but is such it can be written and read from disk and still represent the same data. 

The approach taken in Slang is to have each 'native' type (ie the C++ type) that is being serialized have a serializable 'dual' type. The serial type can be an explicit C++ type, or it might implicit (ie not have a C++ type) and calculated at Slang startup. 

The important point here is that the Serial type must writable on one target/process and readable correctly on another. 

The easy cases are types that have an alignment and representation that will work over all targets. These would be most built in types - integrals 8,16,32 and float32. Note that int64 and double are *not* so trivial, because on some targets that require 8 byte alignment - so they must be specially defined to have 8 byte alignment. 

Another odd case is bool - it has been on some compilers 32 bits, and on others 8 bits. Thus we need to potentially convert.

For this and other types it is therefore necessary to have function that can convert to and from the serialized dual representation.

## Generalized Field Conversion

For types that contain fields, it would be somewhat laborious to have to write all of the conversion functions by hand. To avoid this we use the macro output of the C++ extractor to automatically generate the appropriate functions. 

Take DeclRefExpr from the AST hierarchy - the extractor produces a macro something like...

```
#define SLANG_FIELDS_ASTNode_DeclRefExpr(_x_, _param_)\
    _x_(scope, (RefPtr<Scope>), _param_)\
    _x_(declRef, (DeclRef<Decl>), _param_)\
    _x_(name, (Name*), _param_)
``` 

DeclRefExpr derives from Expr and this might hold other fields and so forth. 

The macros can generate the appropriate conversion functions *if* we have the conversion functions for the field types. Field type conversions can be specified via a special macro that describes how the conversion to and from the type work. To make the association between the native and serial type, as well as provide the functions to convert, we use the template

```
template <typename T>
struct SerialTypeInfo;
```
and specialize it for each native type. The specialization holds

* SerialType - The type that will be used to represent the native type
* NativeType - The native type
* SerialAlignment - A value that holds what kind of alignment the SerialType needs to be serializable (it may be different from SLANG_ALIGN_OF(SerialType)!)
* toSerial - A function that with the help of SerialWriter convert the NativeType into the SerialType
* toNative - A function that with the help of SerialReader convert the SerialType into the NativeType

It is useful to have a structure that can hold the type information, so it can be stored. That is achieved with

```
template <typename T>
struct SerialGetType;
```

This template can be specialized for a specific native types - but all it holds is just a function getType, which returns a `SerialType*`, which just holds the information held in the SerialTypeInfo template, but additionally including the size of the SerialType.

So we need to define a specialized SerialTypeInfo for each type that can be a field in a NodeBase/RefObject derived type. We don't need to define anything explicitly for the NodeBase derived types, as we will just generate the layout from the fields. How do we know the fields? We just used the macros generated from the C++ extractor.

So first a few things to observe...

1) Some types don't need any conversion to be serializable - int8_t, or float the bits can just be written out and read in (1)
2) Some types need a conversion but it's very simple - for example an enum without explicit size, being written as an explicit size
3) Some types can be written out but would not be directly readable or usable with different targets/processors, so need converting
4) Some types require complex conversions that require programmer code - like Dictionary/List

For types that need no conversion (1), we can just use the template SerialIdentityTypeInfo

```
template <>
struct SerialTypeInfo<SomeType> : public SerialIdentityTypeInfo<SomeType> {};
```

This specialization means that SomeType can be written out and read in across targets/compilers without problems.

For (2) we have another template that will do the conversion for us

```
template <typename NATIVE_T, typename SERIAL_T>
struct SerialConvertTypeInfo;
```

That we can use as above, and specify the native and serial types.

For (3) there are a few scenarios. For any field in a serial type we must store in the serialized type such that the representation will work across all processors/compilers. So one problematic type is `bool`. It's not specified how it's laid out in memory - and some compiles have stored it as a word. Most recently it's been stored as a byte. To make sure bool is ok for serialization therefore we store as a uint8_t.

Another example would be double. It's 64 bits, but on some arches/compilers it's SLANG_ALIGN_OF is 4 and on others it's 8. On some architectures a non aligned read will lead to a fault, on others it might be very slow. To work around this problem therefore we have to ensure double has the alignment that will work across all targets - and that alignment is 8. In that specific case that issue is handled via SerialBasicTypeInfo, which makes the SerialAlignment the sizeof the type.

For (4) there are a few things to say. First a type can always implement a custom version of how to do a conversion by specializing `SerialTypeInfo`. But there remains another nagging issue - types which allocate/use other memory that changes at runtime. Clearly we cannot define 'any size of memory' in a fixed SerialType defined in a specialization of SerialTypeInfo. The mechanism to work around this is to allow arbitrary arrays to be stored, that can be accessed via an SerialIndex. This will be discussed more once we discuss a little more about the file system, and SerialIndex. 

## Struct value types

There is a mechanism to allow the simple serialization of 'value' struct types for this to work it requires

* The fields of the struct are serializable and public
* The super class (if there is one) is serializable

If this is the case, it is not necessary to write a `SerialTypeInfo<T>` specialization, the C++ extractor and it's reflection can generate the specialization for you. The steps needed

* Place SLANG_VALUE_CLASS(your type) in the definition of your struct 
* Make sure that the header containing the struct definition is included in the ones C++ extractor examines
* Instead of implementing SerialTypeInfo for your type use the macro SLANG_VALUE_TYPE_INFO(your type)

If there are problems looking at the contents of `slang-generated-value.h` and `slang-generated-value-macro.h`.

It should be noted that currently because of limitations in the C++ extractor, all of the types must be defined in the same scope.

Also because value types are always fields in generalized serialization, they do not need to be identified with a sub type, even though C++ extractor does generate a ValueType enum.

## Generalized Serialization Format

The serialization format used is 'stream-like' with each 'object' stored in order. Each object is given an index starting from 1. 0 is used to be in effect nullptr. The stream looks like

```
SerialInfo::Entry (for index 1)
Payload for type in entry

SerialInfo::Entry (for index 2)
Payload for type in entry

... 
... 
```

That when writing we have an array that maps each index to a pointer to the associated header. We also have a map that maps native pointers to their indices. The Payload *is* the SerialType for thing saved. The payload directly follows the Entry data. Each object in this list can only be a few types of things

* NodeBase derived type
* RefObject derived type
* String
* Array

The actual Entry followed by the payloads are allocated and stored when writing in a MemoryArena. When we want to write into a stream, we can just iterate over each entry in order and write it out.

You may have spotted a problem here - that some Entry types can be stored without alignment (for example a string - which stores the length VarInt encoded followed by the characters). Others require an alignment - for example an NodeBase derived type that contains a int64_t will *require* 8 byte alignment. That as a feature of the serialization format we want to be able to just map the data into memory, and be able to access all the SerialType as is on the CPU. For that to work we *require* that the payload for each entry has the right alignment for the associated SerialType.

To achieve this we store in the Entry it's alignment requirement *AND* the next entries alignment. With this when we read, as we as stepping through the entries we can find where the next Entry starts. Because the payload comes directly after the Entry - the Entrys size must be a modulo of the largest alignment the payload can have.

For the code that does the conversion between native and serial types it uses either the SerialWriter or SerialReader. This provides the mechanism to turn a pointer into a serializable `SerialIndex` and vice versa. There are some special functions for turning string like types to and forth.

The final mechanism is that of 'Arrays'. An array allows reading or writing a chunk of data associated with a `SerialIndex`. The chunk of data *must* hold data that is serializable. If the array holds pointers - then the serialized array must hold an array of `SerialIndex` values that represent those pointers. When reading back in `SerialIndex` is converted back to a pointer.

Arrays are the escape hatch that allows for more complex types to serialize. Dictionaries for example are saved as a serial type that is two SerialIndices one to a keys array and one to a values array.

Note that writing has two phases, serializing out into an SerialWriter, and then secondly writing out to a stream. 

## Object/Reference Types

When talking about Object/Reference types this means types that can be referenced natively as pointers. Currently that means `NodeBase` and `SerialRefObject` derived types. 

The SerialTypeInfo mechanism is generally for *fields* of object types. That for derived types we use the C++ extractors field list to work out the native fields offsets and types. With this we can then calculate the layout for NodeBase types such that they follow the requirements for serialization - such as alignment and so forth.

This information is held in the SerialClasses, which for a given TypeKind/SubType gives a SerialClassInfo, that specifies fields for just that type. 

It is trivial to work out the SubType for a NodeBase derived class - its just the astTypeNode member in the `NodeBase` type. For a SerialRefObject it is determined by first calling 

```
const ReflectClassInfo* getClassInfo() const;
```

Then the m_classID in the `ReflectClassInfo` is the subtype.

## Reading

Due to the care in writing reading is relatively simple. We can just take the contents of the file and put in memory, as long as in memory it has an alignment of at least MAX_ALIGNMENT. Then we can build up an entries table by stepping through the data and writing the pointer.

The toNative functions take an SerialReader - this allows the implementation to ask for pointers and arrays from other parts of the serialized data. It also allows for types to be lazily reconstructed if necessary.

Lazy reconstruction may be useful in the future to partially reconstruct a sub part of the serialized data. In the current implementation, lazy evaluation is used on Strings. The m_objects array holds all of the recreated native 'objects'. Since the objects can be derived from different base classes the associated Entry will describe what it really is.

For the String type, we initially store the object pointer as null. If a string is requested from that index, we see if the object pointer is null, if it is we have to construct the StringRepresentation that will be used. An extra wrinkle is that we allow accessing of a serialized String as a Name or a string or a UnownedSubString. Fortunately a Name just holds a string, and a Name remains in scope as long as it's NamePool does which is passed in.

### Serial type replacement

In generalized serialization systems such as with Java there is a mechanism for reference types to replace their representation on writing, and then on reading replace the read type with the actual type. Write replacement is already used when serializing out modules via the `SerialFilter` mechanism. The actual implementation is `ModuleSerialFilter`, if an object is referenced in a different module that is explicitly specified, it is replaced with `ImportExternalDecl`, that names the actual definition to use. 

Currently when deserializing, the `ImportExternalDecl` is *not* turned back into the item it references. This means there are likely pointers which point to invalid objects. 

If we wanted to do a replacement on reconstruction we could

We could modify reading as follows.

1) Don't construct anything at the start
2) Find 'root's they must be created and deserialized first
  . Any read/writeReplace is a root
  . Any marked (like SourceLocData) is a root. (When deconstructed it also needs to add information to the Reader)
  . The root of the objects (note we could just deserialize first to last if not already constructed)
3) During deserialization pointer references and constructed on demand
4) Extra code is needed to make sure there aren't cycles. Any object is either Pre/Created/Deserialized.

### Other reading issues

As touched on elsewhere SourceLoc information has to be carefully handled. Within the generalized serialization we have the additional problem that we probably don't want to attach SourceLoc or other types explicitly to the SerialReader/SerialWriter. The mechanism to work around this is via the `SerialExtraObjects` structure. This allows types to optionally be available to the Reader/Writer without it having to explicitly know anything about the type.

For all types supporting this mechanism they *require* that they are added to the `SerialExtraType` enum, and that they embed a static kExtraType field in the type. This solution is not as flexible as perhaps using a string map or something of that sort, but it does make lookup very fast and simple which is likely significant as many types contain the SourceLoc type for example.

## Identifying Types

How a NodeBase derived type identifies itself is not directly compatible with how a SerialRefObject represents itself. The NodeBase derived type uses `ASTNodeType` enum. The SerialRefObject uses a `RefObjectType` enum. Thus to uniquely identify a type we typically actually need two bits of information the `SerialTypeKind` as well as the `SerialSubType`. 

```
enum class SerialTypeKind : uint8_t
{
    Unknown,

    String,             ///< String                         
    Array,              ///< Array

    NodeBase,           ///< NodeBase derived
    RefObject,          ///< RefObject derived types

    CountOf,
};
```

String and Array are special cases described elsewhere. 

If the `SerialTypeKind` is `NodeBase`, then the `SerialSubType` *is* the ASTNodeType. If the `SerialTypeKind` is `RefObject` then the `SerialSubType` *is* RefObjectType. 

`SerialClasses` holds the information on how to serialize non-field Serial types. For each `SerialTypeKind`/`SerialSubType` it holds a `SerialClass`. The SerialClass holds the size of the type, the amount of fields, and the field information. The fields themselves contain a `SerialFieldType` - this holds the pointers to the functions to convert to and from `native` to `serial` types. 

In order to set up all types in a SerialClass without tying SerialClasses to an implementation the class `SerialClassesUtil` is used to set up Slang serialized types in a `SerialClasses` instance. 

IR Serialization
================

Currently IR serialization is handled via a separate mechanism to 'generalized' serialization.

This mechanism is *much* simpler than generalized serialization, because by design the IR types are very homogeneous in style. There are a few special cases, but in general an instruction consists of

* It's type
* A SourceLoc
* 0 or more operands.
* 0 or more children. 

Within the IR instructions are pointers to IRInst derived types. As previously discussed serializing pointers directly is generally not a good idea. To work around this the pointers are turned into 32 bit indices. Additionally we know that an instruction can belong to at most one other instruction. 

When serializing out special handling is made for child instructions - their indices are made to be a contiguous range of indices for all instructions that belong to each parent. The indices are ordered into the same order as the children are held in the parent. By using this mechanism it is not necessary to directly save off the indices that belong to a parent, only the range of indices. 

The actual serialization mechanism is similar to the generalized mechanism - referenced objects are saved off in order of their indices. What is different is that the encoding fixes the size of the Inst to `IRSerialData`. That this can hold up to two operands, if the instruction has more than two operands then one of the UInt32 is the operand count and the other is an offset to a list of operands. It probably makes sense to alter this in the future to stream the instructions payload directly. 

IR serialization allows a simple compression mechanism, that works because much of the IR serialized data is UInt32 data, that can use a variable byte encoding.

AST Serialization
=================

AST serialization uses the generalized serialization mechanism. 

When serializing out an AST module it is typical to want to just serialize out the definitions within that module. Without this, the generalized serializer will crawl over the whole of the AST structure serializing every thing that can be reached - including the whole of the core module.

The filter `ModuleSerialFilter` can be used when writing the AST module, it will replace any references to elements outside of the current module with a `ImportExternalDecl`. This contains a mangled name to the item being referenced in another module. 

When serializing back in, it may be possible to turn these references into the actual element, if the module containing the definition has been loaded. This probably can't work in general though, as if we have two modules that reference items in the other, then it isn't possible to fix up on load. 

A way around this would be to not replace on reading (or only replace items that can be found). Then go through the `ImportExternalDecl` elements doing the lookup, and potentially loading other modules. There are several issues here though 

* On first loading pointers that have been replaced will claim to be a type they are typically *NOT*
* Once we have determined what `ImportExternalDecl` should replaced with, how do we replace it?

On the first point, this is perhaps undesirable (on a variety of levels - such as debugging), but isn't as terrible as it could be, as the actual type identification is managed by Slang via the `astTypeNode`. So there is a simple way of identifying what the type actually is.

On the second point - this isn't so simple. If we had an indirection, we could do the replacement quickly and trivially, without having to to fix up all the pointers. We probably don't want to add such an indirection into the pointer based system so choices are

* Store where all the pointers are, and fix them up
* Traverse the hierarchy replacing pointers

Within the current mechanism storing where all the pointers are is not so simple - it would require the setting of any pointer to record where that pointer is stored, and for that to remain the location. Doing so would require setting all pointers to go through some recording mechanism. Pointers held in containers - like the Dictionary may not be directly available. Moreover even if they *were* doing such a behavior may break the containers invariants - for example replacing a keys pointer, may change it's hash.

Traversing the hierarchy would be something akin to the serialization process. It would require specially handling for field types to do the replacement. There would need to be special handling for struct value types. 

SourceLoc Serialization
=======================

SourceLoc serialization presents several problems. Firstly we have two distinct serialization mechanisms that need to use it - IR serialization and generalized serialization. That being the case it cannot be saved directly in either, even though it may be referenced by either. 

To keep things simple for now we build up SourceLoc information for both IR and general serialization via their writers adding their information into a SerialSourceLocWriter. Then we can save this information into a RIFF section, that can be loaded before either general or IR deserialization is used.  

When reading the SourceLoc information has to be located and deserialized before any AST or IR deserialization. The SourceLoc data can then be turned into a SerialSourceLocReader, which is then either set on the `SerialReaders` `SerialExtraObjects`. Or passed to the `IRSerialReader`.

Riff Container
==============

[Riff](https://en.wikipedia.org/wiki/Resource_Interchange_File_Format) is used as a mechanism to store binary sections. The format allows for a hierarchy of `chunks` that hold binary data. How the data is interpreted depends on the [FOURCC](https://en.wikipedia.org/wiki/FourCC) associated with each chunk. 

As previously touched on there are multiple different mechanisms used for serialization. IR serialization, generalized serialization, SourceLoc serialization - there are also other uses, such as serializing of entry point information. Riff is used to combine all of these incompatible binary parts together such that they can be stored together.

The handling of these riff containers is held within the `SerialContainerUtil` class. 

C++ Extractor
=============

The C++ Extractor is the tool `slang-cpp-extractor` that can be used to example C++ files to extract class definitions and associated fields. These files contain in the form of macros information about each class as well as reflected fields. These generated files can then be used to implement serialization without having to explicitly specify fields in C++ source code.

Issues
======

* No support for forward/backward compatibility. 
** Adding fields/classes will typically break compatibility
* Binary files do not contain data to describe themselves
** It is *not* possible to write a stand alone tool that can dump any serialized file - it's iterpretation depends on the version of Slang it was written from
* The Riff mechanism use for container usage is somewhat ad-hoc
* Re-referencing AST nodes from other modules does not happen automatically on deserialization
* There are several mechanisms used for serialization that are not directly compatible

## C++ extractor issues

* All types (and typedefs) that are serialized must be defined in the same scope - child types don't work correctly 
* When using value serialization serialization all the members that are serializable must be public
* The types output in slang fields do not correctly take into account scope (this is a similar issue to the issue above)
