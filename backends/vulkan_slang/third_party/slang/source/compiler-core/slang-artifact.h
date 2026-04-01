// slang-artifact.h
#ifndef SLANG_ARTIFACT_H
#define SLANG_ARTIFACT_H

#include "../core/slang-basic.h"
#include "slang-com-helper.h"

#include <type_traits>

namespace Slang
{

/* Simplest slice types. We can't use UnownedStringSlice etc, because they implement functionality
in libraries, and we want to use these types in headers. If we wanted a C implementation it would be
easy to use a macro to generate the functionality */

template<typename T>
struct Slice
{
    const T* begin() const { return data; }
    const T* end() const { return data + count; }

    const T& operator[](Index index) const
    {
        SLANG_ASSERT(index >= 0 && index < count);
        return data[index];
    }

    Slice()
        : count(0), data(nullptr)
    {
    }
    Slice(const T* inData, Count inCount)
        : data(inData), count(inCount)
    {
    }

    const T* data;
    Count count;
};

template<typename T>
SLANG_FORCE_INLINE Slice<T> makeSlice(const T* inData, Count inCount)
{
    return Slice<T>(inData, inCount);
}

struct CharSlice : public Slice<char>
{
    typedef CharSlice ThisType;
    typedef Slice<char> Super;

    bool operator==(const ThisType& rhs) const
    {
        return count == rhs.count && (data == rhs.data || ::memcmp(data, rhs.data, count) == 0);
    }
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    explicit CharSlice(const char* in)
        : Super(in, ::strlen(in))
    {
    }
    CharSlice(const char* in, Count inCount)
        : Super(in, inCount)
    {
    }
    CharSlice()
        : Super(nullptr, 0)
    {
    }
    explicit CharSlice(const String& s)
        : CharSlice(s.begin(), s.getLength()){};
};
static_assert(std::is_trivially_copyable_v<CharSlice>);

struct TerminatedCharSlice : public CharSlice
{
    typedef TerminatedCharSlice ThisType;
    typedef CharSlice Super;

    SLANG_FORCE_INLINE bool operator==(const ThisType& rhs) const { return Super::operator==(rhs); }
    SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    /// Make convertable to char*
    SLANG_FORCE_INLINE operator const char*() const { return data; }

    explicit TerminatedCharSlice(const char* in)
        : Super(in)
    {
    }
    TerminatedCharSlice(const char* in, Count inCount)
        : Super(in, inCount)
    {
        SLANG_ASSERT(in[inCount] == 0);
    }
    TerminatedCharSlice()
        : Super("", 0)
    {
    }
};
static_assert(std::is_trivially_copyable_v<TerminatedCharSlice>);

/* As a rule of thumb, if we can define some aspect in a hierarchy then we should do so at the
highest level. If some aspect can apply to multiple items identically we move that to a separate
enum.

NOTE!
New Kinds must be added at the end. Values can be deprecated, or disabled
but never removed, without breaking binary compatability.

Any change requires a change to SLANG_ARTIFACT_KIND
*/
enum class ArtifactKind : uint8_t
{
    Invalid, ///< Invalid
    Base,    ///< Base kind of all valid kinds

    None,    ///< Doesn't contain anything
    Unknown, ///< Unknown

    BinaryFormat, ///< A generic binary format.

    Container,            ///< Container like types
    Zip,                  ///< Zip container
    RiffContainer,        ///< Riff container
    RiffLz4Container,     ///< Riff container using Lz4 compression
    RiffDeflateContainer, ///< Riff container using deflate compression

    Text, ///< Representation is text. Encoding is utf8, unless prefixed with 'encoding'.

    Source,    ///< Source (Source type is in payload)
    Assembly,  ///< Assembly (Type is in payload)
    HumanText, ///< Text for human consumption

    CompileBinary, ///< Kinds which are 'binary like' - can be executed, linked with and so forth.

    ObjectCode,    ///< Object file
    Library,       ///< Library (collection of object code)
    Executable,    ///< Executable
    SharedLibrary, ///< Shared library - can be dynamically linked
    HostCallable,  ///< Code can be executed directly on the host

    Instance, ///< Primary representation is an interface/class instance

    Json, ///< It's JSON

    CountOf,
};

/* Payload.

SlangIR and LLVMIR can be GPU or CPU orientated, so put in own category.

NOTE!
New Payloads must be added at the end. Values can be deprecated, or disabled
but never removed, without breaking binary compatability.

Any change requires a change to SLANG_ARTIFACT_PAYLOAD
*/
enum class ArtifactPayload : uint8_t
{
    Invalid, ///< Is invalid - indicates some kind of problem
    Base,    ///< The base of the hierarchy

    None,    ///< Doesn't have a payload
    Unknown, ///< Unknown but probably valid

    Source, ///< Source code

    C,     ///< C source
    Cpp,   ///< C++ source
    HLSL,  ///< HLSL source
    GLSL,  ///< GLSL source
    CUDA,  ///< CUDA source
    Metal, ///< Metal source
    Slang, ///< Slang source
    WGSL,  ///< WGSL source

    KernelLike, ///< GPU Kernel like

    DXIL,       ///< DXIL
    DXBC,       ///< DXBC
    SPIRV,      ///< SPIR-V
    PTX,        ///< PTX. NOTE! PTX is a text format, but is handable to CUDA API.
    MetalAIR,   ///< Metal AIR
    CuBin,      ///< CUDA binary
    WGSL_SPIRV, ///< SPIR-V derived via WebGPU shading language

    CPULike, ///< CPU code

    UnknownCPU,   ///< CPU code for unknown/undetermined type
    X86,          ///< X86
    X86_64,       ///< X86_64
    Aarch,        ///< 32 bit arm
    Aarch64,      ///< Aarch64
    HostCPU,      ///< HostCPU
    UniversalCPU, ///< CPU code for multiple CPU types

    GeneralIR, ///< General purpose IR representation (IR)

    SlangIR, ///< Slang IR
    LLVMIR,  ///< LLVM IR

    AST, ///< Abstract syntax tree (AST)

    SlangAST, ///< Slang AST

    CompileResults, ///< Payload is a collection of compilation results

    Metadata, ///< Metadata

    DebugInfo,   ///< Debugging information
    Diagnostics, ///< Diagnostics information

    Miscellaneous, ///< Category for miscellaneous payloads (like Log/Lock)

    Log,  ///< Log file
    Lock, ///< Typically some kind of 'lock' file. Contents is typically not important.

    PdbDebugInfo, ///< PDB debug info

    SourceMap, ///< A source map

    PostEmitMetadata, ///< Metadata from post emit (binding information)

    CountOf,
};

/* Style.

NOTE!
New Styles must be added at the end. Values can be deprecated, or disabled
but never removed, without breaking binary compatability.

Any change requires a change to SLANG_ARTIFACT_STYLE
*/
enum class ArtifactStyle : uint8_t
{
    Invalid, ///< Invalid style (indicating an error)
    Base,

    None, ///< A style is not applicable

    Unknown, ///< Unknown

    CodeLike, ///< For styles that are 'code like' such as 'kernel' or 'host'.

    Kernel,     ///< Compiled as `GPU kernel` style.
    Host,       ///< Compiled in `host` style
    Obfuscated, ///< Holds something specific to obfuscation, such as an obfuscated source map

    CountOf,
};

typedef uint8_t ArtifactFlags;
struct ArtifactFlag
{
    enum Enum : ArtifactFlags
    {
        // Don't currently have any flags
    };
};

/**
A value type to describe aspects of the contents of an Artifact.
**/
struct ArtifactDesc
{
public:
    typedef ArtifactDesc ThisType;

    typedef ArtifactKind Kind;
    typedef ArtifactPayload Payload;
    typedef ArtifactStyle Style;
    typedef ArtifactFlags Flags;

    typedef uint32_t PackedBacking;
    enum class Packed : PackedBacking;

    /// Get in packed format
    inline Packed getPacked() const;

    bool operator==(const ThisType& rhs) const
    {
        return kind == rhs.kind && payload == rhs.payload && style == rhs.style &&
               flags == rhs.flags;
    }
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    /// Construct from the elements
    static ThisType make(
        Kind inKind,
        Payload inPayload,
        Style inStyle = Style::Unknown,
        Flags flags = 0)
    {
        return ThisType{inKind, inPayload, inStyle, flags};
    }
    static ThisType make(Kind inKind, Payload inPayload, const ThisType& base)
    {
        return ThisType{inKind, inPayload, base.style, base.flags};
    }

    /// Construct from the packed format
    inline static ThisType make(Packed inPacked);

    Kind kind;
    Payload payload;
    Style style;
    Flags flags;
};

// --------------------------------------------------------------------------
inline ArtifactDesc::Packed ArtifactDesc::getPacked() const
{
    typedef PackedBacking IntType;
    return Packed((IntType(kind) << 24) | (IntType(payload) << 16) | (IntType(style) << 8) | flags);
}

// --------------------------------------------------------------------------
inline /* static */ ArtifactDesc ArtifactDesc::make(Packed inPacked)
{
    const PackedBacking packed = PackedBacking(inPacked);

    ThisType r;
    r.kind = Kind(packed >> 24);
    r.payload = Payload(uint8_t(packed >> 16));
    r.style = Style(uint8_t(packed >> 8));
    r.flags = uint8_t(packed);

    return r;
}

// Forward declare
class IOSFileArtifactRepresentation;
class IPathArtifactRepresentation;

class IArtifactRepresentation;

// Controls what items can be kept.
enum class ArtifactKeep
{
    No,  ///< Don't keep the item
    Yes, ///< Yes keep the final item
    All, ///< Keep the final item and any intermediataries
};

/// True if can keep an intermediate item
SLANG_INLINE bool canKeepIntermediate(ArtifactKeep keep)
{
    return keep == ArtifactKeep::All;
}
/// True if can keep
SLANG_INLINE bool canKeep(ArtifactKeep keep)
{
    return Index(keep) >= Index(ArtifactKeep::Yes);
}
/// Returns the keep type for an intermediate
SLANG_INLINE ArtifactKeep getIntermediateKeep(ArtifactKeep keep)
{
    return (keep == ArtifactKeep::All) ? ArtifactKeep::All : ArtifactKeep::No;
}

/* Forward define */
class IArtifactHandler;

/* The IArtifact interface is designed to represent some Artifact of compilation. It could be input
to or output from a compilation.

An abstraction is desirable here, because depending on the compiler the artifact/s could be

* A file on the file system
* A blob
* Multiple files
* Some other (perhaps multiple) in memory representations
* A name

The artifact uses the Blob as the canonical in memory representation.

Some downstream compilers require the artifact to be available as a file system file, or to produce
artifacts that are files. The IArtifact type allows to abstract away this difference, including the
ability to turn an in memory representation into a temporary file on the file system.

The mechanism also allows for 'Containers' which allow for Artifacts to contain other Artifacts
(amongst other things). Those artifacts may be other files. For example a downstream compilation
that produces results as well as temporary files could be a Container containing artifacts for

* Diagnostics
* Temporary files (of known and unknown types)
* Files that contain known types
* Callable interface (an ISlangSharedLibrary)

There are several types of ways to associate data with an artifact:

* A representation
* An associated artifact
* A child artifact

A `representation` has to wholly represent the artifact. That representation could be a blob, a file
on the file system, an in memory representation. There are two classes of `Representation` - ones
that can be turned into blobs (and therefore derive from IArtifactRepresentation) and ones that are
in of themselves a representation (such as a blob or or ISlangSharedLibrary).

`Associated artifacts` hold information that is associated with the artifact. It could be part
of the representation, or useful for the implementation of a representation. Could also be
considered as a kind of side channel to associate arbitrary data including temporary data with an
artifact.

A `child artifact` belongs to the artifact, within the hierarchy of artifacts.
*/
class IArtifact : public ICastable
{
public:
    SLANG_COM_INTERFACE(
        0xf90acdb0,
        0x9a4a,
        0x414e,
        {0x85, 0x45, 0x8b, 0x26, 0xc9, 0x2d, 0x94, 0x42})

    enum class ContainedKind
    {
        Representation,
        Associated,
        Children,
    };

    typedef ArtifactDesc Desc;

    typedef ArtifactKind Kind;
    typedef ArtifactPayload Payload;
    typedef ArtifactStyle Style;
    typedef ArtifactFlags Flags;

    typedef ArtifactKeep Keep;

    /// Get the Desc defining the contents of the artifact
    virtual SLANG_NO_THROW Desc SLANG_MCALL getDesc() = 0;

    /// Returns true if the artifact in principal exists
    virtual SLANG_NO_THROW bool SLANG_MCALL exists() = 0;

    /// Load as a blob
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadBlob(Keep keep, ISlangBlob** outBlob) = 0;

    /// Require artifact is available as a file.
    /// NOTE! May need to serialize and write as a temporary file.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    requireFile(Keep keep, IOSFileArtifactRepresentation** outFileRep) = 0;

    /// Load the artifact as a shared library
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadSharedLibrary(ArtifactKeep keep, ISlangSharedLibrary** outSharedLibrary) = 0;

    /// Get the name of the artifact. This can be empty.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getName() = 0;
    /// Set the name associated with the artifact
    virtual SLANG_NO_THROW void SLANG_MCALL setName(const char* name) = 0;

    /// Add associated artifacts with this artifact
    virtual SLANG_NO_THROW void SLANG_MCALL addAssociated(IArtifact* artifact) = 0;
    /// Get the list of associated items
    virtual SLANG_NO_THROW Slice<IArtifact*> SLANG_MCALL getAssociated() = 0;

    /// Add a representation
    virtual SLANG_NO_THROW void SLANG_MCALL addRepresentation(ICastable* castable) = 0;
    /// Add a representation that doesn't derive from IArtifactRepresentation
    virtual SLANG_NO_THROW void SLANG_MCALL addRepresentationUnknown(ISlangUnknown* rep) = 0;
    /// Get all the representations
    virtual SLANG_NO_THROW Slice<ICastable*> SLANG_MCALL getRepresentations() = 0;

    /// Given a typeGuid representing the desired type get or create the representation.
    /// If found outCastable holds an entity that *must* be castable to typeGuid
    /// Use the keep parameter to determine if the representation should be cached on the artifact/s
    /// or not.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getOrCreateRepresentation(const Guid& typeGuid, ArtifactKeep keep, ICastable** outCastable) = 0;

    /// Get the handler used for this artifact. If nullptr means the default handler will be used.
    virtual SLANG_NO_THROW IArtifactHandler* SLANG_MCALL getHandler() = 0;
    /// Set the handler associated with this artifact. Setting nullptr will use the default handler.
    virtual SLANG_NO_THROW void SLANG_MCALL setHandler(IArtifactHandler* handler) = 0;

    /// Returns the result of expansion. Will return SLANG_E_UNINITIALIZED if expansion hasn't
    /// happened
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getExpandChildrenResult() = 0;
    /// Sets all of the children, will set the expansion state to SLANG_OK
    virtual SLANG_NO_THROW void SLANG_MCALL
    setChildren(IArtifact* const* children, Count count) = 0;
    /// Will be called implicitly on access to children
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL expandChildren() = 0;

    /// Add the artifact to the list
    virtual SLANG_NO_THROW void SLANG_MCALL addChild(IArtifact* artifact) = 0;
    /// Get the children, will only remain valid if no mutation of children list
    virtual SLANG_NO_THROW Slice<IArtifact*> SLANG_MCALL getChildren() = 0;

    /// Find a represention from the specified list
    virtual SLANG_NO_THROW void* SLANG_MCALL
    findRepresentation(ContainedKind kind, const Guid& guid) = 0;
    /// Clear all of the contained kind
    virtual SLANG_NO_THROW void SLANG_MCALL clear(ContainedKind kind) = 0;
    /// Remove entry at index for the specified kind
    virtual SLANG_NO_THROW void SLANG_MCALL removeAt(ContainedKind kind, Index i) = 0;
};

template<typename T>
SLANG_FORCE_INLINE T* findRepresentation(IArtifact* artifact)
{
    return reinterpret_cast<T*>(
        artifact->findRepresentation(IArtifact::ContainedKind::Representation, T::getTypeGuid()));
}

template<typename T>
SLANG_FORCE_INLINE T* findAssociatedRepresentation(IArtifact* artifact)
{
    return reinterpret_cast<T*>(
        artifact->findRepresentation(IArtifact::ContainedKind::Associated, T::getTypeGuid()));
}

template<typename T>
SLANG_FORCE_INLINE T* findChildRepresentation(IArtifact* artifact)
{
    return reinterpret_cast<T*>(
        artifact->findRepresentation(IArtifact::ContainedKind::Children, T::getTypeGuid()));
}

/* The IArtifactRepresentation interface represents a single representation that can be part of an
artifact. It's special in so far as

* IArtifactRepresentation can be queried for it's underlying object class
* Can determine if the representation exists (for example if it's on the file system)
* Can optionally serialize into a blob
*/
class IArtifactRepresentation : public ICastable
{
    SLANG_COM_INTERFACE(0xa3790eb, 0x22b9, 0x430e, {0xbf, 0xc6, 0x24, 0x6c, 0x5b, 0x5c, 0xcd, 0x0})

    /// Create a representation of the specified typeGuid interface.
    /// Calling castAs on the castable will return the specific type
    /// Returns SLANG_E_NOT_IMPLEMENTED if an implementation doesn't implement
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    createRepresentation(const Guid& typeGuid, ICastable** outCastable) = 0;

    /// Returns true if this representation exists and is available for use.
    virtual SLANG_NO_THROW bool SLANG_MCALL exists() = 0;
};

/* Handler provides functionality external to the artifact */
class IArtifactHandler : public ICastable
{
    SLANG_COM_INTERFACE(
        0x6a646f57,
        0xb3ac,
        0x4c6a,
        {0xb6, 0xf1, 0x33, 0xb6, 0xef, 0x60, 0xa6, 0xae});

    /// Given an artifact expands children
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL expandChildren(IArtifact* container) = 0;
    /// Given an artifact gets or creates a representation.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getOrCreateRepresentation(
        IArtifact* artifact,
        const Guid& guid,
        ArtifactKeep keep,
        ICastable** outCastable) = 0;
};

} // namespace Slang

#endif
