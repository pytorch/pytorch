#ifndef SLANG_RIFF_H
#define SLANG_RIFF_H

#include "slang-basic.h"
#include "slang-memory-arena.h"
#include "slang-semantic-version.h"
#include "slang-stream.h"
#include "slang-writer.h"

namespace Slang
{

// http://fileformats.archiveteam.org/wiki/RIFF
// http://www.fileformat.info/format/riff/egff.htm


typedef uint32_t FourCC;

/* Use of macros to construct and extract from FourCC means the FourCC ordering can be fixed for
 * endian differences. */

#if SLANG_LITTLE_ENDIAN

#define SLANG_FOUR_CC(c0, c1, c2, c3) \
    ((FourCC(c0) << 0) | (FourCC(c1) << 8) | (FourCC(c2) << 16) | (FourCC(c3) << 24))

#else

#define SLANG_FOUR_CC(c0, c1, c2, c3) \
    ((FourCC(c0) << 24) | (FourCC(c1) << 16) | (FourCC(c2) << 8) | (FourCC(c3) << 0))

#endif

enum
{
    kRiffPadSize = 2, ///< We only align to 2 bytes
    kRiffPadMask = kRiffPadSize - 1,
};

// Uses it's own version of a hash
typedef int RiffHashCode;

struct RiffHeader
{
    FourCC type;   ///< The FourCC code that identifies this chunk
    uint32_t size; ///< Size does *NOT* include the riff chunk size. The size can be byte sized, but
                   ///< on storage it will always be treated as aligned up by 4.
};

struct RiffListHeader
{
    RiffHeader chunk;
    FourCC subType;
    // This is then followed by the contained subchunk/s
};

struct RiffFourCC
{
    /// A 'riff' is the high level file container. It is followed by a subtype and then the
    /// contained chunks.
    static const FourCC kRiff = SLANG_FOUR_CC('R', 'I', 'F', 'F');
    /// A list is the same as a 'riff' except can be placed anywhere in hierarchy.
    static const FourCC kList = SLANG_FOUR_CC('L', 'I', 'S', 'T');

private:
    RiffFourCC() = delete;
};

// Follows semantic version rules
// https://semver.org/
//
// major.minor.patch
// Patch versions indicate a change.
// Minor means a change that is backwards compatible with previous minor versions. A step in minor
// and/or major zeros patch. Major means a non compatible change. A step in major, zeros minor and
// patch.
struct RiffSemanticVersion
{
    typedef RiffSemanticVersion ThisType;
    typedef uint32_t RawType;

    /// ==
    bool operator==(const ThisType& rhs) const { return m_raw == rhs.m_raw; }
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    /// A patch change indices a different version but does not change the compatibility of the
    /// format
    int getPatch() const { return m_raw & 0xff; }
    /// A minor change implies a format change that is backwards compatible
    int getMinor() const { return (m_raw >> 8) & 0xff; }
    /// A major change is binary incompatible by default
    int getMajor() const { return (m_raw >> 16); }

    SemanticVersion asSemanticVersion() const
    {
        return SemanticVersion(getMajor(), getMinor(), getPatch());
    }

    static RawType makeRaw(int major, int minor, int patch)
    {
        SLANG_ASSERT((major | minor | patch) >= 0);
        SLANG_ASSERT(major < 0x10000 && minor < 0x100 && patch < 0x100);
        return (RawType(major) << 16) | (RawType(minor) << 8) | RawType(patch);
    }

    static RiffSemanticVersion makeFromRaw(RawType raw)
    {
        ThisType version;
        version.m_raw = raw;
        return version;
    }

    static RiffSemanticVersion make(int major, int minor, int patch)
    {
        return makeFromRaw(makeRaw(major, minor, patch));
    }
    static RiffSemanticVersion make(const SemanticVersion& in)
    {
        return makeFromRaw(makeRaw(in.m_major, in.m_minor, in.m_patch));
    }

    /// True if the read version is compatible with the current version, based on semantic rules.
    static bool areCompatible(const ThisType& currentVersion, const ThisType& readVersion)
    {
        const RawType currentRaw = currentVersion.m_raw;
        const RawType readRaw = readVersion.m_raw;

        // Must have same major version.
        // For minor version, the read version must be less than or equal.
        return ((currentRaw & 0xffff0000) == (readRaw & 0xffff0000)) &&
               ((currentRaw & 0xff00) >= (readRaw & 0xff00));
    }

    RawType m_raw;
};

/* A helper class that makes reading data from a data block simpler */
class RiffReadHelper
{
public:
    template<typename T>
    SlangResult read(T& out)
    {
        if (m_cur + sizeof(T) > m_end)
        {
            return SLANG_FAIL;
        }
        // TODO: consider whether this type should enforce alignment.
        // SLANG_ASSERT((size_t(m_cur) & (SLANG_ALIGN_OF(T) - 1)) == 0);
        ::memcpy(&out, m_cur, sizeof(T));
        m_cur += sizeof(T);
        return SLANG_OK;
    }

    /// Get the data
    const uint8_t* getData() const { return m_cur; }
    /// Get the remaining size
    size_t getRemainingSize() const { return size_t(m_end - m_cur); }

    RiffReadHelper(const uint8_t* data, size_t size)
        : m_start(data), m_end(data + size), m_cur(data)
    {
    }

    SlangResult skip(size_t size)
    {
        if (m_cur + size > m_end)
        {
            return SLANG_FAIL;
        }
        m_cur += size;
        return SLANG_OK;
    }

protected:
    const uint8_t* m_start;
    const uint8_t* m_end;
    const uint8_t* m_cur;
};

/* A container for data in RIFF format. Holds the contents in memory.

With the data held in memory allows for adding or removing chunks at will.

A future implementation does not necessarily have to be backed by memory when construction,
as data could be written to stream, and the chunk sizes written by seeking back over the file and
setting the value.

In normal usage the chunk sizes are calculated during construction. If the structure is changed, the
sizes may need to be recalculated, before serialization.
*/
class RiffContainer
{
public:
    // This alignment is only made for arena based allocations.
    // For external blocks it's client code to have appropriate alignment.
    // This is needed because when reading a RiffContainer, all allocation is arena based, and
    // if the payload contains 8 byte aligned data, the overall payload needs to be 8 byte aligned.
    static const size_t kPayloadMinAlignment = 8;

    enum class Ownership
    {
        Uninitialized, ///< Doesn't contain anything
        NotOwned,      ///< It's not owned by the container
        Arena,         ///< It's owned and allocated on the arena
        Owned,         ///< It's owned, but wasn't allocated on the arena
    };

    struct Data
    {
        /// Get the payload
        void* getPayload() { return m_payload; }
        /// Get the end pointer
        void* getPayloadEnd() { return (void*)((uint8_t*)m_payload + m_size); }
        /// Get the size of the payload
        size_t getSize() const { return m_size; }
        /// Get the ownership of the data held in the payload
        Ownership getOwnership() const { return m_ownership; }

        void init()
        {
            m_ownership = Ownership::Uninitialized;
            m_size = 0;
            m_next = nullptr;
            m_payload = nullptr;
        }

        Ownership m_ownership; ///< Stores the ownership of the payload
        size_t m_size;         ///< The size of the payload
        void* m_payload;       ///< The payload
        Data* m_next;          ///< The next Data block in the list
    };

    struct Chunk;
    struct ListChunk;
    struct DataChunk;

    typedef SlangResult (*VisitorCallback)(Chunk* chunk, void* data);

    class Visitor;
    struct Chunk
    {
        enum class Kind
        {
            List, ///< Strictly speaking this can be a 'LIST' or a 'RIFF' as they have the same
                  ///< structure
            Data,
        };

        void init(Kind kind, FourCC fourCC)
        {
            m_kind = kind;
            m_fourCC = fourCC;
            m_payloadSize = 0;
            m_next = nullptr;
            m_parent = nullptr;
        }

        SlangResult visit(Visitor* visitor);
        SlangResult visitPostOrder(VisitorCallback callback, void* data);
        SlangResult visitPreOrder(VisitorCallback callback, void* data);

        /// Returns a single data chunk
        Data* getSingleData() const;

        /// Calculate the payload size
        size_t calcPayloadSize();

        Kind m_kind;          ///< Kind of chunk
        FourCC m_fourCC;      ///< The chunk type for data, or the sub type for a List (riff/list)
        size_t m_payloadSize; ///< The payload size (ie does NOT include RiffChunk header).
        Chunk* m_next;        ///< Next chunk in this list
        ListChunk* m_parent;  ///< The chunk this belongs to
    };

    struct ListChunk : public Chunk
    {
        typedef Chunk Super;
        SLANG_FORCE_INLINE static bool isType(const Chunk* chunk)
        {
            return chunk->m_kind == Kind::List;
        }

        void init(FourCC subType)
        {
            Super::init(Kind::List, subType);
            m_containedChunks = nullptr;
            m_endChunk = nullptr;

            m_payloadSize = uint32_t(sizeof(RiffListHeader) - sizeof(RiffHeader));
        }

        /// Finds chunk (list or data) that matches type. For List/Riff, type is the subtype
        Chunk* findContained(FourCC type) const;

        void* findContainedData(FourCC type, size_t minSize) const;

        ListChunk* findContainedList(FourCC type);

        /// Finds the contained data. NOTE! Assumes that there is only as single data block, and
        /// will return nullptr if there is not
        Data* findContainedData(FourCC type) const;

        template<typename T>
        T* findContainedData(FourCC type) const
        {
            return (T*)findContainedData(type, sizeof(T));
        }

        /// Find all contained that match the type
        void findContained(FourCC type, List<ListChunk*>& out);

        /// Find all contained that match the type
        void findContained(FourCC type, List<DataChunk*>& out);

        /// Find the list (including self) that matches subtype recursively
        ListChunk* findListRec(FourCC subType);

        /// NOTE! Assumes all contained chunks have correct payload sizes
        size_t calcPayloadSize();

        /// Get the sub type
        FourCC getSubType() const { return m_fourCC; }

        /// A singly linked list of contained chunks directly contained in this chunk
        Chunk* getFirstContainedChunk() const { return m_containedChunks; }

        Chunk* m_containedChunks; ///< The contained chunks
        Chunk* m_endChunk;        ///< The last chunk (only set when pushed, and used when popped)
    };

    struct DataChunk : public Chunk
    {
        typedef Chunk Super;
        SLANG_FORCE_INLINE static bool isType(const Chunk* chunk)
        {
            return chunk->m_kind == Kind::Data;
        }

        /// Calculate a hash (not necessarily very fast)
        RiffHashCode calcHash() const;
        /// Calculate the payload size
        size_t calcPayloadSize() const;

        /// Copy the payload to dst. Dst must be at least the payload size.
        void getPayload(void* dst) const;

        /// True if payloads contents is equal to data
        bool isEqual(const void* data, size_t count) const;

        /// Get single data payload.
        Data* getSingleData() const;

        /// Return as read helper
        RiffReadHelper asReadHelper() const;

        void init(FourCC fourCC)
        {
            Super::init(Kind::Data, fourCC);
            m_dataList = nullptr;
            m_endData = nullptr;
        }

        Data* m_dataList; ///< List of 0 or more data items
        Data* m_endData;  ///< The last data point
    };

    class ScopeChunk
    {
    public:
        ScopeChunk(RiffContainer* container, Chunk::Kind kind, FourCC fourCC)
            : m_container(container)
        {
            container->startChunk(kind, fourCC);
        }
        ~ScopeChunk() { m_container->endChunk(); }

    private:
        RiffContainer* m_container;
    };

    class Visitor
    {
    public:
        virtual SlangResult enterList(ListChunk* list) = 0;
        virtual SlangResult handleData(DataChunk* data) = 0;
        virtual SlangResult leaveList(ListChunk* list) = 0;
    };


    /// Add a complete data chunk
    void addDataChunk(FourCC dataFourCC, const void* data, size_t dataSizeInBytes);

    /// Start a chunk
    void startChunk(Chunk::Kind kind, FourCC type);

    /// Write data into a chunk (can only be inside a Kind::Data)
    void write(const void* data, size_t size);

    /// Adds an empty data block
    Data* addData();
    /// Set the payload on a data. Payload can be passed as nullptr, if it is no memory will be
    /// copied.
    void setPayload(Data* data, const void* payload, size_t size);

    /// Move ownership to.
    /// NOTE! The payload *must* be deallocatable via 'free'
    void moveOwned(Data* data, void* payload, size_t size);
    /// Move unowned. The payload scope must last longer than the RiffContainer
    void setUnowned(Data* data, void* payload, size_t size);

    /// End a chunk
    void endChunk();

    /// Get the root
    ListChunk* getRoot() const { return m_rootList; }

    /// Get the current chunk
    Chunk* getCurrentChunk()
    {
        return m_dataChunk ? static_cast<Chunk*>(m_dataChunk) : static_cast<Chunk*>(m_listChunk);
    }

    /// Reset the container
    void reset();

    /// true if has a root container, and nothing remains open
    bool isFullyConstructed()
    {
        return m_rootList && m_listChunk == nullptr && m_dataChunk == nullptr;
    }

    /// Makes a data chunk contain a single contiguous data block
    Data* makeSingleData(DataChunk* dataChunk);

    /// Get the memory arena that is backing the storage of data
    MemoryArena& getMemoryArena() { return m_arena; }

    /// The if the list and sublists appear correct
    static bool isChunkOk(Chunk* chunk);

    /// Traverses over chunk hierarchy and sets the sizes
    static void calcAndSetSize(Chunk* chunk);

    /// Ctor
    RiffContainer();

    void setCurrentChunk(Chunk* chunk);

protected:
    void _addChunk(Chunk* chunk);
    ListChunk* _newListChunk(FourCC subType);
    DataChunk* _newDataChunk(FourCC type);

    ListChunk* m_rootList; ///< Root list

    ListChunk* m_listChunk;
    DataChunk* m_dataChunk;

    MemoryArena m_arena; ///< Can be used to use other owned blocks
};

// -----------------------------------------------------------------------------
template<typename T>
T* as(RiffContainer::Chunk* chunk)
{
    return chunk && T::isType(chunk) ? static_cast<T*>(chunk) : nullptr;
}
// -----------------------------------------------------------------------------
template<typename T>
T* as(RiffContainer::Chunk* chunk, FourCC fourCC)
{
    return chunk && chunk->m_fourCC == fourCC && T::isType(chunk) ? static_cast<T*>(chunk)
                                                                  : nullptr;
}

struct RiffUtil
{
    typedef RiffContainer::Chunk Chunk;
    typedef RiffContainer::ListChunk ListChunk;
    typedef RiffContainer::DataChunk DataChunk;

    static int64_t calcChunkTotalSize(const RiffHeader& chunk);

    static SlangResult skip(const RiffHeader& chunk, Stream* stream, int64_t* remainingBytesInOut);

    static SlangResult readChunk(Stream* stream, RiffHeader& outChunk);

    static SlangResult writeData(
        const RiffHeader* header,
        size_t headerSize,
        const void* payload,
        size_t payloadSize,
        Stream* out);
    static SlangResult readData(
        Stream* stream,
        RiffHeader* outHeader,
        size_t headerSize,
        List<uint8_t>& data);

    static SlangResult readPayload(Stream* stream, size_t size, void* outData, size_t& outReadSize);

    /// Read a header. Handles special case of list/riff types
    static SlangResult readHeader(Stream* stream, RiffListHeader& outHeader);

    /// True if the type is a container type
    static bool isListType(FourCC type)
    {
        return type == RiffFourCC::kRiff || type == RiffFourCC::kList;
    }

    /// Dump the chunk structure
    static void dump(Chunk* chunk, WriterHelper writer);

    /// Get the size taking into account padding
    static size_t getPadSize(size_t in) { return (in + kRiffPadMask) & ~size_t(kRiffPadMask); }

    /// Write a chunk list and contents to a stream
    static SlangResult write(ListChunk* listChunk, bool isRoot, Stream* stream);
    /// Write a container to the stream
    static SlangResult write(RiffContainer* container, Stream* stream);

    /// Read the stream into the container
    static SlangResult read(Stream* stream, RiffContainer& outContainer);
};

} // namespace Slang

#endif
