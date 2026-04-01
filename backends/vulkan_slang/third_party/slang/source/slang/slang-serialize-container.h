// slang-serialize-container.h
#ifndef SLANG_SERIALIZE_CONTAINER_H
#define SLANG_SERIALIZE_CONTAINER_H

#include "../core/slang-riff.h"
#include "slang-ir-insts.h"
#include "slang-profile.h"
#include "slang-serialize-types.h"

namespace Slang
{

class EndToEndCompileRequest;


struct SerialContainerUtil
{
    struct WriteOptions
    {
        SerialOptionFlags optionFlags =
            SerialOptionFlag::ASTModule |
            SerialOptionFlag::IRModule; ///< Flags controlling what is written
        SourceManager* sourceManager =
            nullptr; ///< The source manager used for the SourceLoc in the input
    };

    struct ReadOptions
    {
        Session* session = nullptr;
        SourceManager* sourceManager = nullptr;
        NamePool* namePool = nullptr;
        SharedASTBuilder* sharedASTBuilder = nullptr;
        ASTBuilder* astBuilder =
            nullptr; // Optional. If not provided will create one in SerialContainerData.
        Linkage* linkage = nullptr;
        DiagnosticSink* sink = nullptr;
        bool readHeaderOnly = false;
        String modulePath;
    };

    /// Verify IR serialization
    static SlangResult verifyIRSerialize(
        IRModule* module,
        Session* session,
        const WriteOptions& options);

    /// Write the request to the stream
    static SlangResult write(
        FrontEndCompileRequest* frontEndReq,
        const WriteOptions& options,
        Stream* stream);
    static SlangResult write(
        EndToEndCompileRequest* request,
        const WriteOptions& options,
        Stream* stream);
    static SlangResult write(Module* module, const WriteOptions& options, Stream* stream);
};


struct ChunkRef
{
public:
    ChunkRef(RiffContainer::Chunk* chunk)
        : _chunk(chunk)
    {
    }

    RiffContainer::Chunk* ptr() const { return _chunk; }

protected:
    RiffContainer::Chunk* _chunk = nullptr;
};

struct DataChunkRef : ChunkRef
{
public:
    DataChunkRef(RiffContainer::DataChunk* chunk)
        : ChunkRef(chunk)
    {
    }

    RiffContainer::DataChunk* ptr() const { return static_cast<RiffContainer::DataChunk*>(_chunk); }

    operator RiffContainer::DataChunk*() const { return ptr(); }
};


template<typename T>
struct ChunkRefList
{
public:
    struct Iterator
    {
    public:
        Iterator(RiffContainer::Chunk* chunk)
            : _chunk(chunk)
        {
        }

        bool operator!=(Iterator const& other) const { return _chunk != other._chunk; }

        void operator++() { _chunk = _chunk->m_next; }

        T operator*()
        {
            ChunkRef ref(_chunk);
            return *(T*)&ref;
        }

    private:
        RiffContainer::Chunk* _chunk = nullptr;
    };

    Iterator begin() const { return _list ? _list->getFirstContainedChunk() : nullptr; }
    Iterator end() const { return Iterator(nullptr); }

    Count getCount()
    {
        Count count = 0;
        for (auto i : *this)
            count++;
        return count;
    }

    T getFirst() { return *begin(); }

    ChunkRefList() {}

    ChunkRefList(RiffContainer::ListChunk* list)
        : _list(list)
    {
    }

    operator RiffContainer::ListChunk*() const { return _list; }

private:
    RiffContainer::ListChunk* _list = nullptr;
};

struct ListChunkRef : ChunkRef
{
public:
    ListChunkRef(RiffContainer::Chunk* chunk)
        : ChunkRef(chunk)
    {
    }

    RiffContainer::ListChunk* ptr() const { return static_cast<RiffContainer::ListChunk*>(_chunk); }

    operator RiffContainer::ListChunk*() const { return ptr(); }
};


struct StringChunkRef : DataChunkRef
{
public:
    String getValue();
};

struct IRModuleChunkRef : ListChunkRef
{
public:
    explicit IRModuleChunkRef(RiffContainer::ListChunk* chunk)
        : ListChunkRef(chunk)
    {
    }
};

struct ASTModuleChunkRef : ListChunkRef
{
public:
    explicit ASTModuleChunkRef(RiffContainer::ListChunk* chunk)
        : ListChunkRef(chunk)
    {
    }
};

struct ModuleChunkRef : ListChunkRef
{
public:
    static ModuleChunkRef find(RiffContainer* container);

    String getName();

    IRModuleChunkRef findIR();
    ASTModuleChunkRef findAST();

    SHA1::Digest getDigest();

    ChunkRefList<StringChunkRef> getFileDependencies();

protected:
    ModuleChunkRef(RiffContainer::Chunk* chunk)
        : ListChunkRef(chunk)
    {
    }
};

struct EntryPointChunkRef : ListChunkRef
{
public:
    String getMangledName() const;
    String getName() const;
    Profile getProfile() const;

protected:
    EntryPointChunkRef(RiffContainer::Chunk* chunk)
        : ListChunkRef(chunk)
    {
    }
};

struct ContainerChunkRef : ListChunkRef
{
public:
    static ContainerChunkRef find(RiffContainer* container);

    ChunkRefList<ModuleChunkRef> getModules();

    ChunkRefList<EntryPointChunkRef> getEntryPoints();

protected:
    ContainerChunkRef(RiffContainer::Chunk* chunk)
        : ListChunkRef(chunk)
    {
    }
};

/// Attempt to find a debug-info chunk relative to
/// the given `startingChunk`.
///
RiffContainer::ListChunk* findDebugChunk(RiffContainer::Chunk* startingChunk);

SlangResult readSourceLocationsFromDebugChunk(
    RiffContainer::ListChunk* debugChunk,
    SourceManager* sourceManager,
    RefPtr<SerialSourceLocReader>& outReader);

SlangResult decodeModuleIR(
    RefPtr<IRModule>& outIRModule,
    RiffContainer::Chunk* chunk,
    Session* session,
    SerialSourceLocReader* sourceLocReader);

} // namespace Slang

#endif
